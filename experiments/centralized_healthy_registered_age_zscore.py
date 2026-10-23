"""
中心化补充实验：健康受试者子集 + 配准到标准模板 + 年龄标签 z-score，
对加载了 MONAI SwinViT 自监督预训练权重的领域模型 SwinMultiTask 做「中心化全参数微调」。

与 experiments/healthy_subset_registered_age_zscore.py（M4Fed 联邦训练）的对照关系：
  - 数据：同一健康受试者队列、同一配准流水线（register_to_template=True，
    复用 data/mri_registered_cache）。默认直接按
    results/healthy_subset_registered_133/healthy_split_assignment.csv 还原受试者级 4 等分
    （experiments/export_healthy_subset_predictions.rebuild_parts），因此：
      训练集 = client_0 / client_1 / server_private 的训练部分 + server_public 全量
      测试集 = client_0 / client_1 / server_private 的测试部分（与联邦实验三个节点的测试集
               完全相同，逐样本 CSV 的 node 列记录样本来自哪个节点的测试集，可分节点对比）
    划分 CSV 不存在时按 healthy_subset.py 的规则（同 seed）重新划分并保存。
  - 年龄标签 z-score：用「合并后的全部训练集」的年龄 mean / std 标准化训练目标，
    评估时把输出反变换回原始年龄再计算 MAE / RMSE / r / R2（口径与联邦实验一致），
    统计量保存在 centralized/age_zscore_stats.json。
  - 训练：没有联邦学习、没有 MedMD 合成数据、没有 QLoRA；所有数据在一个节点上，
    SwinTrainer(use_qlora=False) 全参数微调 --epochs 轮（默认 10）。lr / batch_size /
    混合精度 / age_loss_weight / seed 取自 --config（与联邦实验相同）。

每一轮（epoch）结束后保存：
  centralized/epochs/epoch_XX_test_predictions.csv   测试集原始逐样本预测（age_true / age_pred /
                                                       age_pred_z / residual / sex / node ...）
  centralized/epoch_metrics.csv                       每轮：train_loss、test_loss、整体 MAE/RMSE/r/R2
  centralized/epoch_node_metrics.csv                  每轮：按来源节点（client_0/client_1/server）的指标
最后一轮额外保存：
  centralized/centralized_test_scatter[_by_node].<fmt>   原始预测对配值散点图（整体 / 按来源节点着色）
  centralized/centralized_error_heatmap_age_sex.<fmt>    年龄段 × 性别 平均带符号误差（pred - true）热图，
                                                          格内标注均值、n 与单样本 t 检验显著性（* p<0.05）
  centralized/centralized_mae_heatmap_age_sex.<fmt>      年龄段 × 性别 MAE 热图
  centralized/error_heatmap_age_sex.csv                  热图对应的分组统计（mean_error / MAE / n / p 值）
  centralized/loss_curve.<fmt>                           train / test 损失曲线
  centralized/swin_final.pt                              最终 SwinMultiTask 权重（state_dict）

用法：
  python experiments/centralized_healthy_registered_age_zscore.py --config configs/m4fed.yaml \
      --results_dir results/centralized_healthy_registered_age_zscore \
      --split_dir results/healthy_subset_registered_133 --epochs 10
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.env_utils import load_local_env
load_local_env()

from configs import load_config
from utils.seed_utils import set_seed
from utils.io_utils import ensure_dir, save_csv, save_model
from utils.log_utils import setup_run_logger
from data.openmind_dataset import (
    OpenMindDataset, make_dataloader, make_openmind_sample_loader, _collate_fn,
)
from models.swin.swin_multitask import SwinMultiTask
from training.swin_trainer import SwinTrainer
from evaluation.metrics import compute_domain_metrics
from evaluation.visualization import _setup_font, save_figure, plot_loss_curves, plot_error_heatmap
from experiments.healthy_subset import (
    _load_all_ds_ids, collect_healthy_samples, group_by_subject, even_4way_split_subjects,
    _flatten_subjects, _save_split_assignment, log_cohort_stats,
)
from experiments.export_healthy_subset_predictions import rebuild_parts
from experiments.export_test_predictions import _sample_info, plot_method_scatter, _axis_limits
from federated.methods.m4fed import _age_stats
from data.splits import train_test_split_indices

import matplotlib.pyplot as plt
import seaborn as sns

# 受试者级 4 等分的角色下标 → 测试集来源节点名（与联邦实验的节点名一致）
_TEST_NODES = {0: "client_0", 1: "client_1", 2: "server"}
_PUBLIC_PART = 3   # server_public：全量作训练，无测试集
_AGE_BIN = 10      # 偏差热图的年龄分段（年）


def parse_args():
    p = argparse.ArgumentParser(description="中心化补充实验：健康子集 + 配准 + 年龄 z-score，SwinMultiTask 全参数微调")
    p.add_argument("--config", default="configs/m4fed.yaml", help="YAML 配置文件路径（train_split / seed / lr 等）")
    p.add_argument("--results_dir", default="results/centralized_healthy_registered_age_zscore",
                   help="结果保存目录")
    p.add_argument("--split_dir", default="results/healthy_subset_registered_133",
                   help="含 healthy_split_assignment.csv 的联邦实验目录；不存在则按同 seed 重新划分")
    p.add_argument("--epochs", type=int, default=10, help="中心化训练轮数")
    p.add_argument("--batch_size", type=int, default=None, help="训练批大小，默认取 cfg.training.batch_size")
    p.add_argument("--eval_batch_size", type=int, default=4, help="测试集推理批大小")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  数据                                                                          #
# --------------------------------------------------------------------------- #

def build_parts(cfg, split_dir: str, out_dir: str) -> list:
    """优先按联邦实验的 healthy_split_assignment.csv 还原划分，保证测试集与联邦实验逐样本一致。"""
    if split_dir and os.path.exists(os.path.join(split_dir, "healthy_split_assignment.csv")):
        print(f"[centralized] 按 {split_dir}/healthy_split_assignment.csv 还原受试者级划分")
        parts = rebuild_parts(cfg, split_dir)
    else:
        print(f"[centralized] 未找到 {split_dir}/healthy_split_assignment.csv，按 healthy_subset.py 规则重新划分")
        sample_loader_fn = make_openmind_sample_loader(
            cfg.data.hf_dataset_id, cache_dir=cfg.data.cache_dir or None,
            max_cache_gb=cfg.data.max_cache_gb,
        )
        samples = collect_healthy_samples(cfg, sample_loader_fn, _load_all_ds_ids(cfg))
        parts = even_4way_split_subjects(group_by_subject(samples), seed=cfg.data.seed)
    _save_split_assignment(parts, out_dir)
    log_cohort_stats([s for part in parts for _, samps in part for s in samps], out_dir)
    return parts


def build_centralized_datasets(parts: list, cfg, mri_shape) -> tuple:
    """
    与 build_node_datasets_from_parts 相同的受试者级 train/test 划分，再把各节点合并到一个节点：
    返回 (train_ds, test_ds, test_nodes)，test_nodes[i] 为 test_ds.samples[i] 的来源节点。
    """
    ts, seed = cfg.data.train_split, cfg.data.seed
    train_samples, test_samples, test_nodes = [], [], []
    for i, part in enumerate(parts):
        if i == _PUBLIC_PART:
            train_samples.extend(_flatten_subjects(part))
            continue
        tr_idx, te_idx = train_test_split_indices(len(part), ts, seed)
        train_samples.extend(_flatten_subjects([part[j] for j in tr_idx]))
        te = _flatten_subjects([part[j] for j in te_idx])
        test_samples.extend(te)
        test_nodes.extend([_TEST_NODES[i]] * len(te))
    train_ds = OpenMindDataset(train_samples, mri_shape, register_to_template=True)
    test_ds = OpenMindDataset(test_samples, mri_shape, register_to_template=True)
    print(f"[centralized] 合并后 训练集 {len(train_ds)} 张 / 测试集 {len(test_ds)} 张 "
          f"（测试集来源：" + ", ".join(f"{n}={test_nodes.count(n)}" for n in _TEST_NODES.values()) + "）")
    return train_ds, test_ds, test_nodes


# --------------------------------------------------------------------------- #
#  评估：一次推理同时得到逐样本预测与测试损失                                         #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_test(model: SwinMultiTask, trainer: SwinTrainer, test_ds: OpenMindDataset,
                  test_nodes: list, age_norm: tuple, batch_size: int, device: str):
    """
    逐样本推理测试集。与 export_test_predictions.predict_node 一样直接丢弃不可用 / 年龄缺失的
    样本（不使用 __getitem__ 的「替换为下一个样本」），避免同一样本被重复计入。

    返回：(DataFrame 逐样本预测, 测试集平均多任务损失（与训练损失同口径，z-score 空间）, 丢弃数)
    """
    model.eval()
    mean, std = age_norm
    rows, items, infos = [], [], []
    loss_sum, n_batches, n_skipped = 0.0, 0, 0

    def _flush():
        nonlocal loss_sum, n_batches
        if not items:
            return
        batch = _collate_fn(items)
        preds = model(batch["mri"].to(device))
        targets = trainer._build_targets(batch)
        targets["age"] = (targets["age"] - mean) / std
        loss_sum += float(model.compute_loss(preds, targets).item())
        n_batches += 1
        z = preds["age"].squeeze(-1).float().cpu().reshape(-1).numpy()
        for info, zi in zip(infos, z):
            rows.append({**info, "age_pred_z": float(zi), "age_pred": float(zi * std + mean)})
        items.clear()
        infos.clear()

    for idx in range(len(test_ds)):
        s = test_ds.samples[idx]
        if s.get("_unusable"):
            n_skipped += 1
            continue
        item = test_ds[idx]
        if s.get("_unusable"):   # 本次加载才发现不可用，__getitem__ 返回的是替身样本
            n_skipped += 1
            continue
        age = float(item["age"])
        if np.isnan(age):
            n_skipped += 1
            continue
        items.append(item)
        infos.append({"node": test_nodes[idx], **_sample_info(s, idx), "age_true": age})
        if len(items) >= batch_size:
            _flush()
    _flush()

    df = pd.DataFrame(rows)
    if len(df):
        df["residual"] = df["age_pred"] - df["age_true"]
    return df, loss_sum / max(n_batches, 1), n_skipped


def _metrics(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {"MAE": float("nan"), "RMSE": float("nan"), "Pearson_r": float("nan"), "R2": float("nan")}
    return compute_domain_metrics(df["age_true"].to_numpy(), df["age_pred"].to_numpy())


# --------------------------------------------------------------------------- #
#  最终一轮的图                                                                   #
# --------------------------------------------------------------------------- #

def plot_overall_scatter(df: pd.DataFrame, m: dict, save_path: str, fmt: str) -> None:
    """全部测试样本的 真实年龄-预测年龄 散点图，标注整体指标。"""
    _setup_font()
    lims = _axis_limits(df)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(lims, lims, color="#888888", linestyle="--", linewidth=1.2, zorder=1, label="y = x")
    ax.scatter(df["age_true"], df["age_pred"], s=18, alpha=0.6, color="#2a78d6",
               edgecolors="white", linewidths=0.4, zorder=2, label=f"Test (n={len(df)})")
    ax.text(0.97, 0.03, f"MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}\n"
                        f"r={m['Pearson_r']:.3f}  R²={m['R2']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"))
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("True Age", fontsize=11)
    ax.set_ylabel("Predicted Age", fontsize=11)
    ax.set_title("Centralized SwinMTL", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    fig.tight_layout()
    save_figure(fig, save_path, fmt=fmt)


def age_sex_bias_table(df: pd.DataFrame) -> pd.DataFrame:
    """按 (年龄段, 性别) 统计带符号误差均值、MAE、n 与单样本 t 检验 p 值（H0: 平均误差 = 0）。"""
    from scipy.stats import ttest_1samp
    d = df[df["sex"].notna()].copy()
    d["sex"] = d["sex"].astype(str).str.strip().str.lower()
    d["age_bin"] = (d["age_true"] // _AGE_BIN * _AGE_BIN).astype(int)
    rows = []
    for (b, sex), g in d.groupby(["age_bin", "sex"]):
        err = g["residual"].to_numpy()
        p = float(ttest_1samp(err, 0.0).pvalue) if len(err) >= 3 else float("nan")
        rows.append({"age_bin": f"{b}-{b + _AGE_BIN}", "age_bin_start": b, "sex": sex, "n": len(err),
                     "mean_error": float(err.mean()), "std_error": float(err.std()),
                     "MAE": float(np.abs(err).mean()), "p_value": p})
    return pd.DataFrame(rows).sort_values(["age_bin_start", "sex"]).reset_index(drop=True)


def plot_bias_heatmap(table: pd.DataFrame, save_path: str, fmt: str) -> None:
    """年龄段 × 性别 平均带符号误差热图（发散色系以 0 为中心：红=高估，蓝=低估）。"""
    _setup_font()
    order = table.drop_duplicates("age_bin").sort_values("age_bin_start")["age_bin"].tolist()
    mean = table.pivot(index="age_bin", columns="sex", values="mean_error").reindex(order)
    n = table.pivot(index="age_bin", columns="sex", values="n").reindex(order)
    p = table.pivot(index="age_bin", columns="sex", values="p_value").reindex(order)
    annot = mean.copy().astype(object)
    for r in mean.index:
        for c in mean.columns:
            if pd.isna(mean.loc[r, c]):
                annot.loc[r, c] = ""
            else:
                star = "*" if pd.notna(p.loc[r, c]) and p.loc[r, c] < 0.05 else ""
                annot.loc[r, c] = f"{mean.loc[r, c]:+.2f}{star}\n(n={int(n.loc[r, c])})"
    vmax = float(np.nanmax(np.abs(mean.to_numpy()))) if mean.notna().any().any() else 1.0
    fig, ax = plt.subplots(figsize=(max(5, mean.shape[1] * 1.8), max(4, mean.shape[0] * 0.6)))
    sns.heatmap(mean, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, annot=annot, fmt="",
                linewidths=0.5, cbar_kws={"label": "Mean error, Pred − True (years)"},
                annot_kws={"fontsize": 8})
    ax.set_xlabel("Sex", fontsize=11)
    ax.set_ylabel("Age Bin", fontsize=11)
    ax.set_title("Error heatmap (age × sex), * p<0.05", fontsize=11)
    fig.tight_layout()
    save_figure(fig, save_path, fmt=fmt)


def save_final_figures(df: pd.DataFrame, overall: dict, history: list, out: str, fmt: str) -> None:
    plot_overall_scatter(df, overall, os.path.join(out, "centralized_test_scatter"), fmt)
    plot_method_scatter(df, "Centralized SwinMTL (by source node)",
                        os.path.join(out, "centralized_test_scatter_by_node"), fmt)

    table = age_sex_bias_table(df)
    if len(table):
        save_csv(table, os.path.join(out, "error_heatmap_age_sex.csv"))
        plot_bias_heatmap(table, os.path.join(out, "centralized_error_heatmap_age_sex"), fmt)
        order = table.drop_duplicates("age_bin").sort_values("age_bin_start")["age_bin"].tolist()
        plot_error_heatmap(table.pivot(index="age_bin", columns="sex", values="MAE").reindex(order),
                           xlabel="Sex", ylabel="Age Bin",
                           save_path=os.path.join(out, "centralized_mae_heatmap_age_sex"), fmt=fmt)
        sig = table[(table["p_value"] < 0.05)]
        print("[centralized] 年龄段 × 性别 带符号误差（pred - true）：")
        with pd.option_context("display.width", 200):
            print(table[["age_bin", "sex", "n", "mean_error", "MAE", "p_value"]].to_string(index=False))
        print(f"[centralized] 平均误差显著偏离 0（p<0.05）的分组：{len(sig)}/{len(table)}")
    else:
        print("[centralized] 测试集无性别信息，跳过年龄 × 性别误差热图")

    plot_loss_curves({"Train": {"mean": [h["train_loss"] for h in history]},
                      "Test": {"mean": [h["test_loss"] for h in history]}},
                     xlabel="Epoch", ylabel="Multi-task loss (z-scored age)",
                     save_path=os.path.join(out, "loss_curve"), fmt=fmt)


# --------------------------------------------------------------------------- #
#  主流程                                                                        #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    log_path = setup_run_logger(args.results_dir)
    print(f"[centralized] 终端日志将同步保存至 {log_path}")
    cfg = load_config(args.config)
    out_root = ensure_dir(args.results_dir)
    out = ensure_dir(os.path.join(out_root, "centralized"))
    ep_dir = ensure_dir(os.path.join(out, "epochs"))
    fmt = cfg.eval.figure_format
    mri_shape = tuple(cfg.data.mri_shape)
    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or cfg.training.batch_size
    set_seed(cfg.training.seed)

    # ---- 1. 数据：健康子集、受试者级划分、配准；各节点合并为一个训练集 / 一个测试集 ----
    parts = build_parts(cfg, args.split_dir, out_root)
    train_ds, test_ds, test_nodes = build_centralized_datasets(parts, cfg, mri_shape)
    train_loader = make_dataloader(train_ds, batch_size, shuffle=True,
                                   num_workers=cfg.training.num_workers, seed=cfg.training.seed)

    # ---- 2. 年龄标签 z-score：合并训练集的统计量 ----
    mean, std, n_age = _age_stats(train_ds.samples)
    age_norm = (mean, std)
    with open(os.path.join(out, "age_zscore_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"centralized": {"mean": mean, "std": std, "n": n_age}}, f, ensure_ascii=False, indent=2)
    print(f"[centralized] 年龄 z-score：mean={mean:.3f} std={std:.3f} (n={n_age})")

    # ---- 3. 模型：MONAI SwinViT SSL 预训练权重 + 多任务头，全参数微调 ----
    task_specs = {
        "age":           {"type": "regression",     "n_classes": 1},
        "sex":           {"type": "classification", "n_classes": 2},
        "health_status": {"type": "classification", "n_classes": 2},
        "race":          {"type": "classification", "n_classes": 5},
    }
    loader_fn = make_openmind_sample_loader(cfg.data.hf_dataset_id, cache_dir=cfg.data.cache_dir or None,
                                            max_cache_gb=cfg.data.max_cache_gb)
    for field, n in getattr(loader_fn, "n_classes", {}).items():
        if field in task_specs and n > 0:
            task_specs[field]["n_classes"] = n
    set_seed(cfg.training.seed)
    model = SwinMultiTask(
        pretrain_url=cfg.model.swinunetr_pretrain_url,
        img_size=mri_shape,
        feature_size=cfg.model.swinunetr_feature_size,
        task_specs=task_specs,
        device=device,
        age_loss_weight=cfg.training.age_loss_weight,
    ).to(device)
    trainer = SwinTrainer(model, device=device, lr=cfg.training.lr_swin,
                          mixed_precision=cfg.training.mixed_precision,
                          use_qlora=False, age_norm=age_norm)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[centralized] 设备={device} epochs={args.epochs} batch_size={batch_size} "
          f"lr={cfg.training.lr_swin} 可训练参数={n_param / 1e6:.2f}M "
          f"预训练权重={cfg.model.swinunetr_pretrain_url or '<随机初始化>'}")

    # ---- 4. 训练：每轮结束评估测试集并落盘 ----
    history, node_history = [], []
    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = trainer.train_epoch(train_loader)
        t_train = time.perf_counter() - t0
        df, test_loss, n_skip = evaluate_test(model, trainer, test_ds, test_nodes, age_norm,
                                              args.eval_batch_size, device)
        df.insert(0, "epoch", ep)
        save_csv(df, os.path.join(ep_dir, f"epoch_{ep:02d}_test_predictions.csv"))

        overall = _metrics(df)
        history.append({"epoch": ep, "train_loss": train_loss, "test_loss": test_loss,
                        "n_train": len(train_ds), "n_eval": len(df), "n_skipped": n_skip,
                        **overall, "train_seconds": t_train,
                        "eval_seconds": time.perf_counter() - t0 - t_train})
        for node in _TEST_NODES.values():
            d = df[df["node"] == node]
            node_history.append({"epoch": ep, "node": node, "n_eval": len(d), **_metrics(d)})
        save_csv(pd.DataFrame(history), os.path.join(out, "epoch_metrics.csv"))
        save_csv(pd.DataFrame(node_history), os.path.join(out, "epoch_node_metrics.csv"))
        print(f"[centralized] epoch {ep}/{args.epochs} train_loss={train_loss:.4f} "
              f"test_loss={test_loss:.4f} MAE={overall['MAE']:.3f} RMSE={overall['RMSE']:.3f} "
              f"r={overall['Pearson_r']:.3f} R2={overall['R2']:.3f} "
              f"(n={len(df)}, 跳过 {n_skip}, 训练 {t_train / 60:.1f} min)")

    # ---- 5. 最终一轮：散点图 / 年龄 × 性别误差热图 / 损失曲线 / 模型权重 ----
    save_final_figures(df, overall, history, out, fmt)
    save_model(model, os.path.join(out, "swin_final.pt"))
    print(f"[centralized] 最终模型已保存至 {out}/swin_final.pt")

    from utils.data_quality_utils import save_unusable_csv
    save_unusable_csv(os.path.join(out_root, "unusable_samples.csv"))
    print(f"[centralized] 全部结果已保存至 {out_root}")


if __name__ == "__main__":
    main()
