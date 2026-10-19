"""
从健康受试者子集实验（experiments/healthy_subset*.py）留下的 M4Fed 领域模型权重
（<results_dir>/m4fed/<node>_swin.pt）重新推理各节点测试集，导出逐样本的
「真实年龄 / 预测年龄」CSV 与多节点散点图。不重新训练。

与 experiments/export_test_predictions.py 的区别（模型加载 / 推理 / 作图函数直接复用它的）：
  - 节点划分不是按子数据集 ID 重建，而是严格按 <results_dir>/healthy_split_assignment.csv
    逐行还原训练时的「受试者级 4 等分」（角色内受试者顺序、受试者内扫描顺序都与 CSV 一致），
    再用与训练时相同的 build_node_datasets_from_parts（同 train_split / seed）切出测试集。
    这样即使 data/known_unusable_samples.csv 在训练后又新增了条目，也不会让划分漂移。
  - 若 <results_dir>/m4fed/age_zscore_stats.json 存在（年龄标签 z-score 实验），按各节点的
    (mean, std) 把模型输出反变换回原始年龄：age_pred = z * std + mean，并额外保留 age_pred_z 列。

输出：
  <results_dir>/m4fed/<node>_test_predictions.csv     单节点逐样本预测
  <results_dir>/m4fed/test_predictions.csv            全部节点合并
  <results_dir>/m4fed/m4fed_test_scatter.<fmt>        多节点散点图（颜色/形状区分节点）
  <results_dir>/test_predictions_metrics_check.csv    重算指标 vs 训练时保存的指标
  可选 --compare_with：<results_dir>/figures/test_scatter_compare.<fmt>（坐标轴统一的并排对比图）

用法（复现 healthy_subset_registered_133 的测试集结果）：
  python experiments/export_healthy_subset_predictions.py --config configs/m4fed.yaml \
      --results_dir results/healthy_subset_registered_133
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.env_utils import load_local_env
load_local_env()

from configs import load_config
from utils.seed_utils import set_seed
from utils.io_utils import ensure_dir, save_csv
from data.openmind_dataset import make_openmind_sample_loader
from evaluation.metrics import compute_domain_metrics
from evaluation.visualization import _setup_font, save_figure
from experiments.healthy_subset import (
    _load_all_ds_ids, _is_healthy, build_node_datasets_from_parts,
)
from experiments.export_test_predictions import (
    NODE_ORDER, load_swin_from_checkpoint, predict_node, plot_method_scatter,
    _draw_scatter, _axis_limits,
)

import matplotlib.pyplot as plt

# healthy_split_assignment.csv 中的角色名（顺序即 parts 下标）
ROLE_NAMES = ["client_0_private", "client_1_private", "server_private", "server_public"]
# 推理节点 → build_node_datasets_from_parts 返回字典中的键
NODE_KEYS = {"client_0": "client_0", "client_1": "client_1", "server": "server_private"}


def parse_args():
    p = argparse.ArgumentParser(description="健康子集 M4Fed：从 checkpoint 导出测试集逐样本预测与散点图")
    p.add_argument("--config", default="configs/m4fed.yaml")   # 必须与训练时一致（train_split / seed / mri_shape）
    p.add_argument("--results_dir", required=True)             # 训练时的 --results_dir（含 healthy_split_assignment.csv）
    p.add_argument("--no_register", action="store_true",       # 未配准的 healthy_subset.py 结果需加此开关
                   help="训练时未配准到标准模板（experiments/healthy_subset.py）时指定")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--fmt", default=None, help="图表格式，默认取 cfg.eval.figure_format")
    p.add_argument("--compare_with", nargs="*", default=[],
                   help="其他已导出的结果目录，与本目录并排画对比散点图")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  按 healthy_split_assignment.csv 还原划分                                      #
# --------------------------------------------------------------------------- #

def rebuild_parts(cfg, results_dir: str) -> list:
    """
    返回与训练时完全相同的 parts：[[(subject, [samples]), ...] × 4]。

    样本字典用当前的 sample_loader 重新生成（含 hf_image_path 等加载信息），按 unique_id
    对齐到 CSV；CSV 中找不到对应样本时直接报错，避免静默得到不同的测试集。
    """
    assign = pd.read_csv(os.path.join(results_dir, "healthy_split_assignment.csv"))
    sample_loader_fn = make_openmind_sample_loader(
        cfg.data.hf_dataset_id, cache_dir=cfg.data.cache_dir or None,
        max_cache_gb=cfg.data.max_cache_gb,
    )
    by_uid = {}
    for ds_id in _load_all_ds_ids(cfg):
        for s in sample_loader_fn(ds_id):
            if _is_healthy(s):
                by_uid[s["metadata"].get("unique_id")] = s
    missing = [u for u in assign["unique_id"] if u not in by_uid]
    if missing:
        raise RuntimeError(f"healthy_split_assignment.csv 中 {len(missing)} 个 unique_id 在当前元数据中"
                           f"找不到（如 {missing[:3]}），无法还原训练时的划分")

    parts = []
    for role in ROLE_NAMES:
        rows = assign[assign["role"] == role]
        groups: dict = {}   # dict 保持插入顺序 = CSV 中受试者首次出现顺序
        for subj, uid in zip(rows["subject"], rows["unique_id"]):
            groups.setdefault(subj, []).append(by_uid[uid])
        parts.append(list(groups.items()))
    print("[export] 已按 healthy_split_assignment.csv 还原划分: " + ", ".join(
        f"{r}={len(p)} 名/{sum(len(x[1]) for x in p)} 张" for r, p in zip(ROLE_NAMES, parts)))
    return parts


# --------------------------------------------------------------------------- #
#  导出                                                                          #
# --------------------------------------------------------------------------- #

def export_predictions(cfg, results_dir: str, register_to_template: bool = True,
                       batch_size: int = 4, fmt: str | None = None) -> pd.DataFrame:
    """推理 server / client_0 / client_1 测试集，保存 CSV、散点图与指标校验，返回合并后的 DataFrame。"""
    set_seed(cfg.training.seed)
    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    fmt = fmt or cfg.eval.figure_format
    mri_shape = tuple(cfg.data.mri_shape)
    m_dir = os.path.join(results_dir, "m4fed")

    stats_path = os.path.join(m_dir, "age_zscore_stats.json")
    age_norms = {}
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            age_norms = json.load(f)
        print(f"[export] 检测到 {stats_path}：模型输出为 z-score，将按节点反变换回原始年龄")

    node_datasets = build_node_datasets_from_parts(rebuild_parts(cfg, results_dir), cfg, mri_shape,
                                                   register_to_template=register_to_template)

    frames, check_rows = [], []
    for node in [n for n in NODE_ORDER if n in NODE_KEYS]:
        ckpt = os.path.join(m_dir, f"{node}_swin.pt")
        if not os.path.exists(ckpt):
            print(f"[跳过] {node}：{ckpt} 不存在")
            continue
        test_ds = node_datasets[NODE_KEYS[node]]["test"]
        print(f"\n[推理] 节点={node} 测试样本数={len(test_ds)}")
        model = load_swin_from_checkpoint(ckpt, mri_shape, cfg.federated.qlora_alpha, device)
        info_rows, preds, n_skip = predict_node({"m4fed": model}, test_ds, batch_size, device)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        df = pd.DataFrame(info_rows)
        df.insert(0, "node", node)
        df.insert(0, "method", "m4fed")
        out = np.asarray(preds["m4fed"], dtype=float)
        if node in age_norms:
            df["age_pred_z"] = out
            out = out * age_norms[node]["std"] + age_norms[node]["mean"]
        df["age_pred"] = out
        df["residual"] = df["age_pred"] - df["age_true"]
        save_csv(df, os.path.join(m_dir, f"{node}_test_predictions.csv"))
        frames.append(df)

        row = {"node": node, "n_eval": len(df), "n_skipped": n_skip}
        if len(df) >= 2:
            row.update(compute_domain_metrics(df["age_true"].to_numpy(), df["age_pred"].to_numpy()))
        saved = os.path.join(m_dir, f"{node}_metrics.csv")
        if os.path.exists(saved):
            for k, v in pd.read_csv(saved).iloc[0].items():
                row[f"saved_{k}"] = v
        check_rows.append(row)

    if not frames:
        raise SystemExit(f"{m_dir} 下没有任何 <node>_swin.pt")
    all_df = pd.concat(frames, ignore_index=True)
    save_csv(all_df, os.path.join(m_dir, "test_predictions.csv"))
    plot_method_scatter(all_df, "m4fed", os.path.join(m_dir, "m4fed_test_scatter"), fmt)
    check_df = pd.DataFrame(check_rows)
    save_csv(check_df, os.path.join(results_dir, "test_predictions_metrics_check.csv"))

    cols = [c for c in ["node", "n_eval", "n_skipped", "MAE", "saved_MAE", "RMSE", "saved_RMSE",
                        "Pearson_r", "saved_Pearson_r", "R2", "saved_R2"] if c in check_df]
    print("\n[校验] 重算指标 vs 训练时保存的指标：")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(check_df[cols].to_string(index=False))
    print(f"[完成] 逐样本预测与散点图已保存至 {m_dir}")
    return all_df


def plot_comparison(dfs: dict, save_path: str, fmt: str) -> None:
    """{标签: test_predictions DataFrame} 并排画多节点散点图，坐标轴范围统一。"""
    _setup_font()
    lims = _axis_limits(pd.concat(list(dfs.values()), ignore_index=True))
    fig, axes = plt.subplots(1, len(dfs), figsize=(4.8 * len(dfs), 5.0), squeeze=False)
    for ax, (label, df) in zip(axes[0], dfs.items()):
        _draw_scatter(ax, df, lims, label)
    fig.tight_layout()
    save_figure(fig, save_path, fmt=fmt)
    print(f"[完成] 对比散点图已保存至 {save_path}.{fmt}")


def compare_with(results_dir: str, all_df: pd.DataFrame, other_dirs: list, fmt: str) -> None:
    """把本目录与其他已导出目录的 m4fed/test_predictions.csv 画成并排对比图。"""
    dfs = {}
    for d in other_dirs:
        p = os.path.join(d, "m4fed", "test_predictions.csv")
        if os.path.exists(p):
            dfs[os.path.basename(os.path.normpath(d))] = pd.read_csv(p)
        else:
            print(f"[对比] 跳过 {d}：{p} 不存在（先对该目录运行本脚本导出）")
    if not dfs:
        return
    dfs[os.path.basename(os.path.normpath(results_dir))] = all_df
    plot_comparison(dfs, os.path.join(ensure_dir(os.path.join(results_dir, "figures")),
                                      "test_scatter_compare"), fmt)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    fmt = args.fmt or cfg.eval.figure_format
    all_df = export_predictions(cfg, args.results_dir, register_to_template=not args.no_register,
                                batch_size=args.batch_size, fmt=fmt)
    if args.compare_with:
        compare_with(args.results_dir, all_df, args.compare_with, fmt)


if __name__ == "__main__":
    main()
