"""
从 run_all（experiments/run_all_baselines.py）留下的领域模型权重（<method>/<node>_swin.pt）
重新推理各节点测试集，补全逐样本的「真实年龄 / 预测年龄」配对 CSV 与综合散点图。

背景：
  run_all_baselines 只保存了各方法、各节点的汇总指标（MAE/RMSE/Pearson_r/R2），
  没有保存逐样本的预测值。本脚本不重新训练，只做：
    1. 用与 run_all 完全相同的配置/种子重建节点划分（M4Fed：服务器 + 2 客户端；
       FedAvg/FedProx/FAN/GPT-FL：3 客户端），取各节点的私有测试集
    2. 从 checkpoint 重建 SwinMultiTask 并加载权重（任务头维度、feature_size 均从
       checkpoint 推断；M4Fed 客户端是 QLoRA 微调后保存的 PEFT 包装权重，
       先把 LoRA 增量合并回基础权重 W = W0 + (alpha/r)·B·A 再严格加载）
    3. 逐节点推理一次数据，同一节点上的所有方法模型共享这次数据读取

输出（均写入 --results_dir，默认 results/main）：
  <method>/<node>_test_predictions.csv   ：单节点逐样本预测
  <method>/test_predictions.csv          ：该方法全部节点合并
  <method>/<method>_test_scatter.<fmt>   ：该方法全部节点的综合散点图（颜色/形状区分节点）
  test_predictions_all_methods.csv       ：全部方法、全部节点合并
  test_predictions_metrics_check.csv     ：重算指标 vs run_all 保存的指标（校验划分与权重加载是否一致）
  figures/test_scatter_all_methods.<fmt> ：各方法散点图并排（坐标轴范围统一，便于横向对比）

与原评估的差异：
  OpenMindDataset.__getitem__ 遇到不可用样本时会静默替换为下一个可用样本，
  原评估因此可能把同一个样本算两次。本脚本直接丢弃不可用样本（不替换），
  丢弃数记录在 metrics_check 的 n_skipped 列，所以重算指标可能与保存值有微小差异。

用法：
  python experiments/export_test_predictions.py --config configs/m4fed.yaml --results_dir results/main
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import load_config
from utils.seed_utils import set_seed
from utils.io_utils import ensure_dir, save_csv
from data.openmind_dataset import build_node_datasets, make_openmind_sample_loader, _collate_fn
from models.swin.swin_multitask import SwinMultiTask, _Backbone3DCNN, _TaskHead
from evaluation.metrics import compute_domain_metrics
from evaluation.visualization import _setup_font, save_figure
from experiments.run_all_baselines import (
    default_task_specs, _load_openmind_ds_ids, _load_ds_quality_stats,
)

import matplotlib.pyplot as plt

# 方法顺序与 run_all_baselines 一致；显示名称与论文图表一致
METHODS = ["fedavg", "fedprox", "fan", "gptfl", "m4fed"]
DISPLAY_NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "fan": "FAN",
                 "gptfl": "GPT-FL", "m4fed": "M4Fed"}

# 节点颜色/形状固定绑定节点名（不随方法、节点数变化而重排），形状作为颜色之外的第二编码
NODE_ORDER = ["client_0", "client_1", "client_2", "server"]
NODE_COLORS = {"client_0": "#2a78d6", "client_1": "#eb6834",
               "client_2": "#1baf7a", "server": "#eda100"}
NODE_MARKERS = {"client_0": "o", "client_1": "s", "client_2": "^", "server": "D"}
NODE_LABELS = {"client_0": "Client 0", "client_1": "Client 1",
               "client_2": "Client 2", "server": "Server"}

_NODE_RE = re.compile(r"^(server|client_\d+)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m4fed.yaml")   # 必须与 run_all 时的配置一致
    p.add_argument("--results_dir", default="results/main")     # run_all 的 --results_dir
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--batch_size", type=int, default=4)         # 推理批大小（无梯度，可比训练时大）
    p.add_argument("--lora_alpha", type=float, default=None,    # 默认取 cfg.federated.qlora_alpha
                   help="M4Fed 客户端 QLoRA 的 lora_alpha（合并 LoRA 时的缩放 alpha/r）")
    p.add_argument("--fmt", default=None, help="图表格式，默认取 cfg.eval.figure_format")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  checkpoint → SwinMultiTask                                                   #
# --------------------------------------------------------------------------- #

def _merge_lora_state(state: dict, lora_alpha: float):
    """
    把 PEFT 包装后保存的 state_dict 还原为普通 SwinMultiTask 的 state_dict。

    PEFT 键名形如：
      backbone.base_model.model.<path>.base_layer.weight   ← 冻结的基础权重 W0
      backbone.base_model.model.<path>.lora_A.default.weight  (r, in)
      backbone.base_model.model.<path>.lora_B.default.weight  (out, r)
    合并：W = W0 + (lora_alpha / r) · B @ A（与 peft 的 merge 一致，bias="none"、非 rslora）
    注意 PEFT 按名称后缀匹配目标层，"proj" 同时命中了 patch_embed.proj（Conv3d）：
    卷积层 lora_A 为 (r, in, k, k, k)、lora_B 为 (out, r, 1, 1, 1)，增量为
    B.flatten(1) @ A.flatten(1) 再 reshape 成 W0 形状；线性层是同一公式的特例。

    返回：(plain_state, 合并的 LoRA 层数)；不含 PEFT 键时原样返回、层数为 0
    """
    if not any("base_model.model." in k for k in state):
        return state, 0
    plain, lora_a, lora_b = {}, {}, {}
    for k, v in state.items():
        k = k.replace("base_model.model.", "")
        if ".lora_A." in k:
            lora_a[k.split(".lora_A.")[0]] = v
        elif ".lora_B." in k:
            lora_b[k.split(".lora_B.")[0]] = v
        elif ".lora_" in k:
            continue   # 其他 LoRA 附属参数（本项目不会出现，保险起见跳过）
        else:
            plain[k.replace(".base_layer.", ".")] = v
    for mod, a in lora_a.items():
        b = lora_b[mod]
        w = plain[f"{mod}.weight"]
        scaling = lora_alpha / a.shape[0]
        delta = (b.float().flatten(1) @ a.float().flatten(1)).reshape(w.shape)
        plain[f"{mod}.weight"] = (w.float() + scaling * delta).to(w.dtype)
    return plain, len(lora_a)


def load_swin_from_checkpoint(path: str, mri_shape: tuple, lora_alpha: float, device: str):
    """
    从 save_model 保存的 state_dict 重建 SwinMultiTask 并严格加载权重。

    任务头列表与类别数、embed_dim（→ feature_size = embed_dim / 16）均从 checkpoint 推断，
    不依赖重建时的数据统计；骨干随机初始化后被权重完全覆盖（不触发预训练权重下载）。
    """
    state = torch.load(path, map_location="cpu")
    state, n_lora = _merge_lora_state(state, lora_alpha)

    defaults = default_task_specs()
    fields = sorted({k.split(".")[1] for k in state if k.startswith("heads.")})
    task_specs = {}
    for f in fields:
        out_dim = state[f"heads.{f}.head.2.weight"].shape[0]
        ttype = defaults.get(f, {}).get("type", "regression" if out_dim == 1 else "classification")
        task_specs[f] = {"type": ttype, "n_classes": out_dim}
    embed_dim = state[f"heads.{fields[0]}.head.0.weight"].shape[1]

    model = SwinMultiTask(pretrain_url="", img_size=mri_shape, feature_size=embed_dim // 16,
                          task_specs=task_specs, device="cpu", random_init=True)
    if not any(k.startswith("backbone.swin_vit.") for k in state):
        # 训练时 SwinUNETR 构建失败退回了 3D CNN 骨干：按 checkpoint 的结构重建
        model.backbone = _Backbone3DCNN(embed_dim=embed_dim)
        model.embed_dim = embed_dim
        model.heads = nn.ModuleDict({
            f: _TaskHead(embed_dim, 1 if s["type"] == "regression" else s["n_classes"], s["type"])
            for f, s in task_specs.items()
        })
    model.load_state_dict(state, strict=True)   # 严格加载：任何键不匹配都直接报错，避免静默用随机权重推理
    print(f"  [加载] {path}（LoRA 合并层数={n_lora}，任务头={list(task_specs)}）")
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
#  推理                                                                          #
# --------------------------------------------------------------------------- #

def _sample_info(s: dict, idx: int) -> dict:
    """样本标识信息（写入 CSV，便于事后与元数据对齐或做偏差分析）。"""
    meta = s.get("metadata", {})
    get = meta.get if hasattr(meta, "get") else (lambda k, d=None: d)
    uid = get("unique_id", None)
    ds_id = get("ds_id", None) or (uid.split("__")[0] if isinstance(uid, str) else None)
    return {
        "sample_idx": idx,                     # 在该节点测试集中的下标（test_loader 顺序）
        "ds_id": ds_id,
        "unique_id": uid,
        "image_path": s.get("hf_image_path") or s.get("mri_path"),
        "sex": get("sex", None),
        "health_status": get("health_status", None),
    }


@torch.no_grad()
def predict_node(models: dict, test_ds, batch_size: int, device: str):
    """
    对一个节点的测试集推理一次，把每个批次同时送入该节点的所有方法模型。

    返回：
      info_rows : 每个有效样本的标识 + age_true
      preds     : {method: [age_pred, ...]}（与 info_rows 一一对应）
      n_skipped : 被丢弃的样本数（不可用 / 年龄缺失）
    """
    info_rows, preds = [], {m: [] for m in models}
    items, infos = [], []
    n_skipped = 0

    def _flush():
        if not items:
            return
        mri = _collate_fn(items)["mri"].to(device)
        for m, model in models.items():
            preds[m].extend(model(mri)["age"].squeeze(-1).float().cpu().reshape(-1).tolist())
        info_rows.extend(infos)
        items.clear()
        infos.clear()

    for idx in range(len(test_ds)):
        s = test_ds.samples[idx]
        if s.get("_unusable"):
            n_skipped += 1
            continue
        item = test_ds[idx]
        # __getitem__ 加载失败时会把样本标记为不可用并返回下一个样本的替身，丢弃之
        if s.get("_unusable"):
            n_skipped += 1
            continue
        age = float(item["age"])
        if np.isnan(age):
            n_skipped += 1
            continue
        items.append(item)
        infos.append({**_sample_info(s, idx), "age_true": age})
        if len(items) >= batch_size:
            _flush()
        if (idx + 1) % 50 == 0:
            print(f"    已处理 {idx + 1}/{len(test_ds)}")
    _flush()
    return info_rows, preds, n_skipped


# --------------------------------------------------------------------------- #
#  作图                                                                          #
# --------------------------------------------------------------------------- #

def _draw_scatter(ax, df: pd.DataFrame, lims: tuple, title: str) -> None:
    """在 ax 上绘制一个方法所有节点的 真实年龄-预测年龄 散点（颜色+形状区分节点）。"""
    ax.plot(lims, lims, color="#888888", linestyle="--", linewidth=1.2, zorder=1, label="y = x")
    nodes = [n for n in NODE_ORDER if n in set(df["node"])]
    for node in nodes:
        d = df[df["node"] == node]
        mae = float(np.mean(np.abs(d["age_pred"] - d["age_true"])))
        ax.scatter(d["age_true"], d["age_pred"], s=18, alpha=0.7,
                   color=NODE_COLORS[node], marker=NODE_MARKERS[node],
                   edgecolors="white", linewidths=0.4, zorder=2,
                   label=f"{NODE_LABELS[node]} (n={len(d)}, MAE={mae:.2f})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("True Age", fontsize=11)
    ax.set_ylabel("Predicted Age", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)


def _axis_limits(df: pd.DataFrame) -> tuple:
    """x/y 共用的坐标范围（覆盖全部真值和预测值，两侧留 3% 边距）。"""
    lo = float(min(df["age_true"].min(), df["age_pred"].min()))
    hi = float(max(df["age_true"].max(), df["age_pred"].max()))
    pad = 0.03 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def plot_method_scatter(df: pd.DataFrame, method: str, save_path: str, fmt: str) -> None:
    _setup_font()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _draw_scatter(ax, df, _axis_limits(df), DISPLAY_NAMES.get(method, method))
    fig.tight_layout()
    save_figure(fig, save_path, fmt=fmt)


def plot_all_methods_grid(all_df: pd.DataFrame, methods: list, save_path: str, fmt: str) -> None:
    """各方法散点图并排，坐标轴范围统一。"""
    _setup_font()
    lims = _axis_limits(all_df)
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.8), squeeze=False)
    for ax, m in zip(axes[0], methods):
        _draw_scatter(ax, all_df[all_df["method"] == m], lims, DISPLAY_NAMES.get(m, m))
    fig.tight_layout()
    save_figure(fig, save_path, fmt=fmt)


# --------------------------------------------------------------------------- #
#  主流程                                                                        #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    fmt = args.fmt or cfg.eval.figure_format
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else cfg.federated.qlora_alpha
    mri_shape = tuple(cfg.data.mri_shape)
    results_dir = args.results_dir

    # ---- 找到各方法各节点的领域模型权重 ----
    ckpts = {}   # {method: {node: path}}
    for m in args.methods:
        for path in sorted(glob.glob(os.path.join(results_dir, m, "*_swin.pt"))):
            node = os.path.basename(path)[: -len("_swin.pt")]
            if _NODE_RE.match(node):
                ckpts.setdefault(m, {})[node] = path
        if m not in ckpts:
            print(f"[跳过] {m}：{os.path.join(results_dir, m)} 下没有 *_swin.pt")
    if not ckpts:
        raise SystemExit("没有找到任何领域模型权重，请检查 --results_dir")

    # ---- 重建与 run_all 相同的节点划分 ----
    all_ds_ids = _load_openmind_ds_ids(cfg)
    quality_stats = _load_ds_quality_stats(cfg)
    sample_loader_fn = make_openmind_sample_loader(
        cfg.data.hf_dataset_id, cache_dir=cfg.data.cache_dir or None,
        max_cache_gb=cfg.data.max_cache_gb,
    )
    split_kwargs = dict(
        all_ds_ids=all_ds_ids, n_cross=cfg.data.n_cross_ds,
        n_private_per_node=cfg.data.n_private_ds_per_node, train_split=cfg.data.train_split,
        mri_shape=mri_shape, seed=cfg.data.seed, sample_loader_fn=sample_loader_fn,
        quality_stats=quality_stats, max_missing_ratio=cfg.data.cross_max_missing_ratio,
        n_public_max=cfg.data.n_public_ds_max,
    )
    # M4Fed 用服务器 + n_clients（与 run_all_baselines 相同的取值逻辑）；其余基线用 3 客户端
    m4fed_n_clients = cfg.data.n_clients if hasattr(cfg.data, "n_clients") else 2
    split_of = {m: ("m4fed" if m == "m4fed" else "3c") for m in ckpts}
    splits = {}
    if "m4fed" in split_of.values():
        splits["m4fed"] = build_node_datasets(n_clients=m4fed_n_clients, **split_kwargs)
    if "3c" in split_of.values():
        splits["3c"] = build_node_datasets(n_clients=3, **split_kwargs)

    # ---- 逐 (划分, 节点) 推理：同一节点测试集只读取一次，送入所有相关方法的模型 ----
    per_node = {}    # {(method, node): DataFrame}
    skipped = {}     # {(method, node): n_skipped}
    for split_name, node_datasets in splits.items():
        methods_here = [m for m in ckpts if split_of[m] == split_name]
        nodes = [n for n in NODE_ORDER if any(n in ckpts[m] for m in methods_here)]
        for node in nodes:
            key = "server_private" if node == "server" else node
            test_ds = node_datasets[key]["test"]
            print(f"\n[推理] 划分={split_name} 节点={node} 测试样本数={len(test_ds)}")
            models = {m: load_swin_from_checkpoint(ckpts[m][node], mri_shape, lora_alpha, device)
                      for m in methods_here if node in ckpts[m]}
            info_rows, preds, n_skip = predict_node(models, test_ds, args.batch_size, device)
            for m in models:
                df = pd.DataFrame(info_rows)
                df.insert(0, "node", node)
                df.insert(0, "method", m)
                df["age_pred"] = preds[m]
                df["residual"] = df["age_pred"] - df["age_true"]
                per_node[(m, node)] = df
                skipped[(m, node)] = n_skip
            del models
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---- 保存 CSV、散点图、指标校验 ----
    check_rows, all_frames = [], []
    for m in ckpts:
        nodes = [n for n in NODE_ORDER if (m, n) in per_node]
        frames = [per_node[(m, n)] for n in nodes]
        for node, df in zip(nodes, frames):
            save_csv(df, os.path.join(results_dir, m, f"{node}_test_predictions.csv"))
            row = {"method": m, "node": node, "n_eval": len(df), "n_skipped": skipped[(m, node)]}
            if len(df) >= 2:
                row.update(compute_domain_metrics(df["age_true"].to_numpy(), df["age_pred"].to_numpy()))
            saved_path = os.path.join(results_dir, m, f"{node}_metrics.csv")
            if os.path.exists(saved_path):
                for k, v in pd.read_csv(saved_path).iloc[0].items():
                    row[f"saved_{k}"] = v
            check_rows.append(row)
        method_df = pd.concat(frames, ignore_index=True)
        save_csv(method_df, os.path.join(results_dir, m, "test_predictions.csv"))
        if len(method_df):
            plot_method_scatter(method_df, m, os.path.join(results_dir, m, f"{m}_test_scatter"), fmt)
        all_frames.append(method_df)

    all_df = pd.concat(all_frames, ignore_index=True)
    save_csv(all_df, os.path.join(results_dir, "test_predictions_all_methods.csv"))
    check_df = pd.DataFrame(check_rows)
    save_csv(check_df, os.path.join(results_dir, "test_predictions_metrics_check.csv"))
    plot_all_methods_grid(all_df, list(ckpts),
                          os.path.join(ensure_dir(os.path.join(results_dir, "figures")),
                                       "test_scatter_all_methods"), fmt)

    print("\n[校验] 重算指标 vs run_all 保存的指标：")
    cols = [c for c in ["method", "node", "n_eval", "n_skipped", "MAE", "saved_MAE",
                        "RMSE", "saved_RMSE", "Pearson_r", "saved_Pearson_r"] if c in check_df]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(check_df[cols].to_string(index=False))
    print(f"\n[完成] 逐样本预测与散点图已保存至 {results_dir}")


if __name__ == "__main__":
    main()
