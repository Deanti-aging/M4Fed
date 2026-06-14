"""
运行全部 5 个联邦学习基线方法并生成对比图表。

5 个方法：
  M4Fed  —— 本论文提出的方法（服务器 + 2 个客户端）
  FedAvg —— 联邦平均（3 个客户端）
  FedProx—— FedAvg + 近端正则项（3 个客户端）
  FAN    —— 联邦增强网络，去中心化合成数据交换（3 个客户端）
  GPT-FL —— 服务器端文本条件生成（3 个客户端）

客户端数量差异：
  M4Fed 使用服务器 + 2 个客户端的三方结构（服务器持有公开数据）；
  其余基线使用 3 个纯客户端结构（无区分服务器的公开数据）。
  因此 FedAvg/FedProx/FAN/GPT-FL 都重新构建了 3 客户端的数据集划分。

用法：
  python experiments/run_all_baselines.py --config configs/m4fed.yaml [--dummy]
"""
from __future__ import annotations
import argparse
import os
import sys
import copy
import numpy as np
import pandas as pd

# 将项目根目录加入路径（使实验脚本可以直接运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import load_config
from utils.seed_utils import set_seed
from utils.io_utils import ensure_dir, save_csv
from data.openmind_dataset import build_node_datasets
from data.splits import assign_ds_splits
from federated.server import Server
from federated.client import Client
from federated.methods import run_m4fed, run_fedavg, run_fedprox, run_fan, run_gptfl
from evaluation.visualization import (
    plot_loss_curves, plot_metric_bars, save_figure,
    plot_scatter, plot_age_bin_mae, plot_residuals, plot_error_heatmap,
    save_all_bias_figures,
)
from evaluation.metrics import compute_domain_metrics, compute_bias_analysis


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m4fed.yaml")
    p.add_argument("--dummy", action="store_true",
                   help="使用极小虚拟数据进行冒烟测试（秒级完成）")
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


def build_participants(cfg, node_datasets: dict, device: str, n_clients: int):
    """
    从预构建的节点数据集字典实例化 Server 和 Client 对象。

    参数：
      cfg          : 当前方法的配置对象
      node_datasets: build_node_datasets 返回的节点数据集字典
      device       : 计算设备（'cuda' 或 'cpu'）
      n_clients    : 客户端数量

    返回：
      (server, clients) 元组
    """
    server = Server(
        cfg=cfg,
        private_datasets=node_datasets["server_private"],
        public_datasets=node_datasets["server_public"],
        device=device,
    )
    clients = [
        Client(
            client_id=i,
            cfg=cfg,
            private_datasets=node_datasets[f"client_{i}"],
            device=device,
        )
        for i in range(n_clients)
    ]
    return server, clients


def default_task_specs():
    """
    返回默认任务规格字典。

    包含两个任务：
      age：回归任务，输出维度 1
      sex：分类任务，类别数 2（男/女）
    """
    return {
        "age": {"type": "regression", "n_classes": 1},
        "sex": {"type": "classification", "n_classes": 2},
    }


def main():
    """主函数：顺序运行 5 个方法并汇总对比结果。"""
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    device = cfg.training.device
    # 若 CUDA 不可用但配置要求 CUDA，自动降级到 CPU
    if not _cuda_available() and device == "cuda":
        device = "cpu"
        cfg.training.device = "cpu"
        print("[run_all] CUDA 不可用，已切换至 CPU。")

    results_dir = ensure_dir(args.results_dir)
    task_specs = default_task_specs()

    # ------------------------------------------------------------------ #
    # 数据集准备                                                            #
    # ------------------------------------------------------------------ #
    if args.dummy:
        # 虚拟模式：生成 100 个假 ID，MRI 尺寸缩小为 16³
        all_ds_ids = [f"ds{i:06d}" for i in range(100)]
        mri_shape = (16, 16, 16)
    else:
        # 正式模式：从 HuggingFace OpenMind 流式加载数据集 ID
        all_ds_ids = _load_openmind_ds_ids(cfg)
        mri_shape = tuple(cfg.data.mri_shape)

    # 构建 M4Fed 专用划分（服务器 + 2 个客户端）
    node_datasets = build_node_datasets(
        all_ds_ids=all_ds_ids,
        n_cross=cfg.data.n_cross_ds,
        n_private_per_node=cfg.data.n_private_ds_per_node,
        n_clients=cfg.data.n_clients if hasattr(cfg.data, "n_clients") else 2,
        train_split=cfg.data.train_split,
        mri_shape=mri_shape,
        seed=cfg.data.seed,
        dummy_mode=args.dummy,
        n_dummy_per_ds=2 if args.dummy else 10,
    )

    # 构建 3 客户端划分（FedAvg/FedProx/FAN/GPT-FL 共用）
    node_datasets_3 = build_node_datasets(
        all_ds_ids=all_ds_ids,
        n_cross=cfg.data.n_cross_ds,
        n_private_per_node=cfg.data.n_private_ds_per_node,
        n_clients=3,
        train_split=cfg.data.train_split,
        mri_shape=mri_shape,
        seed=cfg.data.seed,
        dummy_mode=args.dummy,
        n_dummy_per_ds=2 if args.dummy else 10,
    )

    # ------------------------------------------------------------------ #
    # 依次运行各方法                                                        #
    # ------------------------------------------------------------------ #
    all_results = {}

    # ---- M4Fed（本论文方法）----
    print("\n======== M4Fed ========")
    server, clients = build_participants(cfg, node_datasets, device, n_clients=2)
    set_seed(cfg.training.seed)
    all_results["m4fed"] = run_m4fed(server, clients, cfg, task_specs, results_dir)
    # 保存服务器 MedMD 路径供 GPT-FL 复用（GPT-FL 需要预训练好的生成模型）
    m4fed_medmd_path = os.path.join(results_dir, "m4fed", "server_medmd.pt")

    # ---- FedAvg（联邦平均）----
    print("\n======== FedAvg ========")
    cfg_fa = copy.deepcopy(cfg)
    cfg_fa.federated.method = "fedavg"
    cfg_fa.federated.n_clients = 3
    server3, clients3 = build_participants(cfg_fa, node_datasets_3, device, n_clients=3)
    set_seed(cfg.training.seed)
    all_results["fedavg"] = run_fedavg(server3, clients3, cfg_fa, task_specs, results_dir)

    # ---- FedProx（联邦近端正则）----
    print("\n======== FedProx ========")
    cfg_fp = copy.deepcopy(cfg)
    cfg_fp.federated.method = "fedprox"
    cfg_fp.federated.n_clients = 3
    server3b, clients3b = build_participants(cfg_fp, node_datasets_3, device, n_clients=3)
    set_seed(cfg.training.seed)
    all_results["fedprox"] = run_fedprox(server3b, clients3b, cfg_fp, task_specs, results_dir)

    # ---- FAN（联邦增强网络）----
    print("\n======== FAN ========")
    cfg_fan = copy.deepcopy(cfg)
    cfg_fan.federated.method = "fan"
    cfg_fan.federated.n_clients = 3
    server3c, clients3c = build_participants(cfg_fan, node_datasets_3, device, n_clients=3)
    set_seed(cfg.training.seed)
    all_results["fan"] = run_fan(server3c, clients3c, cfg_fan, task_specs, results_dir)

    # ---- GPT-FL（服务器端文本生成基线）----
    print("\n======== GPT-FL ========")
    cfg_gfl = copy.deepcopy(cfg)
    cfg_gfl.federated.method = "gptfl"
    cfg_gfl.federated.n_clients = 3
    server3d, clients3d = build_participants(cfg_gfl, node_datasets_3, device, n_clients=3)
    set_seed(cfg.training.seed)
    all_results["gptfl"] = run_gptfl(
        server3d, clients3d, cfg_gfl, task_specs, results_dir,
        pretrained_medmd_path=m4fed_medmd_path,  # 复用 M4Fed 预训练好的服务器 MedMD
    )

    # ------------------------------------------------------------------ #
    # 汇总指标并生成对比图表                                                 #
    # ------------------------------------------------------------------ #
    _save_summary_and_figures(all_results, results_dir, cfg)


def _save_summary_and_figures(all_results: dict, results_dir: str, cfg) -> None:
    """
    将所有方法的客户端均值指标汇总为 CSV，并生成各指标的柱状图。

    参数：
      all_results : {方法名: run_xxx() 的返回字典}
      results_dir : 根结果目录
      cfg         : 配置对象（读取 figure_format 和 palette）
    """
    fmt = cfg.eval.figure_format
    palette = cfg.eval.palette

    # ---- 构造汇总表格 ----
    summary_rows = []
    for method, res in all_results.items():
        mean = res.get("clients_mean", {})
        std = res.get("clients_std", {})
        summary_rows.append({
            "method": method,
            "MAE_mean": mean.get("MAE", float("nan")),
            "MAE_std": std.get("MAE", float("nan")),
            "RMSE_mean": mean.get("RMSE", float("nan")),
            "RMSE_std": std.get("RMSE", float("nan")),
            "Pearson_r_mean": mean.get("Pearson_r", float("nan")),
            "Pearson_r_std": std.get("Pearson_r", float("nan")),
            "R2_mean": mean.get("R2", float("nan")),
            "R2_std": std.get("R2", float("nan")),
        })

    # 单独列出 M4Fed 服务器端指标（与客户端指标并列对比）
    if "m4fed" in all_results:
        sr = all_results["m4fed"].get("server", {})
        summary_rows.append({
            "method": "M4Fed-server",
            "MAE_mean": sr.get("MAE", float("nan")), "MAE_std": 0,
            "RMSE_mean": sr.get("RMSE", float("nan")), "RMSE_std": 0,
            "Pearson_r_mean": sr.get("Pearson_r", float("nan")), "Pearson_r_std": 0,
            "R2_mean": sr.get("R2", float("nan")), "R2_std": 0,
        })

    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, os.path.join(results_dir, "summary_domain_metrics.csv"))
    print("[汇总] 领域指标汇总表已保存。")

    # ---- 绘制各指标的柱状图（含误差棒）----
    # 定义显示标签映射（内部键 → 论文图表展示名称）
    bname_map = {
        "fedavg": "FedAvg", "fedprox": "FedProx", "fan": "FAN",
        "gptfl": "GPT-FL", "m4fed": "M4Fed-clients", "M4Fed-server": "M4Fed-server",
    }
    metric_means = {}
    metric_stds = {}
    for row in summary_rows:
        m = row["method"]
        label = bname_map.get(m, m)
        metric_means[label] = {
            "MAE": row.get("MAE_mean", float("nan")),
            "RMSE": row.get("RMSE_mean", float("nan")),
            "Pearson_r": row.get("Pearson_r_mean", float("nan")),
            "R2": row.get("R2_mean", float("nan")),
        }
        metric_stds[label] = {
            "MAE": row.get("MAE_std", 0),
            "RMSE": row.get("RMSE_std", 0),
            "Pearson_r": row.get("Pearson_r_std", 0),
            "R2": row.get("R2_std", 0),
        }

    figs_dir = ensure_dir(os.path.join(results_dir, "figures"))
    for metric in ["MAE", "RMSE", "Pearson_r", "R2"]:
        plot_metric_bars(
            metric_means, metric_stds,
            metric_name=metric,
            palette=palette,
            save_path=os.path.join(figs_dir, f"bar_{metric}"),
            fmt=fmt,
        )
    print("[汇总] 指标柱状图已保存至", figs_dir)


def _cuda_available() -> bool:
    """检查 CUDA 是否可用。"""
    import torch
    return torch.cuda.is_available()


def _load_openmind_ds_ids(cfg) -> list:
    """
    从 HuggingFace OpenMind 流式加载子数据集 ID 列表。

    使用流式模式（streaming=True）避免下载完整数据集。
    最多读取 200 个唯一 dataset_id（满足后停止，提高速度）。

    若加载失败（网络问题/权限问题），回退到虚拟 ID 列表并打印警告。

    返回：
      排序后的子数据集 ID 字符串列表
    """
    print("[数据] 流式加载 OpenMind 数据集 ID...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            cfg.data.hf_dataset_id,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ds_ids = set()
        for sample in ds:
            ds_id = sample.get("dataset_id") or sample.get("ds_id") or "unknown"
            ds_ids.add(str(ds_id))
            if len(ds_ids) >= 200:   # 达到上限后停止（避免无限流式）
                break
        return sorted(ds_ids)
    except Exception as e:
        print(f"[数据] 无法从 HuggingFace 加载：{e}，使用虚拟 ID 代替。")
        return [f"ds{i:06d}" for i in range(100)]


if __name__ == "__main__":
    main()
