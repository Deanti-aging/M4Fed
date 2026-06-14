"""
M4Fed 消融实验（Ablation Study）。

验证三个组件的有效性（通过逐一移除）：
  变体 A：w/o server public data    —— 服务器公开数据集置空，验证公开数据的作用
  变体 B：w/o TabPFN                —— 禁用表格补全（缺失值保持 NaN），验证 TabPFN 的作用
  变体 C：random-init NeuroFM      —— NeuroFM 随机初始化（不使用预训练权重），
                                      验证预训练骨干的作用

每个变体都在相同的数据集划分和超参数下运行完整的 M4Fed 协议，
唯一的区别是对应组件被移除或替换。
"""
from __future__ import annotations
import os, sys, copy
import numpy as np
import pandas as pd

# 将项目根目录加入路径（使实验脚本可以直接运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import load_config
from utils.seed_utils import set_seed
from utils.io_utils import ensure_dir, save_csv
from data.openmind_dataset import build_node_datasets
from federated.server import Server
from federated.client import Client
from federated.methods import run_m4fed
from evaluation.visualization import plot_metric_bars


def _build(cfg, node_datasets, device, n_clients=2):
    """
    快速构建服务器和客户端参与者对象。

    参数：
      cfg          : 配置对象（已为当前变体深拷贝并修改）
      node_datasets: 节点数据集字典
      device       : 计算设备

    返回：
      (server, clients) 元组
    """
    server = Server(cfg, node_datasets["server_private"],
                    node_datasets["server_public"], device)
    clients = [Client(i, cfg, node_datasets[f"client_{i}"], device)
               for i in range(n_clients)]
    return server, clients


def run_ablation(args, cfg, all_ds_ids, mri_shape, task_specs, results_dir):
    """
    运行全部三个消融变体并生成对比结果。

    每个变体：
      1. 深拷贝基础配置（避免变体间相互影响）
      2. 深拷贝基础数据集划分
      3. 修改对应组件（公开数据/TabPFN/NeuroFM 初始化）
      4. 运行完整 M4Fed 协议
      5. 记录结果

    参数：
      args       : 命令行参数（.dummy, 等）
      cfg        : 基础 M4FedConfig
      all_ds_ids : 所有子数据集 ID 列表
      mri_shape  : MRI 体积目标尺寸
      task_specs : 任务规格字典
      results_dir: 结果保存根目录

    返回：
      all_ablation_results 字典（键为变体名，值为对应结果字典）
    """
    device = cfg.training.device
    ablation_out = ensure_dir(os.path.join(results_dir, "ablation"))

    # 构建基础数据集（所有变体共用相同的划分）
    node_datasets = build_node_datasets(
        all_ds_ids=all_ds_ids,
        n_cross=cfg.data.n_cross_ds,
        n_private_per_node=cfg.data.n_private_ds_per_node,
        n_clients=2,
        train_split=cfg.data.train_split,
        mri_shape=mri_shape,
        seed=cfg.data.seed,
        dummy_mode=args.dummy,
        n_dummy_per_ds=2 if args.dummy else 10,
    )

    ablation_results = {}

    # ------------------------------------------------------------------
    # 变体 A：移除服务器公开数据
    # 实现：将 server_public 的 samples 列表清空
    # ------------------------------------------------------------------
    print("[消融] 变体 A：无服务器公开数据")
    set_seed(cfg.training.seed)
    cfg_a = copy.deepcopy(cfg)
    nd_a = copy.deepcopy(node_datasets)
    nd_a["server_public"]["train"].samples = []   # 清空公开训练集
    nd_a["server_public"]["test"].samples = []    # 清空公开测试集
    s, c = _build(cfg_a, nd_a, device)
    ablation_results["w/o_public"] = run_m4fed(
        s, c, cfg_a, task_specs, os.path.join(ablation_out, "wo_public"))

    # ------------------------------------------------------------------
    # 变体 B：移除 TabPFN 表格补全
    # 实现：Monkey-patch TabPFNImputer.fit_transform 为 identity 函数
    # 效果：所有 NaN 值保持原始状态（不做填充），文本条件质量下降
    # ------------------------------------------------------------------
    print("[消融] 变体 B：无 TabPFN 表格补全")
    set_seed(cfg.training.seed)
    cfg_b = copy.deepcopy(cfg)
    s, c = _build(cfg_b, node_datasets, device)
    # Monkey-patch：让 fit_transform 直接返回原始 DataFrame（不做任何填充）
    from models.tabpfn_wrapper import TabPFNImputer
    TabPFNImputer.fit_transform = lambda self, df: df
    ablation_results["w/o_tabpfn"] = run_m4fed(
        s, c, cfg_b, task_specs, os.path.join(ablation_out, "wo_tabpfn"))

    # ------------------------------------------------------------------
    # 变体 C：NeuroFM 随机初始化（验证预训练骨干的作用）
    # 实现：Monkey-patch Server.build_models，使其调用 random_init_neurofm=True
    # ------------------------------------------------------------------
    print("[消融] 变体 C：NeuroFM 随机初始化（无预训练骨干）")
    set_seed(cfg.training.seed)
    cfg_c = copy.deepcopy(cfg)
    s, c = _build(cfg_c, node_datasets, device)
    # 保存原始 build_models 方法，实验结束后恢复
    original_build = Server.build_models
    def _random_build(self, ts, **kw):
        """强制 random_init_neurofm=True 的 build_models 替代版本。"""
        original_build(self, ts, random_init_neurofm=True)
    Server.build_models = _random_build
    ablation_results["random_init_neurofm"] = run_m4fed(
        s, c, cfg_c, task_specs, os.path.join(ablation_out, "random_init"))
    Server.build_models = original_build   # 恢复原始方法（避免影响后续实验）

    # ------------------------------------------------------------------
    # 汇总并保存结果
    # ------------------------------------------------------------------
    rows = []
    for variant, res in ablation_results.items():
        mean = res.get("clients_mean", {})
        std = res.get("clients_std", {})
        rows.append({"variant": variant,
                     "MAE": mean.get("MAE", float("nan")),
                     "RMSE": mean.get("RMSE", float("nan")),
                     "Pearson_r": mean.get("Pearson_r", float("nan")),
                     "R2": mean.get("R2", float("nan")),
                     "MAE_std": std.get("MAE", 0),
                     })
    summary_df = pd.DataFrame(rows)
    save_csv(summary_df, os.path.join(ablation_out, "ablation_summary.csv"))
    print("[消融] 结果已保存至", ablation_out)

    # 绘制各变体 MAE 对比柱状图
    fmt = cfg.eval.figure_format
    metric_means = {r["variant"]: {k: r[k] for k in ["MAE","RMSE","Pearson_r","R2"]}
                    for r in rows}
    plot_metric_bars(metric_means, metric_name="MAE",
                     title="消融实验：MAE 对比",
                     save_path=os.path.join(ablation_out, "ablation_MAE"), fmt=fmt)
    return ablation_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m4fed.yaml")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()
    cfg = load_config(args.config)
    all_ds_ids = [f"ds{i:06d}" for i in range(100)] if args.dummy else []
    mri_shape = (16, 16, 16) if args.dummy else tuple(cfg.data.mri_shape)
    task_specs = {"age": {"type": "regression", "n_classes": 1}}
    run_ablation(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)
