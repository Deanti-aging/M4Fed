"""
M4Fed 敏感性分析（Sensitivity Analysis）。

对三个关键超参数进行独立的网格搜索，每次仅改变一个变量，其余保持默认值：

  1. 合成数据比例（synth_ratio）：[0.1, 1.0, 10.0]
     控制每轮联邦循环和本地训练中生成样本数与真实样本数的比值。
     0.1 → 生成较少合成数据，10.0 → 生成 10 倍于真实数据的合成数据。

  2. MedMD 学习率（lr_medmd）：[1e-5, 1e-4, 1e-3]
     控制扩散模型 U-Net 的训练速度与收敛稳定性的平衡。

  3. 数据异质性（Dirichlet α）：[0.1, 0.5, 1.0]
     α → 0 时数据分布极不均匀（高度 Non-IID），α → ∞ 时趋向 IID。
     通过 dirichlet_resplit() 重新划分各节点私有数据来模拟不同异质程度。

每个配置点均运行完整的 M4Fed 协议，汇总结果保存为 CSV。
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
from data.openmind_dataset import build_node_datasets, OpenMindDataset
from federated.server import Server
from federated.client import Client
from federated.methods import run_m4fed
from evaluation.visualization import plot_loss_curves


def _build(cfg, node_datasets, device):
    """
    快速构建服务器和客户端（2 个客户端）。

    参数：
      cfg          : 当前扫描点的配置（已深拷贝）
      node_datasets: 节点数据集字典
      device       : 计算设备
    """
    server = Server(cfg, node_datasets["server_private"],
                    node_datasets["server_public"], device)
    clients = [Client(i, cfg, node_datasets[f"client_{i}"], device)
               for i in range(2)]
    return server, clients


def _base_datasets(cfg, all_ds_ids, mri_shape, dummy, n_dummy_per_ds=2):
    """
    构建基础节点数据集（所有扫描点共用相同的初始划分）。

    参数与 build_node_datasets 一致，n_clients 固定为 2（服务器 + 2 个客户端）。
    """
    return build_node_datasets(
        all_ds_ids=all_ds_ids,
        n_cross=cfg.data.n_cross_ds,
        n_private_per_node=cfg.data.n_private_ds_per_node,
        n_clients=2,
        train_split=cfg.data.train_split,
        mri_shape=mri_shape,
        seed=cfg.data.seed,
        dummy_mode=dummy,
        n_dummy_per_ds=n_dummy_per_ds,
    )


def dirichlet_resplit(node_datasets: dict, alpha: float, seed: int) -> dict:
    """
    按 Dirichlet(α) 分布重新划分各节点的私有训练数据，模拟 Non-IID 场景。

    算法步骤：
      1. 收集所有节点（server_private + client_0 + client_1）的训练样本
      2. 用 np.random.dirichlet([α, α, α]) 采样三节点的数据比例
      3. 对全量样本随机打乱并按比例切分
      4. 将切分结果重新赋值给各节点的 .samples 属性（原地修改）

    参数：
      node_datasets : build_node_datasets 返回的数据集字典
      alpha         : Dirichlet 浓度参数（越小越 Non-IID）
      seed          : 随机种子，确保可复现

    返回：
      重新划分后的 node_datasets（原地修改并返回）
    """
    rng = np.random.RandomState(seed)
    nodes = ["server_private"] + [f"client_{i}" for i in range(2)]

    # 汇总所有节点的训练样本
    all_samples = []
    for node_key in nodes:
        ds = node_datasets[node_key]["train"]
        if hasattr(ds, "samples"):
            all_samples.extend(ds.samples)

    n = len(all_samples)
    if n == 0:
        return node_datasets  # 空数据集，直接返回（避免除零错误）

    # Dirichlet 采样各节点的数据比例
    proportions = rng.dirichlet([alpha] * len(nodes))
    # 将比例转换为整数边界（累积和）
    boundaries = (np.cumsum(proportions) * n).astype(int)
    # 随机打乱所有样本索引（确保各节点收到的数据不带偏序）
    indices = rng.permutation(n)
    starts = [0] + list(boundaries[:-1])

    # 按边界切分并重新赋值
    for i, node_key in enumerate(nodes):
        node_idxs = indices[starts[i]:boundaries[i]]
        new_samples = [all_samples[j] for j in node_idxs]
        ds = node_datasets[node_key]["train"]
        if hasattr(ds, "samples"):
            ds.samples = new_samples

    return node_datasets


def run_sensitivity(args, cfg, all_ds_ids, mri_shape, task_specs, results_dir):
    """
    运行三组敏感性分析并汇总结果。

    参数：
      args       : 命令行参数（.dummy 标志等）
      cfg        : 基础 M4FedConfig
      all_ds_ids : 子数据集 ID 列表
      mri_shape  : MRI 体积目标尺寸
      task_specs : 任务规格字典
      results_dir: 结果保存根目录

    返回：
      all_sensitivity_results 字典（键为扫描点标签，值为 {param, value, MAE, RMSE, ...}）
    """
    sens_out = ensure_dir(os.path.join(results_dir, "sensitivity"))
    device = cfg.training.device
    dummy = args.dummy
    # 构建基础数据集（各扫描点深拷贝，避免相互污染）
    nd_base = _base_datasets(cfg, all_ds_ids, mri_shape, dummy)

    all_sensitivity_results = {}

    # ================================================================ #
    # 扫描 1：合成数据比例（synth_ratio）                                  #
    # ================================================================ #
    for ratio in [0.1, 1.0, 10.0]:
        tag = f"synth_ratio_{ratio}"
        print(f"[敏感性] 合成比例 synth_ratio = {ratio}")
        set_seed(cfg.training.seed)
        cfg_r = copy.deepcopy(cfg)
        cfg_r.federated.synth_ratio_fed = ratio    # 联邦循环阶段的合成比例
        cfg_r.federated.synth_ratio_local = ratio  # 本地训练阶段的合成比例
        nd = copy.deepcopy(nd_base)
        s, c = _build(cfg_r, nd, device)
        res = run_m4fed(s, c, cfg_r, task_specs,
                        os.path.join(sens_out, tag))
        all_sensitivity_results[tag] = {
            "param": "synth_ratio", "value": ratio,
            **res.get("clients_mean", {}),   # 展开 MAE/RMSE/Pearson_r/R2
        }

    # ================================================================ #
    # 扫描 2：MedMD 学习率（lr_medmd）                                    #
    # ================================================================ #
    for lr in [1e-5, 1e-4, 1e-3]:
        tag = f"lr_{lr:.0e}"
        print(f"[敏感性] 学习率 lr_medmd = {lr}")
        set_seed(cfg.training.seed)
        cfg_l = copy.deepcopy(cfg)
        cfg_l.training.lr_medmd = lr
        nd = copy.deepcopy(nd_base)
        s, c = _build(cfg_l, nd, device)
        res = run_m4fed(s, c, cfg_l, task_specs,
                        os.path.join(sens_out, tag))
        all_sensitivity_results[tag] = {
            "param": "lr_medmd", "value": lr,
            **res.get("clients_mean", {}),
        }

    # ================================================================ #
    # 扫描 3：数据异质性（Dirichlet α）                                   #
    # ================================================================ #
    for alpha in [0.1, 0.5, 1.0]:
        tag = f"dirichlet_alpha_{alpha}"
        print(f"[敏感性] 数据异质性 Dirichlet α = {alpha}")
        set_seed(cfg.training.seed)
        cfg_d = copy.deepcopy(cfg)
        # 重新按 Dirichlet 分布划分各节点私有数据
        nd = dirichlet_resplit(copy.deepcopy(nd_base), alpha=alpha,
                               seed=cfg.training.seed)
        s, c = _build(cfg_d, nd, device)
        res = run_m4fed(s, c, cfg_d, task_specs,
                        os.path.join(sens_out, tag))
        all_sensitivity_results[tag] = {
            "param": "dirichlet_alpha", "value": alpha,
            **res.get("clients_mean", {}),
        }

    # ---- 汇总保存 ----
    summary_df = pd.DataFrame(list(all_sensitivity_results.values()))
    summary_df.insert(0, "tag", list(all_sensitivity_results.keys()))
    save_csv(summary_df, os.path.join(sens_out, "sensitivity_summary.csv"))
    print("[敏感性] 结果已保存至", sens_out)
    return all_sensitivity_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m4fed.yaml")
    p.add_argument("--dummy", action="store_true", help="使用虚拟数据快速验证")
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()
    cfg = load_config(args.config)
    all_ds_ids = [f"ds{i:06d}" for i in range(100)] if args.dummy else []
    mri_shape = (16, 16, 16) if args.dummy else tuple(cfg.data.mri_shape)
    task_specs = {"age": {"type": "regression", "n_classes": 1}}
    run_sensitivity(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)
