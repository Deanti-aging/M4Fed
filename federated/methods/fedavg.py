"""
FedAvg 基线（McMahan et al., 2017）。

协议：
  - 只有领域预测模型（NeuroFM）参与联邦聚合，无生成模型
  - NeuroFM 随机初始化（对应"无预训练骨干"的基线场景）
  - 每轮：服务器广播全局模型 → 客户端本地训练 1 epoch → 加权平均聚合

联邦平均（FedAvg）聚合公式：
  w_global = Σ_k (n_k / N) · w_k
  其中 n_k 为客户端 k 的样本数，N 为所有客户端样本总数
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import os
from typing import List

from training.neurofm_trainer import NeuroFMTrainer
from evaluation.metrics import compute_domain_metrics
from utils.io_utils import save_csv, save_model, ensure_dir


def run_fedavg(
    server,
    clients: List,
    cfg,
    task_specs: dict,
    results_dir: str,
) -> dict:
    """
    运行 FedAvg 联邦训练协议。

    参数：
      server      : Server 实例（FedAvg 中服务器只做模型广播，不做训练）
      clients     : Client 实例列表
      cfg         : M4FedConfig 配置
      task_specs  : 任务规格字典
      results_dir : 结果保存根目录

    返回：
      结果字典：
        {
          'client_0': {'MAE': ..., 'RMSE': ..., 'Pearson_r': ..., 'R2': ...},
          'client_1': {...},
          'clients_mean': {...},  # 所有客户端的平均指标
          'clients_std': {...},   # 所有客户端的标准差
        }
    """
    device = cfg.training.device
    out = ensure_dir(os.path.join(results_dir, "fedavg"))
    n_rounds = cfg.federated.federated_rounds

    # 所有客户端随机初始化 NeuroFM（FedAvg 不使用预训练骨干）
    for c in clients:
        c.build_models(task_specs, random_init_neurofm=True)

    # 全局模型以第一个客户端的初始权重为基础
    ref_client = clients[0]
    global_state = copy.deepcopy(ref_client.neurofm.state_dict())

    # 记录每个客户端每轮的损失
    all_losses = {f"client_{c.client_id}": [] for c in clients}

    # ---- 联邦训练循环 ----
    for rnd in range(n_rounds):
        print(f"[FedAvg] 第 {rnd+1}/{n_rounds} 轮")

        local_states = []    # 收集各客户端本地训练后的模型状态
        local_ns = []        # 收集各客户端的样本数（用于加权平均）

        for c in clients:
            # 步骤 1：广播全局模型到客户端（覆盖本地权重）
            c.neurofm.load_state_dict(copy.deepcopy(global_state))

            # 步骤 2：客户端在本地数据上训练 1 epoch
            trainer = NeuroFMTrainer(c.neurofm, device=device,
                                     lr=cfg.training.lr_neurofm,
                                     mixed_precision=cfg.training.mixed_precision,
                                     use_qlora=False)
            losses = trainer.train(c.train_loader, n_epochs=1)
            all_losses[f"client_{c.client_id}"].append({"round": rnd, "loss": losses[0]})
            local_states.append(copy.deepcopy(c.neurofm.state_dict()))
            local_ns.append(c.n_train_samples())

        # 步骤 3：服务器执行 FedAvg 加权平均聚合
        global_state = _fedavg_aggregate(local_states, local_ns)

    # ---- 评估：将最终全局模型分发给各客户端并评估 ----
    results = {}
    for c in clients:
        c.neurofm.load_state_dict(copy.deepcopy(global_state))
        trainer = NeuroFMTrainer(c.neurofm, device=device, lr=0,
                                 mixed_precision=False, use_qlora=False)
        y_true, y_pred = trainer.evaluate(c.test_loader)
        results[f"client_{c.client_id}"] = compute_domain_metrics(y_true, y_pred)

    # 计算客户端平均和标准差
    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _aggregate_metrics({k: results[k] for k in client_keys})
    results["clients_mean"] = agg["mean"]
    results["clients_std"] = agg["std"]

    # ---- 保存结果 ----
    for c in clients:
        key = f"client_{c.client_id}"
        save_csv(pd.DataFrame(all_losses[key]), os.path.join(out, f"{key}_losses.csv"))
        save_csv(pd.DataFrame([results[key]]), os.path.join(out, f"{key}_metrics.csv"))

    if cfg.eval.save_model:
        for c in clients:
            save_model(c.neurofm, os.path.join(out, f"client_{c.client_id}_neurofm.pt"))

    return results


def _fedavg_aggregate(states: list, ns: list) -> dict:
    """
    执行 FedAvg 加权平均聚合。

    公式：w_global[key] = Σ_k (n_k / N) · w_k[key]
    各张量按样本数权重加权平均，得到全局模型参数。

    参数：
      states : 各客户端本地模型的 state_dict 列表
      ns     : 各客户端的训练样本数列表

    返回：
      聚合后的全局 state_dict
    """
    total = sum(ns)
    agg = copy.deepcopy(states[0])
    for key in agg:
        # 逐参数加权平均（转 float 避免整数类型的精度问题）
        agg[key] = sum(s[key].float() * (n / total) for s, n in zip(states, ns))
    return agg


def _aggregate_metrics(metrics_dict: dict) -> dict:
    """
    计算多个客户端指标字典的均值和标准差。

    参数：
      metrics_dict : {client_key: {metric_name: value}} 格式

    返回：
      {'mean': {metric_name: float}, 'std': {metric_name: float}}
    """
    from collections import defaultdict
    vals = defaultdict(list)
    for v in metrics_dict.values():
        for k, val in v.items():
            vals[k].append(val)
    return {
        "mean": {k: float(np.nanmean(v)) for k, v in vals.items()},
        "std":  {k: float(np.nanstd(v))  for k, v in vals.items()},
    }
