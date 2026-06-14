"""
FedProx 基线（Li et al., 2020）。

与 FedAvg 的唯一区别：在本地损失函数中加入近端正则化项：
  L_local(w) + (μ/2) ‖w - w_global‖²

该正则项惩罚本地模型偏离全局模型过远，在 Non-IID 数据分布下
比 FedAvg 更稳定（局部更新不会因为数据分布差异而导致全局模型发散）。

当 μ=0 时，FedProx 退化为 FedAvg。
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


def run_fedprox(
    server,
    clients: List,
    cfg,
    task_specs: dict,
    results_dir: str,
) -> dict:
    """
    运行 FedProx 联邦训练协议。

    与 FedAvg 的实现差异：
      - NeuroFMTrainer 接收 proximal_mu 和 global_state 参数
      - 每次本地训练前将当前全局状态传入 Trainer，用于计算近端项
      - 聚合仍使用 FedAvg 加权平均（FedProx 只改变本地目标，不改变聚合策略）

    参数说明同 run_fedavg。

    返回格式同 run_fedavg。
    """
    device = cfg.training.device
    mu = cfg.federated.fedprox_mu    # 近端正则化系数
    out = ensure_dir(os.path.join(results_dir, "fedprox"))

    # NeuroFM 随机初始化
    for c in clients:
        c.build_models(task_specs, random_init_neurofm=True)

    global_state = copy.deepcopy(clients[0].neurofm.state_dict())
    all_losses = {f"client_{c.client_id}": [] for c in clients}

    # ---- 联邦训练循环 ----
    for rnd in range(cfg.federated.federated_rounds):
        print(f"[FedProx] 第 {rnd+1}/{cfg.federated.federated_rounds} 轮，μ={mu}")

        local_states, local_ns = [], []
        for c in clients:
            # 广播全局模型
            c.neurofm.load_state_dict(copy.deepcopy(global_state))
            # 本地训练：传入 proximal_mu 和当前全局状态，训练器内部计算近端项
            trainer = NeuroFMTrainer(
                c.neurofm, device=device,
                lr=cfg.training.lr_neurofm,
                mixed_precision=cfg.training.mixed_precision,
                use_qlora=False,
                proximal_mu=mu,                          # FedProx 近端系数
                global_state=copy.deepcopy(global_state), # 当前轮的全局权重（用于近端项）
            )
            losses = trainer.train(c.train_loader, n_epochs=1)
            all_losses[f"client_{c.client_id}"].append({"round": rnd, "loss": losses[0]})
            local_states.append(copy.deepcopy(c.neurofm.state_dict()))
            local_ns.append(c.n_train_samples())

        # 聚合（与 FedAvg 相同：加权平均）
        global_state = _fedavg_aggregate(local_states, local_ns)

    # ---- 评估 ----
    results = {}
    for c in clients:
        c.neurofm.load_state_dict(copy.deepcopy(global_state))
        trainer = NeuroFMTrainer(c.neurofm, device=device, lr=0,
                                 mixed_precision=False, use_qlora=False)
        y_true, y_pred = trainer.evaluate(c.test_loader)
        results[f"client_{c.client_id}"] = compute_domain_metrics(y_true, y_pred)

    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _agg({k: results[k] for k in client_keys})
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


def _fedavg_aggregate(states, ns):
    """FedAvg 加权平均聚合（实现同 fedavg.py）。"""
    total = sum(ns)
    agg = copy.deepcopy(states[0])
    for key in agg:
        agg[key] = sum(s[key].float() * (n / total) for s, n in zip(states, ns))
    return agg


def _agg(d):
    """计算指标字典的均值和标准差。"""
    from collections import defaultdict
    vals = defaultdict(list)
    for v in d.values():
        for k, val in v.items():
            vals[k].append(val)
    return {
        "mean": {k: float(np.nanmean(v)) for k, v in vals.items()},
        "std":  {k: float(np.nanstd(v)) for k, v in vals.items()},
    }
