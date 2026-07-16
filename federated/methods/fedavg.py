"""
FedAvg 基线（McMahan et al., 2017）。

协议：
  - 只有领域预测模型（SwinMultiTask）参与联邦聚合，无生成模型
  - SwinUNETR 骨干随机初始化（对应"无预训练骨干"的基线场景）
  - 每轮：服务器广播全局模型 → 客户端本地训练 1 epoch → 加权平均聚合

联邦平均（FedAvg）聚合公式：
  w_global = Σ_k (n_k / N) · w_k
  其中 n_k 为客户端 k 的样本数，N 为所有客户端样本总数
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import copy                            # 标准库：深拷贝（用于保护全局/本地状态字典）
import numpy as np                     # NumPy：计算指标的均值和标准差
import pandas as pd                    # Pandas：保存损失曲线和指标到 CSV
import os                              # 标准库：路径拼接
from typing import List                # 类型注解：客户端列表

from training.swin_trainer import SwinTrainer                  # SwinMultiTask 本地训练器
from evaluation.metrics import compute_domain_metrics          # 领域预测评估指标（MAE、RMSE 等）
from utils.io_utils import save_csv, save_model, ensure_dir    # 文件保存工具
from utils.timing_utils import record_timing                   # 计时记录工具
import time                            # 标准库：高精度时间戳（perf_counter）


def run_fedavg(
    server,           # Server 实例（FedAvg 中仅做广播，不参与本地训练）
    clients: List,    # Client 实例列表（每个代表一个联邦参与节点）
    cfg,              # 全局 M4FedConfig 配置对象
    task_specs: dict, # 预测任务规格字典（任务名 → 输出维度/类型）
    results_dir: str, # 结果保存根目录（会在其下创建 "fedavg/" 子目录）
) -> dict:
    """
    运行 FedAvg 联邦训练协议。

    返回：
      结果字典，格式：
        {
          'client_0': {'MAE': ..., 'RMSE': ..., 'Pearson_r': ..., 'R2': ...},
          'client_1': {...},
          'clients_mean': {...},
          'clients_std': {...},
        }
    """
    device = cfg.training.device   # 从配置读取计算设备（"cuda" 或 "cpu"）
    out = ensure_dir(os.path.join(results_dir, "fedavg"))  # 创建并返回结果保存目录
    n_rounds = cfg.federated.federated_rounds   # 联邦训练总轮数

    # FedAvg 基线：SwinUNETR 骨干随机初始化（不依赖预训练权重）
    for c in clients:
        c.build_models(task_specs, random_init_swin=True)

    _train_t0 = time.perf_counter()

    ref_client = clients[0]
    global_state = copy.deepcopy(ref_client.swin.state_dict())

    all_losses = {f"client_{c.client_id}": [] for c in clients}

    for rnd in range(n_rounds):
        print(f"[FedAvg] 第 {rnd+1}/{n_rounds} 轮")

        local_states = []
        local_ns = []

        for c in clients:
            # 步骤 1：广播全局权重到客户端
            c.swin.load_state_dict(copy.deepcopy(global_state))

            # 步骤 2：客户端本地训练 1 epoch
            trainer = SwinTrainer(
                c.swin,
                device=device,
                lr=cfg.training.lr_swin,
                mixed_precision=cfg.training.mixed_precision,
                use_qlora=False,
            )
            losses = trainer.train(c.train_loader, n_epochs=1) # 本地训练次数为1
            all_losses[f"client_{c.client_id}"].append({"round": rnd, "loss": losses[0]})
            local_states.append(copy.deepcopy(c.swin.state_dict()))
            local_ns.append(c.n_train_samples())

        # 步骤 3：FedAvg 加权平均聚合
        global_state = _fedavg_aggregate(local_states, local_ns)

    record_timing("fedavg/train", time.perf_counter() - _train_t0)

    # ---- 评估阶段 ----
    _eval_t0 = time.perf_counter()
    results = {}
    for c in clients:
        c.swin.load_state_dict(copy.deepcopy(global_state))
        trainer = SwinTrainer(c.swin, device=device, lr=0,
                              mixed_precision=False, use_qlora=False)
        y_true, y_pred = trainer.evaluate(c.test_loader)
        results[f"client_{c.client_id}"] = compute_domain_metrics(y_true, y_pred)
    record_timing("fedavg/eval", time.perf_counter() - _eval_t0)

    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _aggregate_metrics({k: results[k] for k in client_keys})
    results["clients_mean"] = agg["mean"]
    results["clients_std"] = agg["std"]

    for c in clients:
        key = f"client_{c.client_id}"
        save_csv(pd.DataFrame(all_losses[key]), os.path.join(out, f"{key}_losses.csv"))
        save_csv(pd.DataFrame([results[key]]), os.path.join(out, f"{key}_metrics.csv"))

    if cfg.eval.save_model:
        for c in clients:
            save_model(c.swin, os.path.join(out, f"client_{c.client_id}_swin.pt"))

    return results


def _fedavg_aggregate(states: list, ns: list) -> dict:
    """
    FedAvg 加权平均聚合：w_global[key] = Σ_k (n_k / N) · w_k[key]
    """
    total = sum(ns)
    agg = copy.deepcopy(states[0])
    for key in agg:
        agg[key] = sum(s[key].float() * (n / total) for s, n in zip(states, ns))
    return agg


def _aggregate_metrics(metrics_dict: dict) -> dict:
    """计算多个客户端评估指标字典的均值和标准差。"""
    from collections import defaultdict
    vals = defaultdict(list)
    for v in metrics_dict.values():
        for k, val in v.items():
            vals[k].append(val)
    return {
        "mean": {k: float(np.nanmean(v)) for k, v in vals.items()},
        "std":  {k: float(np.nanstd(v))  for k, v in vals.items()},
    }
