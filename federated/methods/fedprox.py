"""
FedProx 基线（Li et al., 2020）。

与 FedAvg 的唯一区别：在本地损失函数中加入近端正则化项：
  L_local(w) + (μ/2) ‖w - w_global‖²

该正则项惩罚本地模型偏离全局模型过远，在 Non-IID 数据分布下
比 FedAvg 更稳定（局部更新不会因为数据分布差异而导致全局模型发散）。

当 μ=0 时，FedProx 退化为 FedAvg。
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import copy                            # 标准库：深拷贝（保护全局/本地状态字典）
import numpy as np                     # NumPy：计算指标均值和标准差
import pandas as pd                    # Pandas：保存损失和指标到 CSV
import os                              # 标准库：路径拼接
from typing import List                # 类型注解：客户端列表

from training.swin_trainer import SwinTrainer                  # SwinMultiTask 本地训练器（支持近端项）
from evaluation.metrics import compute_domain_metrics          # 领域预测评估指标
from utils.io_utils import save_csv, save_model, ensure_dir    # 文件保存工具
from utils.timing_utils import record_timing                   # 计时记录工具
import time                            # 标准库：高精度时间戳


def run_fedprox(
    server,           # Server 实例（FedProx 中服务器同样只做广播）
    clients: List,    # Client 实例列表
    cfg,              # 全局 M4FedConfig 配置对象
    task_specs: dict, # 预测任务规格字典
    results_dir: str, # 结果保存根目录
) -> dict:
    """
    运行 FedProx 联邦训练协议。

    与 FedAvg 的实现差异：
      - SwinTrainer 接收 proximal_mu 和 global_state 参数
      - 每次本地训练前将当前全局状态传入 Trainer，用于计算近端项
      - 聚合仍使用 FedAvg 加权平均

    返回格式同 run_fedavg（client_0, client_1, clients_mean, clients_std）。
    """
    device = cfg.training.device
    mu = cfg.federated.fedprox_mu
    out = ensure_dir(os.path.join(results_dir, "fedprox"))

    # FedProx 基线：SwinUNETR 骨干随机初始化
    for c in clients:
        c.build_models(task_specs, random_init_swin=True)

    _train_t0 = time.perf_counter()

    global_state = copy.deepcopy(clients[0].swin.state_dict())
    all_losses = {f"client_{c.client_id}": [] for c in clients}

    for rnd in range(cfg.federated.federated_rounds):
        print(f"[FedProx] 第 {rnd+1}/{cfg.federated.federated_rounds} 轮，μ={mu}")

        local_states, local_ns = [], []

        for c in clients:
            c.swin.load_state_dict(copy.deepcopy(global_state))

            trainer = SwinTrainer(
                c.swin,
                device=device,
                lr=cfg.training.lr_swin,
                mixed_precision=cfg.training.mixed_precision,
                use_qlora=False,
                proximal_mu=mu,
                global_state=copy.deepcopy(global_state),
            )
            losses = trainer.train(c.train_loader, n_epochs=1)
            all_losses[f"client_{c.client_id}"].append({"round": rnd, "loss": losses[0]})
            local_states.append(copy.deepcopy(c.swin.state_dict()))
            local_ns.append(c.n_train_samples())

        global_state = _fedavg_aggregate(local_states, local_ns)

    record_timing("fedprox/train", time.perf_counter() - _train_t0)

    # ---- 评估阶段 ----
    _eval_t0 = time.perf_counter()
    results = {}
    for c in clients:
        c.swin.load_state_dict(copy.deepcopy(global_state))
        trainer = SwinTrainer(c.swin, device=device, lr=0,
                              mixed_precision=False, use_qlora=False)
        y_true, y_pred = trainer.evaluate(c.test_loader)
        results[f"client_{c.client_id}"] = compute_domain_metrics(y_true, y_pred)
    record_timing("fedprox/eval", time.perf_counter() - _eval_t0)

    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _agg({k: results[k] for k in client_keys})
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


def _fedavg_aggregate(states, ns):
    """FedAvg 加权平均聚合：w_global[key] = Σ_k (n_k / N) · w_k[key]"""
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
