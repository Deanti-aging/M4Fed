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

from training.neurofm_trainer import NeuroFMTrainer    # NeuroFM 本地训练器（支持近端项）
from evaluation.metrics import compute_domain_metrics   # 领域预测评估指标
from utils.io_utils import save_csv, save_model, ensure_dir   # 文件保存工具
from utils.timing_utils import record_timing             # 计时记录工具
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
      - NeuroFMTrainer 接收 proximal_mu 和 global_state 参数
      - 每次本地训练前将当前全局状态传入 Trainer，用于计算近端项
      - 聚合仍使用 FedAvg 加权平均（FedProx 只改变本地目标，不改变聚合策略）

    参数：
      server      : Server 实例（FedProx 中服务器只做广播）
      clients     : Client 实例列表
      cfg         : M4FedConfig 配置
      task_specs  : 任务规格字典
      results_dir : 结果保存根目录

    返回格式同 run_fedavg（client_0, client_1, clients_mean, clients_std）。
    """
    device = cfg.training.device               # 从配置读取计算设备
    mu = cfg.federated.fedprox_mu              # 近端正则化系数 μ（控制惩罚强度）
    out = ensure_dir(os.path.join(results_dir, "fedprox"))  # 创建结果保存目录

    # 所有客户端随机初始化 NeuroFM（FedProx 基线：不使用预训练骨干）
    for c in clients:
        c.build_models(task_specs, random_init_neurofm=True)   # 随机初始化骨干权重

    _train_t0 = time.perf_counter()   # 记录训练开始时间

    # 用第一个客户端的初始权重作为全局模型起点
    global_state = copy.deepcopy(clients[0].neurofm.state_dict())
    # 初始化损失记录字典（每个客户端的每轮损失）
    all_losses = {f"client_{c.client_id}": [] for c in clients}

    # ---- 联邦训练循环（每轮：广播 → FedProx 本地训练 → FedAvg 聚合） ----
    for rnd in range(cfg.federated.federated_rounds):
        print(f"[FedProx] 第 {rnd+1}/{cfg.federated.federated_rounds} 轮，μ={mu}")

        local_states, local_ns = [], []   # 本轮各客户端的本地权重和样本数

        for c in clients:
            # 步骤 1：广播全局模型权重到客户端（覆盖上轮本地训练结果）
            c.neurofm.load_state_dict(copy.deepcopy(global_state))

            # 步骤 2：FedProx 本地训练（与 FedAvg 的唯一区别在此）
            trainer = NeuroFMTrainer(
                c.neurofm,                                # 要训练的 NeuroFM 实例
                device=device,                            # 计算设备
                lr=cfg.training.lr_neurofm,               # 本地学习率
                mixed_precision=cfg.training.mixed_precision,  # 是否混合精度训练
                use_qlora=False,                          # FedProx 不使用 QLoRA
                proximal_mu=mu,                           # [FedProx专有] 近端正则化系数 μ
                global_state=copy.deepcopy(global_state), # [FedProx专有] 当前全局权重（近端项锚点）
            )
            losses = trainer.train(c.train_loader, n_epochs=1)   # 本地训练 1 epoch
            # 记录本轮损失
            all_losses[f"client_{c.client_id}"].append({"round": rnd, "loss": losses[0]})
            local_states.append(copy.deepcopy(c.neurofm.state_dict()))  # 保存本地更新权重
            local_ns.append(c.n_train_samples())   # 记录样本数

        # 步骤 3：FedAvg 加权平均聚合（FedProx 仅改变本地目标，聚合方式不变）
        global_state = _fedavg_aggregate(local_states, local_ns)

    record_timing("fedprox/train", time.perf_counter() - _train_t0)   # 记录训练总耗时

    # ---- 评估阶段 ----
    _eval_t0 = time.perf_counter()   # 记录评估开始时间
    results = {}   # 存储各客户端的评估指标
    for c in clients:
        c.neurofm.load_state_dict(copy.deepcopy(global_state))   # 分发最终全局权重
        # lr=0 表示只做推理（不更新参数）
        trainer = NeuroFMTrainer(c.neurofm, device=device, lr=0,
                                 mixed_precision=False, use_qlora=False)
        y_true, y_pred = trainer.evaluate(c.test_loader)   # 在测试集推理
        results[f"client_{c.client_id}"] = compute_domain_metrics(y_true, y_pred)  # 计算指标
    record_timing("fedprox/eval", time.perf_counter() - _eval_t0)   # 记录评估耗时

    # 汇总所有客户端指标的均值和标准差
    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _agg({k: results[k] for k in client_keys})
    results["clients_mean"] = agg["mean"]   # 跨客户端指标均值
    results["clients_std"] = agg["std"]     # 跨客户端指标标准差

    # ---- 保存结果到磁盘 ----
    for c in clients:
        key = f"client_{c.client_id}"
        save_csv(pd.DataFrame(all_losses[key]), os.path.join(out, f"{key}_losses.csv"))   # 损失曲线
        save_csv(pd.DataFrame([results[key]]), os.path.join(out, f"{key}_metrics.csv"))   # 评估指标

    if cfg.eval.save_model:
        for c in clients:
            save_model(c.neurofm, os.path.join(out, f"client_{c.client_id}_neurofm.pt"))  # 模型文件

    return results   # 返回结果字典


def _fedavg_aggregate(states, ns):
    """
    FedAvg 加权平均聚合（实现同 fedavg.py，此处复制以避免循环导入）。

    公式：w_global[key] = Σ_k (n_k / N) · w_k[key]

    参数：
      states : 各客户端的 state_dict 列表
      ns     : 各客户端的样本数列表

    返回：
      聚合后的全局 state_dict
    """
    total = sum(ns)                     # 所有客户端总样本数
    agg = copy.deepcopy(states[0])      # 以第一个客户端结构为模板
    for key in agg:
        # 逐参数名加权平均（.float() 避免整数参数精度问题）
        agg[key] = sum(s[key].float() * (n / total) for s, n in zip(states, ns))
    return agg   # 返回聚合后的全局权重字典


def _agg(d):
    """
    计算指标字典的均值和标准差（供汇总报告使用）。

    参数：
      d : {client_key: {metric_name: value}} 格式字典

    返回：
      {'mean': {metric_name: float}, 'std': {metric_name: float}}
    """
    from collections import defaultdict   # 按需导入（避免顶层循环依赖）
    vals = defaultdict(list)              # 按指标名收集各客户端的值
    for v in d.values():
        for k, val in v.items():
            vals[k].append(val)           # 追加每个客户端的该指标值
    return {
        "mean": {k: float(np.nanmean(v)) for k, v in vals.items()},   # 均值（忽略 NaN）
        "std":  {k: float(np.nanstd(v)) for k, v in vals.items()},    # 标准差（忽略 NaN）
    }
