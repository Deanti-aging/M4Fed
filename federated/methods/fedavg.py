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
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import copy                            # 标准库：深拷贝（用于保护全局/本地状态字典）
import numpy as np                     # NumPy：计算指标的均值和标准差
import pandas as pd                    # Pandas：保存损失曲线和指标到 CSV
import os                              # 标准库：路径拼接
from typing import List                # 类型注解：客户端列表

from training.neurofm_trainer import NeuroFMTrainer   # NeuroFM 本地训练器
from evaluation.metrics import compute_domain_metrics  # 领域预测评估指标（MAE、RMSE 等）
from utils.io_utils import save_csv, save_model, ensure_dir  # 文件保存工具
from utils.timing_utils import record_timing            # 计时记录工具
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

    参数：
      server      : Server 实例（FedAvg 中服务器只做模型广播，不做训练）
      clients     : Client 实例列表
      cfg         : M4FedConfig 配置
      task_specs  : 任务规格字典
      results_dir : 结果保存根目录

    返回：
      结果字典，格式：
        {
          'client_0': {'MAE': ..., 'RMSE': ..., 'Pearson_r': ..., 'R2': ...},
          'client_1': {...},
          'clients_mean': {...},  # 所有客户端的平均指标
          'clients_std': {...},   # 所有客户端的标准差
        }
    """
    device = cfg.training.device   # 从配置读取计算设备（"cuda" 或 "cpu"）
    out = ensure_dir(os.path.join(results_dir, "fedavg"))  # 创建并返回结果保存目录
    n_rounds = cfg.federated.federated_rounds   # 联邦训练总轮数

    # 所有客户端随机初始化 NeuroFM（FedAvg 基线：不依赖预训练骨干，体现纯联邦学习性能）
    for c in clients:
        c.build_models(task_specs, random_init_neurofm=True)   # random_init=True: 骨干随机初始化

    _train_t0 = time.perf_counter()   # 记录训练开始时间（用于计时报告）

    # 用第一个客户端的初始权重作为全局模型起点（随机初始化，所有客户端结构相同）
    ref_client = clients[0]   # 取第一个客户端作为参考
    global_state = copy.deepcopy(ref_client.neurofm.state_dict())   # 深拷贝初始全局权重

    # 初始化损失记录字典（每个客户端的每轮损失存储为列表）
    all_losses = {f"client_{c.client_id}": [] for c in clients}

    # ---- 联邦训练循环（每轮：广播 → 本地训练 → 聚合） ----
    for rnd in range(n_rounds):
        print(f"[FedAvg] 第 {rnd+1}/{n_rounds} 轮")

        local_states = []    # 收集各客户端本地训练后的模型权重
        local_ns = []        # 收集各客户端的训练样本数（用于加权平均）

        for c in clients:
            # 步骤 1：广播（服务器将全局模型权重覆盖到客户端）
            c.neurofm.load_state_dict(copy.deepcopy(global_state))   # 深拷贝防止状态污染

            # 步骤 2：客户端本地训练 1 epoch（在私有数据上优化 NeuroFM）
            trainer = NeuroFMTrainer(
                c.neurofm,                              # 要训练的 NeuroFM 实例
                device=device,                          # 计算设备
                lr=cfg.training.lr_neurofm,             # 本地学习率
                mixed_precision=cfg.training.mixed_precision,  # 是否使用混合精度
                use_qlora=False,                        # FedAvg 不使用 QLoRA（标准 SGD）
            )
            losses = trainer.train(c.train_loader, n_epochs=1)   # 训练 1 epoch，返回损失列表
            # 记录本轮损失（取第一个 epoch 的损失）
            all_losses[f"client_{c.client_id}"].append({"round": rnd, "loss": losses[0]})
            local_states.append(copy.deepcopy(c.neurofm.state_dict()))  # 保存本地更新后的权重
            local_ns.append(c.n_train_samples())   # 记录样本数（用于聚合权重计算）

        # 步骤 3：服务器执行 FedAvg 加权平均聚合（样本多的客户端权重更大）
        global_state = _fedavg_aggregate(local_states, local_ns)

    record_timing("fedavg/train", time.perf_counter() - _train_t0)   # 记录训练总耗时

    # ---- 评估阶段：将最终全局模型分发给各客户端并在测试集上评估 ----
    _eval_t0 = time.perf_counter()   # 记录评估开始时间
    results = {}   # 存储各客户端的评估指标
    for c in clients:
        c.neurofm.load_state_dict(copy.deepcopy(global_state))   # 分发最终全局模型
        # 创建评估用 Trainer（lr=0 表示不更新参数，仅推理）
        trainer = NeuroFMTrainer(c.neurofm, device=device, lr=0,
                                 mixed_precision=False, use_qlora=False)
        y_true, y_pred = trainer.evaluate(c.test_loader)   # 在测试集上推理，返回真值和预测值
        results[f"client_{c.client_id}"] = compute_domain_metrics(y_true, y_pred)  # 计算评估指标
    record_timing("fedavg/eval", time.perf_counter() - _eval_t0)   # 记录评估总耗时

    # 计算客户端指标的均值和标准差（用于汇总报告）
    client_keys = [f"client_{c.client_id}" for c in clients]   # 所有客户端的键名列表
    agg = _aggregate_metrics({k: results[k] for k in client_keys})   # 聚合各客户端指标
    results["clients_mean"] = agg["mean"]   # 各指标均值（跨客户端平均）
    results["clients_std"] = agg["std"]     # 各指标标准差（跨客户端波动）

    # ---- 保存结果到磁盘 ----
    for c in clients:
        key = f"client_{c.client_id}"   # 当前客户端的键名
        # 保存该客户端的损失曲线（每轮 loss）
        save_csv(pd.DataFrame(all_losses[key]), os.path.join(out, f"{key}_losses.csv"))
        # 保存该客户端的评估指标（MAE、RMSE 等）
        save_csv(pd.DataFrame([results[key]]), os.path.join(out, f"{key}_metrics.csv"))

    if cfg.eval.save_model:
        # 若配置要求保存模型，则为每个客户端保存最终 NeuroFM 权重
        for c in clients:
            save_model(c.neurofm, os.path.join(out, f"client_{c.client_id}_neurofm.pt"))

    return results   # 返回完整结果字典（供实验脚本汇总报告）


def _fedavg_aggregate(states: list, ns: list) -> dict:
    """
    执行 FedAvg 加权平均聚合。

    公式：w_global[key] = Σ_k (n_k / N) · w_k[key]
    各参数张量按样本数权重加权平均，得到全局模型参数。

    参数：
      states : 各客户端本地模型的 state_dict 列表（本轮训练后的权重）
      ns     : 各客户端的训练样本数列表（与 states 一一对应）

    返回：
      聚合后的全局 state_dict（可直接传入 load_state_dict）
    """
    total = sum(ns)                     # N = 所有客户端的总样本数（分母）
    agg = copy.deepcopy(states[0])      # 以第一个客户端的结构为模板（深拷贝）
    for key in agg:
        # 逐参数名计算加权平均：Σ_k (n_k/N) · w_k[key]
        # 转 float 避免整数类型参数（如 running_mean）在求和时精度问题
        agg[key] = sum(s[key].float() * (n / total) for s, n in zip(states, ns))
    return agg   # 返回聚合后的全局状态字典


def _aggregate_metrics(metrics_dict: dict) -> dict:
    """
    计算多个客户端评估指标字典的均值和标准差。

    参数：
      metrics_dict : {client_key: {metric_name: value}} 格式的字典

    返回：
      {'mean': {metric_name: float}, 'std': {metric_name: float}}
    """
    from collections import defaultdict   # 局部导入：按需加载（避免顶层循环依赖）
    vals = defaultdict(list)              # 按指标名收集所有客户端的值
    for v in metrics_dict.values():
        for k, val in v.items():
            vals[k].append(val)           # 将每个客户端的指标值追加到对应列表
    return {
        "mean": {k: float(np.nanmean(v)) for k, v in vals.items()},   # 均值（忽略 NaN）
        "std":  {k: float(np.nanstd(v))  for k, v in vals.items()},   # 标准差（忽略 NaN）
    }
