"""
FAN 基线（Federated Augmentation Network）。

与 M4Fed 的关键区别：
  - 3 个客户端，无服务器聚合（去中心化结构）
  - 每个客户端独立拥有 MedMD 和 NeuroFM（均随机初始化）
  - 客户端间通过交换合成数据（而非模型权重）来共享知识
  - 使用 MONAI 数据增强而非条件合成（更简单的数据扩充策略）

三个阶段：
  1. 热身阶段：各客户端用增强后的本地数据训练生成和领域模型
  2. 联邦循环：各客户端生成合成数据，广播给其他客户端；
               每个客户端在接收的合成数据上微调生成模型
  3. 本地训练：各客户端合成数据并用于微调领域模型（NeuroFM）
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import copy, math, os                 # copy：深拷贝；math：ceil 向上取整；os：路径拼接
import numpy as np                    # NumPy：随机数（均匀分布采样噪声强度）、统计
import pandas as pd                   # Pandas：CSV 保存损失和指标
from typing import List               # 类型注解：客户端列表
from torch.utils.data import ConcatDataset   # PyTorch：数据集合并工具

from training.medmd_trainer import MedMDTrainer     # MedMD 生成模型训练器
from training.neurofm_trainer import NeuroFMTrainer  # NeuroFM 预测模型训练器
from evaluation.metrics import compute_domain_metrics  # 领域预测评估指标
from data.openmind_dataset import OpenMindDataset, make_dataloader  # 数据集和 DataLoader 工厂
from utils.io_utils import save_csv, save_model, ensure_dir  # 文件保存工具
from utils.timing_utils import record_timing         # 计时记录工具
import time                           # 标准库：高精度时间戳


def _augment_loader(loader, cfg):
    """
    为 DataLoader 的数据集创建带 MONAI 数据增强的新 DataLoader。

    将 OpenMindDataset 的 augment=True 标志开启，
    __getitem__ 时自动应用随机翻转等 3D 变换（MONAI Compose 增强流水线）。

    若数据集为空（无样本），直接返回原始 loader（避免空数据集错误）。

    参数：
      loader : 原始 DataLoader（含 OpenMindDataset）
      cfg    : 全局配置对象

    返回：
      带数据增强的 DataLoader（augment=True）
    """
    # 从原始 DataLoader 的 Dataset 中提取样本列表（若 Dataset 没有 .samples 属性则用空列表）
    aug_ds = OpenMindDataset(
        loader.dataset.samples if hasattr(loader.dataset, "samples") else [],
        mri_shape=tuple(cfg.data.mri_shape),   # MRI 目标分辨率
        augment=True,   # 启用 MONAI 3D 随机翻转增强（每次 __getitem__ 随机变换）
    )
    if len(aug_ds) == 0:
        return loader   # 数据集为空，返回原始 loader（避免空迭代器错误）
    # 创建带增强的新 DataLoader（shuffle=True 保证增强多样性）
    return make_dataloader(aug_ds, cfg.training.batch_size, shuffle=True,
                           seed=cfg.training.seed)


def run_fan(server, clients: List, cfg, task_specs: dict, results_dir: str) -> dict:
    """
    运行 FAN 联邦训练协议（去中心化，无服务器聚合）。

    注意：FAN 仅使用客户端，server 参数被传入但不参与训练（保持接口一致性）。

    参数：
      server      : Server 实例（FAN 中不使用，接口占位）
      clients     : Client 实例列表（FAN 要求 3 个客户端）
      cfg         : 全局 M4FedConfig 配置对象
      task_specs  : 预测任务规格字典
      results_dir : 结果保存根目录

    返回：
      {
        'client_0': 指标字典,
        'client_1': 指标字典,
        'client_2': 指标字典,
        'clients_mean': 均值,
        'clients_std': 标准差,
      }
    """
    device = cfg.training.device                             # 计算设备
    out = ensure_dir(os.path.join(results_dir, "fan"))       # 创建结果保存目录

    # FAN 每个客户端独立拥有模型，均随机初始化（无预训练骨干，对比 M4Fed 的预训练初始化）
    for c in clients:
        c.build_models(task_specs, random_init_neurofm=True)   # 骨干随机初始化

    _train_t0 = time.perf_counter()   # 记录训练总开始时间

    # ------------------------------------------------------------------ #
    # 阶段 1：热身（Warm-up）—— 各客户端在增强数据上独立训练生成和领域模型     #
    # ------------------------------------------------------------------ #
    print("[FAN] === 热身阶段 ===")
    # 为每个客户端创建带 MONAI 数据增强的 DataLoader（随机翻转等 3D 增强）
    aug_loaders = [_augment_loader(c.train_loader, cfg) for c in clients]

    # 为每个客户端创建生成模型（MedMD）和领域模型（NeuroFM）的训练器
    client_gen_trainers = []    # 生成模型（MedMD）训练器列表（每个客户端独立）
    client_dom_trainers = []    # 领域模型（NeuroFM）训练器列表（每个客户端独立）
    for i, c in enumerate(clients):
        # 为客户端 i 创建 MedMD 训练器（不使用 QLoRA，热身阶段完整训练）
        g = MedMDTrainer(c.medmd, device=device, lr=cfg.training.lr_medmd,
                         mixed_precision=cfg.training.mixed_precision, use_qlora=False)
        # 为客户端 i 创建 NeuroFM 训练器
        d = NeuroFMTrainer(c.neurofm, device=device, lr=cfg.training.lr_neurofm,
                           mixed_precision=cfg.training.mixed_precision, use_qlora=False)
        g.train_epoch(aug_loaders[i])   # 在增强数据上训练 MedMD 1 epoch（热身）
        d.train_epoch(aug_loaders[i])   # 在增强数据上训练 NeuroFM 1 epoch（热身）
        client_gen_trainers.append(g)   # 保存生成模型训练器（联邦循环阶段继续使用）
        client_dom_trainers.append(d)   # 保存领域模型训练器（本地训练阶段使用）

    # 初始化损失记录字典（联邦循环阶段的损失）
    all_losses = {f"client_{c.client_id}": [] for c in clients}

    # ------------------------------------------------------------------ #
    # 阶段 2：联邦循环（Federated Loop）—— 交换合成数据，更新生成模型           #
    # ------------------------------------------------------------------ #
    print("[FAN] === 联邦循环阶段 ===")
    for rnd in range(cfg.federated.federated_rounds):
        print(f"[FAN] 第 {rnd+1}/{cfg.federated.federated_rounds} 轮")

        # 每个客户端用当前生成模型（MedMD）生成 n_syn 个合成样本
        synth_per_client = []
        for i, c in enumerate(clients):
            # 计算合成数量：私有训练集大小 × 合成比例（向上取整，至少 1 个）
            n_syn = max(1, math.ceil(c.n_private_train() * cfg.federated.synth_ratio_fed))
            # img2img 合成：从私有数据出发，按 noise_strength 生成变体
            syn = client_gen_trainers[i].synthesise_dataset(
                c.train_loader,                          # 源图像来自私有训练数据
                n_syn,                                   # 生成数量
                noise_strength=cfg.model.noise_strength_fed,  # 合成噪声强度（联邦循环参数）
                ddim_steps=cfg.model.ddim_steps,         # DDIM 去噪步数
            )
            synth_per_client.append(syn)   # 保存合成样本列表（供其他客户端使用）

        # 每个客户端接收其他客户端的合成数据，更新自己的生成模型
        for i, c in enumerate(clients):
            external = []   # 收集来自其他客户端的合成数据
            for j, syn in enumerate(synth_per_client):
                if j != i:     # 只收集其他客户端的数据（不包括自己，避免自循环）
                    external.extend(syn)   # 将其他客户端的合成样本加入 external 列表
            if not external:
                continue   # 没有外部数据（只有 1 个客户端时），跳过
            # 将外部合成数据包装为 OpenMindDataset
            ext_ds = OpenMindDataset(external, tuple(cfg.data.mri_shape))
            aug_base = aug_loaders[i].dataset   # 客户端 i 的本地增强数据集
            # 合并本地增强数据与外部合成数据（扩充生成模型的训练分布）
            merged = ConcatDataset([aug_base, ext_ds])
            merged_loader = make_dataloader(merged, cfg.training.batch_size, shuffle=True,
                                            seed=cfg.training.seed)
            # 用合并数据训练生成模型 1 epoch，使其适应其他客户端的数据分布
            loss = client_gen_trainers[i].train_epoch(merged_loader)
            all_losses[f"client_{c.client_id}"].append({"round": rnd, "loss": loss})   # 记录损失

    # ------------------------------------------------------------------ #
    # 阶段 3：本地训练（Local Training）—— 合成增强数据，训练领域模型 NeuroFM   #
    # ------------------------------------------------------------------ #
    print("[FAN] === 本地训练阶段 ===")
    local_losses = {f"client_{c.client_id}": [] for c in clients}   # 本地训练阶段损失
    noise_range = (cfg.model.noise_strength_local_min, cfg.model.noise_strength_local_max)

    for i, c in enumerate(clients):
        n_syn = c.n_private_train()   # 生成与私有数据等量的合成样本（1:1 比例）
        # 每个客户端随机采样噪声强度（在范围内均匀采样，增加多样性）
        ns = float(np.random.uniform(*noise_range))
        # 生成合成数据（img2img，保持部分结构信息）
        syn = client_gen_trainers[i].synthesise_dataset(
            c.train_loader, n_syn,
            noise_strength=ns,                    # 本地训练阶段的噪声强度（比联邦循环更灵活）
            ddim_steps=cfg.model.ddim_steps,      # DDIM 去噪步数
        )
        aug_base = aug_loaders[i].dataset   # 客户端 i 的本地增强数据集
        syn_ds = OpenMindDataset(syn, tuple(cfg.data.mri_shape))   # 合成数据包装为 Dataset
        # 合并增强本地数据和合成数据，用于训练领域预测模型 NeuroFM
        merged_loader = make_dataloader(
            ConcatDataset([aug_base, syn_ds]),
            cfg.training.batch_size, shuffle=True, seed=cfg.training.seed,
        )
        # 本地训练 local_rounds 轮（每轮 1 epoch）
        for ep in range(cfg.federated.local_rounds):
            loss = client_dom_trainers[i].train_epoch(merged_loader)   # 训练 NeuroFM 1 epoch
            local_losses[f"client_{c.client_id}"].append({"round": ep, "loss": loss})   # 记录损失

    record_timing("fan/train", time.perf_counter() - _train_t0)   # 记录训练总耗时

    # ---- 评估阶段 ----
    _eval_t0 = time.perf_counter()   # 记录评估开始时间
    results = {}   # 各客户端的评估指标
    for c in clients:
        # 创建评估用 Trainer（lr=0，只推理不更新参数）
        t = NeuroFMTrainer(c.neurofm, device=device, lr=0, mixed_precision=False)
        y_true, y_pred = t.evaluate(c.test_loader)   # 在测试集上推理
        results[f"client_{c.client_id}"] = compute_domain_metrics(y_true, y_pred)   # 计算指标
    record_timing("fan/eval", time.perf_counter() - _eval_t0)   # 记录评估耗时

    # 计算跨客户端指标的均值和标准差
    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _agg({k: results[k] for k in client_keys})
    results["clients_mean"] = agg["mean"]   # 均值
    results["clients_std"] = agg["std"]     # 标准差

    # ---- 保存结果到磁盘 ----
    for c in clients:
        key = f"client_{c.client_id}"
        # 合并联邦循环和本地训练的损失（保存为完整损失曲线）
        losses_combined = all_losses[key] + local_losses[key]
        save_csv(pd.DataFrame(losses_combined), os.path.join(out, f"{key}_losses.csv"))
        save_csv(pd.DataFrame([results[key]]), os.path.join(out, f"{key}_metrics.csv"))
        if cfg.eval.save_model:
            save_model(c.neurofm, os.path.join(out, f"client_{c.client_id}_neurofm.pt"))
    return results   # 返回完整结果字典


def _agg(d):
    """
    计算指标字典的均值和标准差（供汇总报告使用）。

    参数：
      d : {client_key: {metric_name: value}} 格式字典

    返回：
      {'mean': {metric_name: float}, 'std': {metric_name: float}}
    """
    from collections import defaultdict   # 按需导入
    vals = defaultdict(list)              # 按指标名收集各客户端的值
    for v in d.values():
        for k, val in v.items():
            vals[k].append(val)           # 追加每个客户端的该指标值
    return {
        "mean": {k: float(np.nanmean(v)) for k, v in vals.items()},   # 均值（忽略 NaN）
        "std":  {k: float(np.nanstd(v)) for k, v in vals.items()},    # 标准差（忽略 NaN）
    }
