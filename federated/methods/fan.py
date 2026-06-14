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
from __future__ import annotations
import copy, math, os
import numpy as np
import pandas as pd
from typing import List
from torch.utils.data import ConcatDataset

from training.medmd_trainer import MedMDTrainer
from training.neurofm_trainer import NeuroFMTrainer
from evaluation.metrics import compute_domain_metrics
from data.openmind_dataset import OpenMindDataset, make_dataloader
from utils.io_utils import save_csv, save_model, ensure_dir


def _augment_loader(loader, cfg):
    """
    为 DataLoader 的数据集创建带 MONAI 数据增强的新 DataLoader。

    将 OpenMindDataset 的 augment=True 标志开启，
    __getitem__ 时自动应用随机翻转等变换。

    若数据集为空（无样本），直接返回原始 loader（避免空数据集错误）。
    """
    aug_ds = OpenMindDataset(
        loader.dataset.samples if hasattr(loader.dataset, "samples") else [],
        mri_shape=tuple(cfg.data.mri_shape),
        augment=True,   # 启用 MONAI 3D 随机翻转增强
    )
    if len(aug_ds) == 0:
        return loader
    return make_dataloader(aug_ds, cfg.training.batch_size, shuffle=True,
                           seed=cfg.training.seed)


def run_fan(server, clients: List, cfg, task_specs: dict, results_dir: str) -> dict:
    """
    运行 FAN 联邦训练协议。

    注意：FAN 仅使用客户端，服务器参数被传入但不参与训练。

    参数说明同 run_fedavg。

    返回格式：
      {
        'client_0': 指标字典,
        'client_1': 指标字典,
        'client_2': 指标字典,
        'clients_mean': 均值,
        'clients_std': 标准差,
      }
    """
    device = cfg.training.device
    out = ensure_dir(os.path.join(results_dir, "fan"))

    # FAN 每个客户端独立拥有模型，均随机初始化（无预训练骨干）
    for c in clients:
        c.build_models(task_specs, random_init_neurofm=True)

    # ------------------------------------------------------------------ #
    # 阶段 1：热身（Warm-up）                                               #
    # ------------------------------------------------------------------ #
    print("[FAN] === 热身阶段 ===")
    # 为每个客户端创建带数据增强的 DataLoader
    aug_loaders = [_augment_loader(c.train_loader, cfg) for c in clients]

    # 初始化各客户端的生成模型训练器和领域模型训练器
    client_gen_trainers = []    # 生成模型（MedMD）训练器列表
    client_dom_trainers = []    # 领域模型（NeuroFM）训练器列表
    for i, c in enumerate(clients):
        g = MedMDTrainer(c.medmd, device=device, lr=cfg.training.lr_medmd,
                         mixed_precision=cfg.training.mixed_precision, use_qlora=False)
        d = NeuroFMTrainer(c.neurofm, device=device, lr=cfg.training.lr_neurofm,
                           mixed_precision=cfg.training.mixed_precision, use_qlora=False)
        # 在增强数据上各训练 1 epoch（热身）
        g.train_epoch(aug_loaders[i])
        d.train_epoch(aug_loaders[i])
        client_gen_trainers.append(g)
        client_dom_trainers.append(d)

    all_losses = {f"client_{c.client_id}": [] for c in clients}

    # ------------------------------------------------------------------ #
    # 阶段 2：联邦循环（Federated Loop）                                    #
    # ------------------------------------------------------------------ #
    print("[FAN] === 联邦循环阶段 ===")
    for rnd in range(cfg.federated.federated_rounds):
        print(f"[FAN] 第 {rnd+1}/{cfg.federated.federated_rounds} 轮")

        # 每个客户端生成 n_syn 个合成样本（高噪声强度，保留结构但改变细节）
        synth_per_client = []
        for i, c in enumerate(clients):
            n_syn = max(1, math.ceil(c.n_private_train() * cfg.federated.synth_ratio_fed))
            syn = client_gen_trainers[i].synthesise_dataset(
                c.train_loader, n_syn,
                noise_strength=cfg.model.noise_strength_fed,
                ddim_steps=cfg.model.ddim_steps,
            )
            synth_per_client.append(syn)

        # 每个客户端收集其他客户端的合成数据，更新自己的生成模型
        for i, c in enumerate(clients):
            external = []
            for j, syn in enumerate(synth_per_client):
                if j != i:     # 只收集其他客户端的数据（不包括自己）
                    external.extend(syn)
            if not external:
                continue
            ext_ds = OpenMindDataset(external, tuple(cfg.data.mri_shape))
            aug_base = aug_loaders[i].dataset
            # 将本地增强数据与外部合成数据合并
            merged = ConcatDataset([aug_base, ext_ds])
            merged_loader = make_dataloader(merged, cfg.training.batch_size, shuffle=True,
                                            seed=cfg.training.seed)
            loss = client_gen_trainers[i].train_epoch(merged_loader)
            all_losses[f"client_{c.client_id}"].append({"round": rnd, "loss": loss})

    # ------------------------------------------------------------------ #
    # 阶段 3：本地训练（Local Training）                                    #
    # ------------------------------------------------------------------ #
    print("[FAN] === 本地训练阶段 ===")
    local_losses = {f"client_{c.client_id}": [] for c in clients}
    noise_range = (cfg.model.noise_strength_local_min, cfg.model.noise_strength_local_max)

    for i, c in enumerate(clients):
        # 生成与本地私有数据等量的合成样本（噪声强度从范围内随机采样）
        n_syn = c.n_private_train()
        ns = float(np.random.uniform(*noise_range))    # 每个客户端随机采样噪声强度
        syn = client_gen_trainers[i].synthesise_dataset(
            c.train_loader, n_syn,
            noise_strength=ns, ddim_steps=cfg.model.ddim_steps,
        )
        aug_base = aug_loaders[i].dataset
        syn_ds = OpenMindDataset(syn, tuple(cfg.data.mri_shape))
        # 合并增强本地数据和合成数据训练领域模型
        merged_loader = make_dataloader(
            ConcatDataset([aug_base, syn_ds]),
            cfg.training.batch_size, shuffle=True, seed=cfg.training.seed,
        )
        for ep in range(cfg.federated.local_rounds):
            loss = client_dom_trainers[i].train_epoch(merged_loader)
            local_losses[f"client_{c.client_id}"].append({"round": ep, "loss": loss})

    # ---- 评估 ----
    results = {}
    for c in clients:
        t = NeuroFMTrainer(c.neurofm, device=device, lr=0, mixed_precision=False)
        y_true, y_pred = t.evaluate(c.test_loader)
        results[f"client_{c.client_id}"] = compute_domain_metrics(y_true, y_pred)

    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _agg({k: results[k] for k in client_keys})
    results["clients_mean"] = agg["mean"]
    results["clients_std"] = agg["std"]

    # ---- 保存结果 ----
    for c in clients:
        key = f"client_{c.client_id}"
        losses_combined = all_losses[key] + local_losses[key]
        save_csv(pd.DataFrame(losses_combined), os.path.join(out, f"{key}_losses.csv"))
        save_csv(pd.DataFrame([results[key]]), os.path.join(out, f"{key}_metrics.csv"))
        if cfg.eval.save_model:
            save_model(c.neurofm, os.path.join(out, f"client_{c.client_id}_neurofm.pt"))
    return results


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
