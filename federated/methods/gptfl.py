"""
GPT-FL 基线（He et al., 2023）。

核心思想：
  服务器端拥有预训练生成模型（MedMD），通过接收客户端的年龄标签分布
  生成"公共合成数据集"，分发给所有客户端，替代真实公开数据集。

与 FAN 的区别：
  - 服务器可以加载 M4Fed 热身阶段训练好的 MedMD 权重（如果可用）
  - 服务器以纯文本条件（只有年龄标签，无 MRI 源图像）生成合成数据
    （即调用 MedMD.synthesise_from_label()，而非 img2img 模式）
  - 客户端收到服务器合成数据后的训练流程与 FAN 类似

三个阶段：
  1. 热身阶段：服务器收集客户端年龄标签，用 MedMD 生成"公共合成集"
  2. 联邦循环：客户端生成数据 + 服务器公共合成数据一起更新客户端 MedMD
  3. 本地训练：客户端合成数据 + 服务器公共合成数据训练 NeuroFM
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
from utils.io_utils import save_csv, save_model, ensure_dir, load_model


def run_gptfl(
    server,
    clients: List,
    cfg,
    task_specs: dict,
    results_dir: str,
    pretrained_medmd_path: str = None,
) -> dict:
    """
    运行 GPT-FL 联邦训练协议。

    参数：
      pretrained_medmd_path : M4Fed 热身阶段保存的服务器 MedMD 权重路径。
                              若存在，加载该权重以获得更好的生成质量；
                              若不存在，服务器 MedMD 随机初始化。
    """
    device = cfg.training.device
    out = ensure_dir(os.path.join(results_dir, "gptfl"))

    # 所有客户端随机初始化 NeuroFM
    for c in clients:
        c.build_models(task_specs, random_init_neurofm=True)

    # ---- 服务器加载预训练 MedMD（如果可用） ----
    if pretrained_medmd_path and os.path.exists(pretrained_medmd_path):
        from models.medmd.medmd import MedMD
        server.build_models(task_specs, random_init_neurofm=False)
        load_model(server.medmd, pretrained_medmd_path, strict=False)
        print(f"[GPT-FL] 成功加载预训练 MedMD：{pretrained_medmd_path}")
    else:
        server.build_models(task_specs, random_init_neurofm=False)
        print("[GPT-FL] 未找到预训练 MedMD，使用随机初始化 MedMD。")

    # ------------------------------------------------------------------ #
    # 阶段 1：热身（服务器根据年龄标签生成公共合成数据集）                      #
    # ------------------------------------------------------------------ #
    print("[GPT-FL] === 热身阶段 ===")

    # 从所有客户端收集年龄标签（不共享 MRI 数据，保护隐私）
    all_ages = set()
    for c in clients:
        for batch in c.train_loader:
            ages = batch["age"].numpy()
            # 过滤 NaN，收集有效年龄值（取整，作为文本条件）
            all_ages.update(int(a) for a in ages if not np.isnan(a))
    all_ages = sorted(all_ages)

    server_gen_trainer = MedMDTrainer(server.medmd, device=device, lr=cfg.training.lr_medmd,
                                       mixed_precision=cfg.training.mixed_precision,
                                       use_qlora=False)

    # 服务器使用纯文本条件（"age is {v}"）生成与公开数据集等量的合成样本
    n_pub = max(1, len(server.public_datasets["train"]))
    server_public_syn = []
    for age_val in all_ages:
        if len(server_public_syn) >= n_pub:
            break
        # synthesise_from_label：从随机噪声出发，以年龄文本条件生成脑 MRI
        batch_mri = server.medmd.synthesise_from_label(
            age_label=age_val,
            batch_size=1,
            mri_shape=tuple(cfg.data.mri_shape),
            noise_strength=cfg.model.noise_strength_fed,
            ddim_steps=cfg.model.ddim_steps,
        )
        server_public_syn.append({
            "mri_array": batch_mri[0].cpu().numpy(),
            "table_text": f"age is {age_val}",
            "age": float(age_val),
            "metadata": {},
        })

    # 初始化各客户端的训练器
    all_losses = {f"client_{c.client_id}": [] for c in clients}
    client_gen_trainers = []
    client_dom_trainers = []
    for c in clients:
        client_gen_trainers.append(
            MedMDTrainer(c.medmd, device=device, lr=cfg.training.lr_medmd,
                         mixed_precision=cfg.training.mixed_precision,
                         use_qlora=False))
        client_dom_trainers.append(
            NeuroFMTrainer(c.neurofm, device=device, lr=cfg.training.lr_neurofm,
                           mixed_precision=cfg.training.mixed_precision,
                           use_qlora=False))

    # ------------------------------------------------------------------ #
    # 阶段 2：联邦循环                                                       #
    # ------------------------------------------------------------------ #
    print("[GPT-FL] === 联邦循环阶段 ===")
    for rnd in range(cfg.federated.federated_rounds):
        print(f"[GPT-FL] 第 {rnd+1}/{cfg.federated.federated_rounds} 轮")

        # 每个客户端生成合成数据
        synth_per_client = []
        for i, c in enumerate(clients):
            n_syn = max(1, math.ceil(c.n_private_train() * cfg.federated.synth_ratio_fed))
            syn = client_gen_trainers[i].synthesise_dataset(
                c.train_loader, n_syn,
                noise_strength=cfg.model.noise_strength_fed,
                ddim_steps=cfg.model.ddim_steps,
            )
            synth_per_client.append(syn)

        # 每个客户端：用服务器公共合成数据 + 其他客户端合成数据更新生成模型
        for i, c in enumerate(clients):
            external = server_public_syn.copy()    # 服务器公共合成数据
            for j, syn in enumerate(synth_per_client):
                if j != i:
                    external.extend(syn)           # 其他客户端的合成数据
            if not external:
                continue
            ext_ds = OpenMindDataset(external, tuple(cfg.data.mri_shape))
            # 合并本地私有数据和外部合成数据
            merged = ConcatDataset([c.private_datasets["train"], ext_ds])
            merged_loader = make_dataloader(merged, cfg.training.batch_size,
                                            shuffle=True, seed=cfg.training.seed)
            loss = client_gen_trainers[i].train_epoch(merged_loader)
            all_losses[f"client_{c.client_id}"].append({"round": rnd, "loss": loss})

    # ------------------------------------------------------------------ #
    # 阶段 3：本地训练                                                       #
    # ------------------------------------------------------------------ #
    print("[GPT-FL] === 本地训练阶段 ===")
    local_losses = {f"client_{c.client_id}": [] for c in clients}
    noise_range = (cfg.model.noise_strength_local_min, cfg.model.noise_strength_local_max)
    for i, c in enumerate(clients):
        ns = float(np.random.uniform(*noise_range))
        # 客户端合成数据（img2img，用于增强本地领域多样性）
        syn = client_gen_trainers[i].synthesise_dataset(
            c.train_loader, c.n_private_train(),
            noise_strength=ns, ddim_steps=cfg.model.ddim_steps,
        )
        # 合并服务器公共合成数据和客户端自身合成数据
        aug_syn_ds = OpenMindDataset(server_public_syn + syn, tuple(cfg.data.mri_shape))
        merged_loader = make_dataloader(
            ConcatDataset([c.private_datasets["train"], aug_syn_ds]),
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
        combined = all_losses[key] + local_losses[key]
        save_csv(pd.DataFrame(combined), os.path.join(out, f"{key}_losses.csv"))
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
