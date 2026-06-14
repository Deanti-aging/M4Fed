"""
M4Fed：完整的联邦学习协议（论文提出的方法）。

三阶段协议（对应论文图 3）：

阶段 1 — 热身（Warm-up）：
  ① TabPFN 表格补全：填充各节点元数据中的缺失字段，提升文本条件质量
  ② 服务器在公开数据上预训练 MedMD（warm_rounds 轮）
  ③ 服务器将 U-Net 权重广播给所有客户端（客户端 MedMD 与服务器对齐）

阶段 2 — 联邦循环（Federated Loop）：
  每轮：
    ① 各客户端用本地 MedMD 生成合成数据 → 发送给服务器
    ② 服务器在收到的合成数据上微调 MedMD
    ③ 服务器用自身私有数据生成合成数据 → 发送给各客户端
    ④ 各客户端用 QLoRA 在服务器合成数据上微调 MedMD

阶段 3 — 本地训练（Local Training）：
  ① 每个节点（服务器+客户端）用自身 MedMD 在本地私有数据上合成样本
  ② 将合成样本与本地真实数据混合，训练 NeuroFM（客户端使用 QLoRA）
"""
from __future__ import annotations
import math
import copy
import numpy as np
from typing import Dict, List, Tuple

from models.tabpfn_wrapper import TabPFNImputer
from training.medmd_trainer import MedMDTrainer
from training.neurofm_trainer import NeuroFMTrainer
from evaluation.metrics import compute_domain_metrics, compute_image_metrics, compute_bias_analysis
from evaluation.visualization import save_all_bias_figures
from data.openmind_dataset import OpenMindDataset, make_dataloader
from utils.io_utils import save_csv, save_model, ensure_dir

import os
import pandas as pd


def run_m4fed(
    server,
    clients: list,
    cfg,
    task_specs: dict,
    results_dir: str,
) -> dict:
    """
    运行完整的 M4Fed 三阶段联邦训练协议。

    参数：
      server      : Server 实例（拥有公开数据和私有数据）
      clients     : Client 实例列表（通常 2 个）
      cfg         : M4FedConfig 配置对象
      task_specs  : 任务规格字典
      results_dir : 结果保存根目录

    返回：
      结果字典，包含服务器和各客户端的指标，以及客户端的均值/标准差
    """
    device = cfg.training.device
    n_clients = len(clients)
    method = "m4fed"
    out = ensure_dir(os.path.join(results_dir, method))

    # ---- 初始化服务器和客户端的模型（从预训练权重加载 NeuroFM 骨干）----
    server.build_models(task_specs, random_init_neurofm=False)
    for c in clients:
        c.build_models(task_specs, random_init_neurofm=False)

    # ================================================================ #
    # 阶段 1：热身阶段                                                    #
    # ================================================================ #
    print("[M4Fed] === 阶段 1：热身 ===")

    # 步骤 1a：TabPFN 表格补全（模拟：实际中应对各节点的元数据 DataFrame 执行）
    imputer = TabPFNImputer(device=device, seed=cfg.training.seed)
    print("[M4Fed] 热身：TabPFN 表格补全完成（已模拟）。")

    # 步骤 1b：服务器在公开数据上预训练 MedMD（warm_rounds 轮）
    server_medmd_trainer = MedMDTrainer(
        server.medmd, device=device,
        lr=cfg.training.lr_medmd,
        mixed_precision=cfg.training.mixed_precision,
        use_qlora=False,
    )
    warmup_losses = []
    for r in range(cfg.federated.warm_rounds):
        loss = server_medmd_trainer.train_epoch(server.public_train_loader)
        warmup_losses.append({"round": r, "loss": loss, "node": "server"})
        print(f"  [热身] 第 {r+1}/{cfg.federated.warm_rounds} 轮  损失={loss:.4f}")

    # 步骤 1c：服务器将预训练 U-Net 权重广播给所有客户端
    server_unet_state = server.medmd.unet_state_dict()
    for c in clients:
        c.medmd.load_unet_state_dict(server_unet_state)
    print("[M4Fed] 服务器 U-Net 权重已分发给所有客户端。")

    # ================================================================ #
    # 阶段 2：联邦循环阶段                                                #
    # ================================================================ #
    print("[M4Fed] === 阶段 2：联邦循环 ===")
    fed_losses: Dict[str, List[dict]] = {
        "server": [], **{f"client_{c.client_id}": [] for c in clients}
    }

    for fed_round in range(cfg.federated.federated_rounds):
        print(f"[M4Fed] 联邦第 {fed_round+1}/{cfg.federated.federated_rounds} 轮")

        # 步骤 2a：各客户端生成合成数据并"发送"给服务器
        all_client_syn = []
        for c in clients:
            # 合成数据量 = synth_ratio_fed × 本地私有样本数
            n_syn = max(1, math.ceil(c.n_private_train() * cfg.federated.synth_ratio_fed))
            c_gen_trainer = MedMDTrainer(c.medmd, device=device,
                                         lr=cfg.training.lr_medmd,
                                         mixed_precision=cfg.training.mixed_precision,
                                         use_qlora=False)
            syn = c_gen_trainer.synthesise_dataset(
                c.train_loader, n_syn,
                noise_strength=cfg.model.noise_strength_fed,
                ddim_steps=cfg.model.ddim_steps,
            )
            all_client_syn.extend(syn)

        # 步骤 2b：服务器在收到的客户端合成数据上微调 MedMD
        syn_loader = make_dataloader(
            OpenMindDataset(all_client_syn, tuple(cfg.data.mri_shape)),
            batch_size=cfg.training.batch_size, shuffle=True,
        )
        s_trainer = MedMDTrainer(server.medmd, device=device, lr=cfg.training.lr_medmd,
                                 mixed_precision=cfg.training.mixed_precision, use_qlora=False)
        s_loss = s_trainer.train_epoch(syn_loader)
        fed_losses["server"].append({"round": fed_round, "loss": s_loss})

        # 步骤 2c：服务器从自身私有数据生成合成数据并"发送"给所有客户端
        n_server_syn = max(1, math.ceil(server.n_private_train() * cfg.federated.synth_ratio_fed))
        server_syn = MedMDTrainer(server.medmd, device=device,
                                  lr=cfg.training.lr_medmd).synthesise_dataset(
            server.private_train_loader, n_server_syn,
            noise_strength=cfg.model.noise_strength_fed,
            ddim_steps=cfg.model.ddim_steps,
        )

        # 步骤 2d：各客户端用 QLoRA 在服务器合成数据上微调 MedMD
        for c in clients:
            c_trainer = MedMDTrainer(
                c.medmd, device=device, lr=cfg.training.lr_medmd,
                mixed_precision=cfg.training.mixed_precision,
                use_qlora=True,                          # 客户端使用 QLoRA 微调（节省资源）
                qlora_r=cfg.federated.qlora_r,
                qlora_alpha=cfg.federated.qlora_alpha,
                qlora_dropout=cfg.federated.qlora_dropout,
            )
            c_loader = make_dataloader(
                OpenMindDataset(server_syn, tuple(cfg.data.mri_shape)),
                batch_size=cfg.training.batch_size, shuffle=True,
            )
            c_loss = c_trainer.train_epoch(c_loader)
            fed_losses[f"client_{c.client_id}"].append({"round": fed_round, "loss": c_loss})

    # ================================================================ #
    # 阶段 3：本地训练阶段                                                #
    # ================================================================ #
    print("[M4Fed] === 阶段 3：本地训练 ===")
    local_losses: Dict[str, List[dict]] = {
        "server": [], **{f"client_{c.client_id}": [] for c in clients}
    }

    def _synth_local(node, medmd_model, noise_range, source_loader=None):
        """
        内部辅助：在本地私有数据上生成合成样本（本地训练阶段使用）。
        噪声强度从 noise_range 随机均匀采样，增加合成多样性。
        """
        n_local = node.n_private_train()
        noise_strength = float(np.random.uniform(*noise_range))
        trainer = MedMDTrainer(medmd_model, device=device, lr=cfg.training.lr_medmd)
        loader = source_loader if source_loader is not None else node.train_loader
        return trainer.synthesise_dataset(
            loader, n_local,
            noise_strength=noise_strength,
            ddim_steps=cfg.model.ddim_steps,
        )

    noise_range = (cfg.model.noise_strength_local_min, cfg.model.noise_strength_local_max)

    # 服务器本地训练：仅使用私有数据生成合成样本（保持私有性）
    server_syn_local = _synth_local(server, server.medmd, noise_range,
                                    source_loader=server.private_train_loader)
    # 合并所有数据（公开+私有+合成）训练 NeuroFM
    server_local_loader = server.add_synthetic_data_to_loader(server_syn_local, merge_with="all")

    # 服务器训练 NeuroFM（全参数微调，服务器资源充足）
    s_nfm_trainer = NeuroFMTrainer(
        server.neurofm, device=device, lr=cfg.training.lr_neurofm,
        mixed_precision=cfg.training.mixed_precision, use_qlora=False,
    )
    for ep in range(cfg.federated.local_rounds):
        loss = s_nfm_trainer.train_epoch(server_local_loader)
        local_losses["server"].append({"round": ep, "loss": loss})

    # 各客户端本地训练：合成样本与私有数据混合，QLoRA 微调 NeuroFM
    for c in clients:
        c_syn_local = _synth_local(c, c.medmd, noise_range)
        c_local_loader = c.add_synthetic_data_to_loader(c_syn_local)
        c_nfm_trainer = NeuroFMTrainer(
            c.neurofm, device=device, lr=cfg.training.lr_neurofm,
            mixed_precision=cfg.training.mixed_precision,
            use_qlora=True,                              # 客户端使用 QLoRA 微调 NeuroFM
            qlora_r=cfg.federated.qlora_r,
            qlora_alpha=cfg.federated.qlora_alpha,
        )
        for ep in range(cfg.federated.local_rounds):
            loss = c_nfm_trainer.train_epoch(c_local_loader)
            local_losses[f"client_{c.client_id}"].append({"round": ep, "loss": loss})

    # ================================================================ #
    # 评估                                                               #
    # ================================================================ #
    print("[M4Fed] === 评估 ===")
    results = {}

    def _eval_node(node_id, neurofm, test_loader):
        """评估单个节点的脑年龄预测性能。"""
        from training.neurofm_trainer import NeuroFMTrainer as T
        trainer = T(neurofm, device=device, lr=0, mixed_precision=False)
        y_true, y_pred = trainer.evaluate(test_loader)
        if len(y_true) == 0:
            return {"MAE": float("nan"), "RMSE": float("nan"),
                    "Pearson_r": float("nan"), "R2": float("nan")}
        return compute_domain_metrics(y_true, y_pred)

    results["server"] = _eval_node("server", server.neurofm, server.test_loader)
    for c in clients:
        results[f"client_{c.client_id}"] = _eval_node(
            f"client_{c.client_id}", c.neurofm, c.test_loader
        )

    # 计算客户端的均值和标准差（用于与基线对比）
    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _aggregate_metrics({k: results[k] for k in client_keys})
    results["clients_mean"] = agg["mean"]
    results["clients_std"] = agg["std"]

    # ================================================================ #
    # 保存结果                                                            #
    # ================================================================ #
    # 服务器：热身损失 + 联邦损失 + 本地训练损失
    _save_losses(warmup_losses + fed_losses.get("server", []) + local_losses.get("server", []),
                 os.path.join(out, "server_losses.csv"))
    # 各客户端：联邦损失 + 本地训练损失
    for c in clients:
        key = f"client_{c.client_id}"
        all_c_loss = fed_losses.get(key, []) + local_losses.get(key, [])
        _save_losses(all_c_loss, os.path.join(out, f"{key}_losses.csv"))

    save_csv(pd.DataFrame([results["server"]]), os.path.join(out, "server_metrics.csv"))
    for c in clients:
        key = f"client_{c.client_id}"
        save_csv(pd.DataFrame([results[key]]), os.path.join(out, f"{key}_metrics.csv"))

    # 保存模型权重（若配置要求）
    if cfg.eval.save_model:
        save_model(server.neurofm, os.path.join(out, "server_neurofm.pt"))
        save_model(server.medmd, os.path.join(out, "server_medmd.pt"))
        for c in clients:
            save_model(c.neurofm, os.path.join(out, f"client_{c.client_id}_neurofm.pt"))

    return results


# --------------------------------------------------------------------------- #
#  内部辅助函数                                                                  #
# --------------------------------------------------------------------------- #

def _make_trainer(medmd_model, cfg, device):
    """
    创建 MedMDTrainer 实例（辅助函数，减少重复代码）。
    全量微调模式（不使用 QLoRA）。
    """
    return MedMDTrainer(medmd_model, device=device, lr=cfg.training.lr_medmd,
                        mixed_precision=cfg.training.mixed_precision, use_qlora=False)


def _aggregate_metrics(metrics_dict: dict) -> dict:
    """
    对多个节点的指标字典计算均值和标准差（忽略 NaN 值）。

    参数：
      metrics_dict : {node_key: {metric_name: value}}

    返回：
      {'mean': {metric_name: float}, 'std': {metric_name: float}}
    """
    from collections import defaultdict
    vals = defaultdict(list)
    for node_metrics in metrics_dict.values():
        for k, v in node_metrics.items():
            vals[k].append(v)
    mean = {k: float(np.nanmean(v)) for k, v in vals.items()}
    std = {k: float(np.nanstd(v)) for k, v in vals.items()}
    return {"mean": mean, "std": std}


def _save_losses(loss_list: list, path: str) -> None:
    """
    将损失记录列表保存为 CSV（列表为空时跳过）。

    每条记录为 {'round': int, 'loss': float} 格式的字典。
    """
    if loss_list:
        save_csv(pd.DataFrame(loss_list), path)
