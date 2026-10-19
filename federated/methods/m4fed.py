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
  ② 将合成样本与本地真实数据混合，训练 SwinMultiTask（客户端使用 QLoRA）
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import math                            # 数学工具：ceil（向上取整，计算合成数量）
import copy                            # 标准库：深拷贝（备用，供权重传输时保护状态）
import numpy as np                     # NumPy：随机数（均匀采样噪声强度）、统计计算
from typing import Dict, List, Tuple   # 类型注解工具

from models.tabpfn_wrapper import TabPFNImputer            # TabPFN 表格缺失值填充器
from training.medmd_trainer import MedMDTrainer             # MedMD 生成模型训练器
from training.swin_trainer import SwinTrainer                # SwinMultiTask 预测模型训练器
from evaluation.metrics import (                            # 评估指标函数
    compute_domain_metrics, compute_image_metrics, compute_bias_analysis
)
from evaluation.generative_eval import (                    # 生成模型评估与汇总工具
    evaluate_generative_model, aggregate_metrics, save_summary, save_loss_summary,
)
from evaluation.visualization import save_all_bias_figures  # 偏差分析可视化保存工具
from data.openmind_dataset import OpenMindDataset, make_dataloader, _PATIENT_FIELDS  # 数据集和 DataLoader 工厂
from data.table_to_text import table_row_to_text             # 元数据行 → 自然语言条件文本
from utils.io_utils import save_csv, save_model, ensure_dir  # 文件保存工具
from utils.timing_utils import record_timing                # 计时记录工具

import os                              # 标准库：路径拼接
import json                            # 标准库：保存年龄 z-score 统计量
import time                            # 标准库：高精度时间戳
import pandas as pd                    # Pandas：损失和指标的 CSV 保存


def _impute_node_metadata(samples: list, imputer: TabPFNImputer, protected_fields: set) -> None:
    """
    对一组样本的元数据做 TabPFN 表格补全，原地更新每个样本的 'metadata' 和 'table_text'。

    填充值的使用范围按用途区分（关键设计）：
      - table_text（生成模型 MedMD 的文本条件，热身/联邦循环阶段训练/合成时使用）：
        _PATIENT_FIELDS 全部列（含 protected_fields，即 age/sex/health_status/race）
        都参与填充，尽可能让条件文本完整，提升多模态生成质量。
      - metadata（本地训练阶段 SwinMultiTask 领域模型的标签来源，
        见 SwinTrainer._build_targets 直接从 batch["metadata"] 取值）：
        protected_fields 永远保持原始真值（缺失则仍为 NaN，训练时被 NaN mask 跳过）。
        绝不能写入填充值，否则等于用填充器的预测冒充监督标签去训练分类/回归头。

    因此同一行元数据会产生两个版本：写回 metadata 的"标签保真版"（仅补非标签字段）
    和仅用于生成 table_text 的"完整补全版"（标签字段也参与填充，但不落盘到 metadata）。

    参数：
      samples          : 样本字典列表（如 dataset.samples，函数会原地修改）
      imputer          : 共享的 TabPFNImputer 实例（跨节点复用，避免重复加载模型）
      protected_fields : 当前 task_specs 的键（领域模型的预测标签字段）
    """
    if not samples:
        return   # 空数据集（如该节点未分配到子数据集）无需处理

    meta_df = pd.DataFrame([s["metadata"] for s in samples]).reset_index(drop=True)
    imputable_cols = [c for c in _PATIENT_FIELDS if c in meta_df.columns]
    if not imputable_cols:
        return   # 元数据中没有可识别的人口学字段

    # 对全部人口学字段（含标签字段）做一次填充，得到仅用于生成文本条件的"完整补全版"
    filled = imputer.fit_transform(meta_df[imputable_cols])
    text_df = meta_df.copy()
    text_df[imputable_cols] = filled[imputable_cols]

    # 写回 metadata 时只更新非标签字段，标签字段（protected_fields）保持原始真值不变
    writable_cols = [c for c in imputable_cols if c not in protected_fields]
    if writable_cols:
        meta_df[writable_cols] = filled[writable_cols]

    for i, s in enumerate(samples):
        s["metadata"] = meta_df.iloc[i]                                        # 标签保真版（供领域模型训练）
        s["table_text"] = table_row_to_text(text_df.iloc[i].reindex(_PATIENT_FIELDS))  # 完整补全版（供生成模型条件）


def run_m4fed(
    server,           # Server 实例（拥有公开数据和私有数据，负责热身和聚合）
    clients: list,    # Client 实例列表（通常 2 个，持有私有数据）
    cfg,              # 全局 M4FedConfig 配置对象
    task_specs: dict, # 预测任务规格字典（任务名 → 类别数/连续类型）
    results_dir: str, # 结果保存根目录（会在其下创建 "m4fed/" 子目录）
    age_zscore: bool = False,  # 是否对领域模型的年龄训练标签做 z-score 标准化
) -> dict:
    """
    运行完整的 M4Fed 三阶段联邦训练协议（论文核心算法）。

    参数：
      server      : Server 实例（拥有公开数据和私有数据）
      clients     : Client 实例列表（通常 2 个）
      cfg         : M4FedConfig 配置对象
      task_specs  : 任务规格字典
      results_dir : 结果保存根目录
      age_zscore  : True 时每个节点用「自身真实训练集」的年龄均值/标准差对 SwinMultiTask
                    的年龄目标做 z-score（各节点独立统计，不跨节点共享标签统计量），评估时
                    反变换回原始年龄再算指标；统计量写入 m4fed/age_zscore_stats.json，
                    供事后从 checkpoint 推理时反变换。只作用于领域模型，MedMD 的文本条件、
                    TabPFN 补全与合成样本携带的标签均保持原始年龄。默认 False，行为不变。

    返回：
      结果字典，包含服务器和各客户端的指标，以及客户端的均值/标准差
    """
    device = cfg.training.device             # 从配置读取计算设备
    n_clients = len(clients)                 # 客户端总数（用于日志）
    method = "m4fed"                         # 方法名（用于目录命名和计时标签）
    out = ensure_dir(os.path.join(results_dir, method))   # 创建结果保存目录

    # M4Fed 使用预训练 SwinUNETR 骨干（random_init=False），这是与 FedAvg/FedProx 的重要区别
    server.build_models(task_specs, random_init_swin=False)
    for c in clients:
        c.build_models(task_specs, random_init_swin=False)

    _train_t0 = time.perf_counter()   # 记录训练总开始时间

    # ================================================================ #
    # 阶段 1：热身阶段（Warm-up Phase）                                   #
    # ================================================================ #
    print("[M4Fed] === 阶段 1：热身 ===")

    # 步骤 1a：TabPFN 表格补全
    # 填充各节点元数据中缺失的人口学字段，提升文本条件质量：
    #   - 生成模型（MedMD）的文本条件：允许使用 age/sex/health_status/race 等
    #     标签字段的填充值，热身预训练和联邦循环阶段合成/微调时都会用到更完整的条件文本
    #   - 领域模型（SwinMultiTask）的监督标签：本地训练阶段直接从 metadata 取真值，
    #     绝不能是填充值，否则等于用伪造标签训练分类/回归头（详见 _impute_node_metadata）
    imputer = TabPFNImputer(device=device, seed=cfg.training.seed)   # 初始化 TabPFN 填充器
    protected_fields = set(task_specs.keys())   # 领域模型的预测标签字段，metadata 中必须保持原始真值
    node_sample_lists = [
        server.private_datasets["train"].samples,
        server.private_datasets["test"].samples,
        server.public_datasets["train"].samples,
    ] + [ds.samples for c in clients for ds in c.private_datasets.values()]
    for samples in node_sample_lists:
        _impute_node_metadata(samples, imputer, protected_fields)
    print("[M4Fed] 热身：TabPFN 表格补全完成。")

    # 步骤 1b：服务器在公开数据上预训练 MedMD（warm_rounds 轮）
    # 公开数据分布广，有助于 U-Net 学习通用的去噪方向
    server_medmd_trainer = MedMDTrainer(
        server.medmd,                               # 服务器的 MedMD 实例
        device=device,                              # 计算设备
        lr=cfg.training.lr_medmd,                  # 热身阶段学习率
        mixed_precision=cfg.training.mixed_precision,  # 是否混合精度训练
        use_qlora=False,                            # 服务器热身阶段完整训练（非 QLoRA）
    )
    warmup_losses = []   # 收集热身阶段的每轮损失
    for r in range(cfg.federated.warm_rounds):
        # 在公开训练数据上训练 1 epoch，返回该 epoch 的平均损失
        loss = server_medmd_trainer.train_epoch(server.public_train_loader)
        warmup_losses.append({"phase": "warmup", "round": r, "loss": loss, "node": "server"})   # 记录阶段、轮次和损失
        print(f"  [热身] 第 {r+1}/{cfg.federated.warm_rounds} 轮  损失={loss:.4f}")

    # 步骤 1c：服务器将预训练好的 U-Net 权重广播给所有客户端
    # 客户端的 MedMD 起点与服务器对齐，避免各客户端在不同初始点出发导致的分歧
    server_unet_state = server.medmd.unet_state_dict()   # 获取服务器 U-Net 的深拷贝权重
    for c in clients:
        c.medmd.load_unet_state_dict(server_unet_state)   # 将服务器权重加载到客户端 U-Net
    print("[M4Fed] 服务器 U-Net 权重已分发给所有客户端。")

    # ================================================================ #
    # 阶段 2：联邦循环阶段（Federated Loop Phase）                        #
    # ================================================================ #
    print("[M4Fed] === 阶段 2：联邦循环 ===")
    # 初始化联邦循环阶段的损失记录字典（服务器 + 所有客户端）
    fed_losses: Dict[str, List[dict]] = {
        "server": [], **{f"client_{c.client_id}": [] for c in clients}
    }

    for fed_round in range(cfg.federated.federated_rounds):
        print(f"[M4Fed] 联邦第 {fed_round+1}/{cfg.federated.federated_rounds} 轮")

        # 步骤 2a：各客户端用本地 MedMD 生成合成数据，"发送"给服务器
        # 合成数据包含客户端私有数据的分布信息，但不暴露真实 MRI（隐私保护）
        all_client_syn = []   # 收集所有客户端生成的合成样本
        for c in clients:
            # 合成数量 = synth_ratio_fed × 私有训练集大小（向上取整，至少 1 个）
            n_syn = max(1, math.ceil(c.n_private_train() * cfg.federated.synth_ratio_fed))
            # 为客户端创建临时 MedMD 训练器（用于 synthesise_dataset 调用）
            c_gen_trainer = MedMDTrainer(
                c.medmd,                               # 客户端的 MedMD 实例
                device=device,                         # 计算设备
                lr=cfg.training.lr_medmd,              # 学习率（此处用于创建 trainer，但不训练）
                mixed_precision=cfg.training.mixed_precision,
                use_qlora=False,                       # 生成阶段不使用 QLoRA
            )
            # img2img 合成：从私有数据出发，按 noise_strength 生成包含目标分布信息的合成 MRI
            syn = c_gen_trainer.synthesise_dataset(
                c.train_loader,                            # 源图像来自私有训练数据
                n_syn,                                     # 生成数量
                noise_strength=cfg.model.noise_strength_fed,  # 联邦循环阶段的噪声强度
                ddim_steps=cfg.model.ddim_steps,           # DDIM 去噪步数
            )
            all_client_syn.extend(syn)   # 将该客户端的合成样本加入合集

        # 步骤 2b：服务器在收到的所有客户端合成数据上微调 MedMD
        # 目的：使服务器 MedMD 适应所有客户端的数据分布（跨节点知识传递）
        syn_loader = make_dataloader(
            OpenMindDataset(all_client_syn, tuple(cfg.data.mri_shape)),  # 合成样本包装为 Dataset
            batch_size=cfg.training.batch_size,   # 批大小
            shuffle=True,                          # 打乱顺序（各客户端数据混合）
        )
        # 服务器 MedMD 训练器（全量微调，服务器资源充足）
        s_trainer = MedMDTrainer(server.medmd, device=device, lr=cfg.training.lr_medmd,
                                 mixed_precision=cfg.training.mixed_precision, use_qlora=False)
        s_loss = s_trainer.train_epoch(syn_loader)   # 服务器在合成数据上训练 1 epoch
        fed_losses["server"].append({"phase": "federated", "round": fed_round, "loss": s_loss})   # 记录服务器损失

        # 步骤 2c：服务器从自身私有数据生成合成数据，"发送"给所有客户端
        # 使客户端 MedMD 适应服务器节点的数据分布（反向知识传递）
        n_server_syn = max(1, math.ceil(server.n_private_train() * cfg.federated.synth_ratio_fed))
        server_syn = MedMDTrainer(server.medmd, device=device,
                                  lr=cfg.training.lr_medmd).synthesise_dataset(
            server.private_train_loader,               # 源图像来自服务器私有数据
            n_server_syn,                              # 生成数量
            noise_strength=cfg.model.noise_strength_fed,  # 噪声强度
            ddim_steps=cfg.model.ddim_steps,           # DDIM 步数
        )

        # 步骤 2d：各客户端用 QLoRA 在服务器合成数据上微调 MedMD
        # QLoRA 参数高效微调：只训练少量 LoRA 适配器参数，降低计算和通信开销
        for c in clients:
            c_trainer = MedMDTrainer(
                c.medmd, device=device, lr=cfg.training.lr_medmd,
                mixed_precision=cfg.training.mixed_precision,
                use_qlora=True,                          # [M4Fed 专有] 客户端使用 QLoRA 微调
                qlora_r=cfg.federated.qlora_r,           # LoRA 秩（控制适配器参数量）
                qlora_alpha=cfg.federated.qlora_alpha,   # LoRA 缩放系数
                qlora_dropout=cfg.federated.qlora_dropout,  # LoRA Dropout 率
            )
            # 将服务器合成数据包装为 DataLoader（客户端在此上微调）
            c_loader = make_dataloader(
                OpenMindDataset(server_syn, tuple(cfg.data.mri_shape)),
                batch_size=cfg.training.batch_size, shuffle=True,
            )
            c_loss = c_trainer.train_epoch(c_loader)   # QLoRA 微调 1 epoch
            fed_losses[f"client_{c.client_id}"].append({"phase": "federated", "round": fed_round, "loss": c_loss})

    # ================================================================ #
    # 阶段 3：本地训练阶段（Local Training Phase）                        #
    # ================================================================ #
    print("[M4Fed] === 阶段 3：本地训练 ===")
    # 初始化本地训练阶段的损失记录字典
    local_losses: Dict[str, List[dict]] = {
        "server": [], **{f"client_{c.client_id}": [] for c in clients}
    }

    def _synth_local(node, medmd_model, noise_range, source_loader=None):
        """
        内部辅助函数：在本地私有数据上生成合成样本（本地训练阶段使用）。

        噪声强度从 noise_range 随机均匀采样，增加合成多样性，
        防止每次生成的合成数据过于相似。

        参数：
          node        : 节点对象（Server 或 Client），用于获取私有数据量
          medmd_model : 该节点的 MedMD 实例（用于合成）
          noise_range : 噪声强度范围 (min, max)，随机采样
          source_loader: 可选的源数据 DataLoader；若为 None，用 node.train_loader

        返回：
          合成样本字典列表（与 OpenMindDataset 格式兼容）
        """
        n_local = node.n_private_train()   # 私有训练集大小（生成等量合成样本）
        noise_strength = float(np.random.uniform(*noise_range))   # 在范围内随机采样噪声强度
        trainer = MedMDTrainer(medmd_model, device=device, lr=cfg.training.lr_medmd)
        loader = source_loader if source_loader is not None else node.train_loader   # 使用指定源数据
        return trainer.synthesise_dataset(
            loader, n_local,                   # 源数据 DataLoader 和生成数量
            noise_strength=noise_strength,     # 采样的噪声强度
            ddim_steps=cfg.model.ddim_steps,   # DDIM 去噪步数
        )

    noise_range = (cfg.model.noise_strength_local_min, cfg.model.noise_strength_local_max)

    # ---- 年龄标签 z-score 统计量（仅 age_zscore=True 时）：各节点只用自己的真实训练标签 ----
    # 服务器领域模型在 公开+私有 训练集上训练（merge_with="all"），客户端只用私有训练集；
    # 合成样本的年龄标签复制自这些真实样本，分布相同，不单独计入。
    age_norms = {"server": None, **{f"client_{c.client_id}": None for c in clients}}
    if age_zscore:
        age_norms["server"] = _age_stats(server.public_datasets["train"].samples
                                         + server.private_datasets["train"].samples)
        for c in clients:
            age_norms[f"client_{c.client_id}"] = _age_stats(c.private_datasets["train"].samples)
        with open(os.path.join(out, "age_zscore_stats.json"), "w", encoding="utf-8") as f:
            json.dump({k: {"mean": v[0], "std": v[1], "n": v[2]} for k, v in age_norms.items()},
                      f, ensure_ascii=False, indent=2)
        for k, v in age_norms.items():
            print(f"  [年龄 z-score] {k}: mean={v[0]:.3f} std={v[1]:.3f} (n={v[2]})")
        age_norms = {k: v[:2] for k, v in age_norms.items()}

    # ---- 服务器本地训练：仅使用私有数据生成合成样本（严格保持私有性）----
    server_syn_local = _synth_local(
        server, server.medmd, noise_range,
        source_loader=server.private_train_loader,   # 服务器用私有训练数据作为源图像
    )
    server_local_loader = server.add_synthetic_data_to_loader(server_syn_local, merge_with="all")

    # 服务器 SwinMultiTask 训练器（全参数微调）
    s_nfm_trainer = SwinTrainer(
        server.swin,
        device=device,
        lr=cfg.training.lr_swin,
        mixed_precision=cfg.training.mixed_precision,
        use_qlora=False,
        age_norm=age_norms["server"],
    )
    for ep in range(cfg.federated.local_rounds):
        loss = s_nfm_trainer.train_epoch(server_local_loader)
        local_losses["server"].append({"phase": "local", "round": ep, "loss": loss})

    # ---- 各客户端本地训练：合成样本与私有数据混合，QLoRA 微调 SwinMultiTask ----
    for c in clients:
        c_syn_local = _synth_local(c, c.medmd, noise_range)
        c_local_loader = c.add_synthetic_data_to_loader(c_syn_local)
        c_nfm_trainer = SwinTrainer(
            c.swin,
            device=device,
            lr=cfg.training.lr_swin,
            mixed_precision=cfg.training.mixed_precision,
            use_qlora=True,                        # [M4Fed 专有] 客户端使用 QLoRA 微调
            qlora_r=cfg.federated.qlora_r,
            qlora_alpha=cfg.federated.qlora_alpha,
            age_norm=age_norms[f"client_{c.client_id}"],
        )
        for ep in range(cfg.federated.local_rounds):
            loss = c_nfm_trainer.train_epoch(c_local_loader)
            local_losses[f"client_{c.client_id}"].append({"phase": "local", "round": ep, "loss": loss})

    record_timing(f"{method}/train", time.perf_counter() - _train_t0)   # 记录训练总耗时

    # ================================================================ #
    # 评估阶段                                                            #
    # ================================================================ #
    print("[M4Fed] === 评估 ===")
    _eval_t0 = time.perf_counter()   # 记录评估开始时间
    results = {}   # 存储所有节点的评估指标

    def _eval_node(node_id, swin_model, test_loader, return_raw: bool = False):
        """评估单个节点的脑年龄预测性能（推理模式，不更新参数）。"""
        from training.swin_trainer import SwinTrainer as T
        trainer = T(swin_model, device=device, lr=0, mixed_precision=False,
                    age_norm=age_norms[node_id])
        y_true, y_pred = trainer.evaluate(test_loader)
        if len(y_true) == 0:
            metrics = {"MAE": float("nan"), "RMSE": float("nan"),
                       "Pearson_r": float("nan"), "R2": float("nan")}
        else:
            metrics = compute_domain_metrics(y_true, y_pred)
        return (metrics, y_true, y_pred) if return_raw else metrics

    results["server"], server_y_true, server_y_pred = _eval_node(
        "server", server.swin, server.test_loader, return_raw=True
    )
    for c in clients:
        results[f"client_{c.client_id}"] = _eval_node(
            f"client_{c.client_id}", c.swin, c.test_loader
        )
    record_timing(f"{method}/eval", time.perf_counter() - _eval_t0)   # 记录评估耗时

    # ---- 服务器 domain model 偏差分析（样本量比 cross_site 更大，分组统计更稳） ----
    # test_loader（shuffle=False）与 private_datasets["test"].samples 顺序一致，
    # 因此可以直接按下标对齐元数据。
    if len(server_y_true) > 0:
        server_metadata = [s.get("metadata", {}) for s in server.private_datasets["test"].samples]
        server_bias = compute_bias_analysis(server_y_true, server_y_pred, server_metadata)
        results["server_bias"] = server_bias
        save_all_bias_figures(
            server_bias, out,
            method_name="m4fed_server",
            metadata=server_metadata,
            fmt=cfg.eval.figure_format,
        )

    # 计算客户端指标的均值和标准差（用于与基线对比的汇总统计）
    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _aggregate_metrics({k: results[k] for k in client_keys})
    results["clients_mean"] = agg["mean"]   # 跨客户端均值
    results["clients_std"] = agg["std"]     # 跨客户端标准差

    # ---- 生成模型评估（服务器 + 各客户端 MedMD：PSNR / SSIM / LPIPS / FID / KID） ----
    # M4Fed 中服务器和每个客户端都持有各自训练好的 MedMD，逐节点用其本地测试数据评估
    _gen_t0 = time.perf_counter()
    gen_results = {}
    gen_results["server"] = evaluate_generative_model(
        server.medmd, server.test_loader, cfg, device)
    for c in clients:
        gen_results[f"client_{c.client_id}"] = evaluate_generative_model(
            c.medmd, c.test_loader, cfg, device)
    gen_agg = aggregate_metrics({k: gen_results[k] for k in client_keys})
    gen_results["clients_mean"] = gen_agg["mean"]   # 跨客户端均值
    gen_results["clients_std"] = gen_agg["std"]     # 跨客户端标准差
    results["gen"] = gen_results
    record_timing(f"{method}/gen_eval", time.perf_counter() - _gen_t0)

    # ================================================================ #
    # 保存结果到磁盘                                                       #
    # ================================================================ #
    # 服务器：热身、联邦循环、本地训练三阶段各自的 round 计数独立从 0 开始，
    # 若合并到同一份 CSV 会导致 round 列跨阶段重复、无法区分，因此按阶段分文件保存
    _save_losses(warmup_losses, os.path.join(out, "server_warmup_losses.csv"))
    _save_losses(fed_losses.get("server", []), os.path.join(out, "server_federated_losses.csv"))
    _save_losses(local_losses.get("server", []), os.path.join(out, "server_local_losses.csv"))
    # 各客户端：联邦循环损失、本地训练损失分开保存
    for c in clients:
        key = f"client_{c.client_id}"
        _save_losses(fed_losses.get(key, []), os.path.join(out, f"{key}_federated_losses.csv"))
        _save_losses(local_losses.get(key, []), os.path.join(out, f"{key}_local_losses.csv"))

    # 保存领域评估指标 CSV（每个节点独立保存）
    save_csv(pd.DataFrame([results["server"]]), os.path.join(out, "server_metrics.csv"))
    for c in clients:
        key = f"client_{c.client_id}"
        save_csv(pd.DataFrame([results[key]]), os.path.join(out, f"{key}_metrics.csv"))

    # 保存生成评估指标 CSV（服务器 + 每个客户端）
    save_csv(pd.DataFrame([gen_results["server"]]), os.path.join(out, "server_gen_metrics.csv"))
    for c in clients:
        key = f"client_{c.client_id}"
        save_csv(pd.DataFrame([gen_results[key]]), os.path.join(out, f"{key}_gen_metrics.csv"))

    # 跨客户端汇总 CSV：领域指标、生成指标、逐轮损失的 mean/std
    save_summary(results["clients_mean"], results["clients_std"],
                 os.path.join(out, "clients_domain_summary.csv"))
    save_summary(gen_results["clients_mean"], gen_results["clients_std"],
                 os.path.join(out, "clients_gen_summary.csv"))
    # 联邦循环、本地训练两个阶段的 round 计数各自独立从 0 开始，汇总时同样分开存，
    # 避免两阶段的 round 位置被错误对齐合并（与上面按阶段分文件保存的原则一致）
    save_loss_summary(
        {f"client_{c.client_id}": fed_losses.get(f"client_{c.client_id}", []) for c in clients},
        os.path.join(out, "clients_federated_loss_summary.csv"),
    )
    save_loss_summary(
        {f"client_{c.client_id}": local_losses.get(f"client_{c.client_id}", []) for c in clients},
        os.path.join(out, "clients_local_loss_summary.csv"),
    )

    # 保存模型权重（若配置要求）：领域模型 + 生成模型，服务器与全部客户端
    if cfg.eval.save_model:
        save_model(server.swin, os.path.join(out, "server_swin.pt"))     # 服务器领域模型
        save_model(server.medmd, os.path.join(out, "server_medmd.pt"))   # 服务器生成模型
        for c in clients:
            key = f"client_{c.client_id}"
            save_model(c.swin, os.path.join(out, f"{key}_swin.pt"))       # 客户端领域模型
            save_model(c.medmd, os.path.join(out, f"{key}_medmd.pt"))     # 客户端生成模型

    return results   # 返回完整结果字典


# --------------------------------------------------------------------------- #
#  内部辅助函数                                                                  #
# --------------------------------------------------------------------------- #

def _make_trainer(medmd_model, cfg, device):
    """
    创建 MedMDTrainer 实例的辅助函数（减少重复代码）。
    全量微调模式（use_qlora=False），不使用 QLoRA。

    参数：
      medmd_model : MedMD 实例
      cfg         : 全局配置对象
      device      : 计算设备

    返回：
      MedMDTrainer 实例
    """
    return MedMDTrainer(medmd_model, device=device, lr=cfg.training.lr_medmd,
                        mixed_precision=cfg.training.mixed_precision, use_qlora=False)


def _age_stats(samples: list) -> Tuple[float, float, int]:
    """真实训练样本年龄标签的 (mean, std, n)，跳过不可用 / 缺失年龄的样本；std 过小时取 1。"""
    ages = np.array([float(s["age"]) for s in samples
                     if not s.get("_unusable") and s.get("age") is not None], dtype=float)
    ages = ages[np.isfinite(ages)]
    if ages.size == 0:
        return 0.0, 1.0, 0
    std = float(np.std(ages))
    return float(np.mean(ages)), (std if std > 1e-6 else 1.0), int(ages.size)


def _aggregate_metrics(metrics_dict: dict) -> dict:
    """
    对多个节点的指标字典计算均值和标准差（忽略 NaN 值）。

    参数：
      metrics_dict : {node_key: {metric_name: value}} 格式字典

    返回：
      {'mean': {metric_name: float}, 'std': {metric_name: float}}
    """
    from collections import defaultdict   # 按需导入
    vals = defaultdict(list)              # 按指标名收集各节点的值
    for node_metrics in metrics_dict.values():
        for k, v in node_metrics.items():
            vals[k].append(v)             # 追加每个节点的该指标值
    mean = {k: float(np.nanmean(v)) for k, v in vals.items()}   # 均值（忽略 NaN）
    std = {k: float(np.nanstd(v)) for k, v in vals.items()}     # 标准差（忽略 NaN）
    return {"mean": mean, "std": std}   # 返回均值和标准差字典


def _save_losses(loss_list: list, path: str) -> None:
    """
    将损失记录列表保存为 CSV 文件（列表为空时跳过，避免空文件错误）。

    参数：
      loss_list : 损失记录列表，每条记录为 {'round': int, 'loss': float, ...} 字典
      path      : CSV 文件保存路径
    """
    if loss_list:   # 仅在有数据时保存（空列表跳过，避免写入空 CSV）
        save_csv(pd.DataFrame(loss_list), path)   # 将列表转为 DataFrame 后保存
