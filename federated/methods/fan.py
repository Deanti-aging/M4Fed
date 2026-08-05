"""
FAN 基线（Federated Augmentation Network）。

与 M4Fed 的关键区别：
  - 3 个客户端，无服务器聚合（去中心化结构）
  - 每个客户端独立拥有 MedMD 和 SwinMultiTask（均随机初始化）
  - 客户端间通过交换合成数据（而非模型权重）来共享知识
  - 使用 MONAI 数据增强而非条件合成（更简单的数据扩充策略）

三个阶段：
  1. 热身阶段：各客户端对本地私有数据做按年龄分箱的平衡过采样 + MONAI 3D 增强，
               用扩充后的数据训练生成和领域模型
  2. 联邦循环：各客户端生成合成数据，广播给其他客户端；
               每个客户端在接收的合成数据上微调生成模型
  3. 本地训练：各客户端合成数据并用于微调领域模型（SwinMultiTask）
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import copy, math, os                 # copy：深拷贝；math：ceil/floor/isnan；os：路径拼接
import numpy as np                    # NumPy：随机数（均匀分布采样噪声强度）、统计
import pandas as pd                   # Pandas：CSV 保存损失和指标
from collections import defaultdict   # 按年龄分箱时的分组容器
from typing import List               # 类型注解：客户端列表
from torch.utils.data import ConcatDataset   # PyTorch：数据集合并工具

from training.medmd_trainer import MedMDTrainer     # MedMD 生成模型训练器
from training.swin_trainer import SwinTrainer         # SwinMultiTask 预测模型训练器
from evaluation.metrics import compute_domain_metrics  # 领域预测评估指标
from evaluation.generative_eval import (                 # 生成模型评估与汇总工具
    evaluate_generative_model, aggregate_metrics, save_summary, save_loss_summary,
)
from data.openmind_dataset import OpenMindDataset, make_dataloader  # 数据集和 DataLoader 工厂
from utils.io_utils import save_csv, save_model, ensure_dir  # 文件保存工具
from utils.timing_utils import record_timing         # 计时记录工具
import time                           # 标准库：高精度时间戳


def _balanced_oversample(samples, bin_width: float, cap: int):
    """
    按年龄分箱做标签平衡过采样：把样本条目重复若干次，使各年龄段的样本数尽量拉平。

    为什么需要分箱：脑龄是连续标签，无法像分类任务那样直接按类别计数，
    因此先以 bin_width（岁）为宽度离散化（如 [0,10)、[10,20) ...），再在箱之间做平衡。

    重复策略（每个箱独立处理）：
      目标数量 = min(最大箱的样本数, 该箱样本数 × cap)
      从该箱内循环取样直到达到目标数量（前 n 个各多复制一次，保证复制次数尽量均匀）

    cap 的作用：若某箱只有 1 个样本而最大箱有 200 个，无上限的平衡会把这 1 个样本
    复制 200 次；数据增强只能提供有限的多样性，这种极端复制会让模型在该样本上过拟合。
    cap 把单样本重复倍数限制在合理范围（默认 5），代价是极稀有箱无法完全拉平。

    注意：返回列表中重复的是同一个样本字典对象的引用（浅拷贝）。这是有意为之——
    样本字典在 _load_mri 中会把下载后的本地路径写回 s["mri_path"] 作为缓存，
    共享引用可以让重复条目复用同一次下载结果。字典内容本身是只读的。

    参数：
      samples   : 样本字典列表（每个含 'age' 字段）
      bin_width : 年龄分箱宽度（岁），<=0 时跳过过采样直接返回原列表
      cap       : 单个样本的最大重复倍数（<=1 时跳过过采样）

    返回：
      过采样后的样本字典列表（长度 >= len(samples)）
    """
    if not samples or bin_width <= 0 or cap <= 1:
        return list(samples)   # 参数关闭过采样或无样本，原样返回

    # ---- 第一步：按年龄分箱，建立 {箱索引: [样本, ...]} ----
    bins = defaultdict(list)
    for s in samples:
        age = s.get("age")
        try:
            age_f = float(age)
        except (TypeError, ValueError):
            age_f = float("nan")
        # 年龄缺失的样本归入独立的 None 箱（不与任何年龄段混合，也参与平衡）
        key = None if math.isnan(age_f) else int(math.floor(age_f / bin_width))
        bins[key].append(s)

    if len(bins) <= 1:
        return list(samples)   # 只有一个箱，本来就是"平衡"的，无需过采样

    # ---- 第二步：以最大箱为目标，逐箱补齐（受 cap 限制）----
    target = max(len(v) for v in bins.values())   # 目标数量 = 最大箱的样本数
    out = []
    for key in sorted(bins.keys(), key=lambda k: (k is None, k)):   # 排序保证结果可复现
        bucket = bins[key]
        n_target = min(target, len(bucket) * cap)   # 该箱的实际目标数量（受 cap 约束）
        # 循环取样：bucket[i % len(bucket)] 使复制次数在箱内均匀分布
        out.extend(bucket[i % len(bucket)] for i in range(n_target))
    return out


def _augment_loader(loader, cfg):
    """
    为 DataLoader 的数据集创建"标签平衡过采样 + MONAI 3D 增强"的新 DataLoader。

    这是 FAN 热身阶段的核心数据处理步骤，包含两个相互配合的机制：
      1. 过采样（本函数）：按年龄分箱重复样本条目，拉平各年龄段的样本数，
         纠正 OpenMind 中年龄分布的长尾（青年受试者远多于老年受试者）
      2. 增强（OpenMindDataset.augment=True）：__getitem__ 时对 3D MRI 随机施加
         翻转/旋转/缩放/弹性形变/裁剪。这是过采样能生效的前提——
         若无增强，重复条目会返回完全相同的体积，等价于给稀有年龄段的损失加权，
         而不是真正扩充数据分布。

    若数据集为空（无样本），直接返回原始 loader（避免空数据集错误）。

    参数：
      loader : 原始 DataLoader（含 OpenMindDataset）
      cfg    : 全局配置对象

    返回：
      过采样 + 增强后的 DataLoader（长度 >= 原 loader）
    """
    # 从原始 DataLoader 的 Dataset 中提取样本列表（若 Dataset 没有 .samples 属性则用空列表）
    base_samples = loader.dataset.samples if hasattr(loader.dataset, "samples") else []
    # 按年龄分箱做平衡过采样（稀有年龄段的样本被重复多次）
    balanced = _balanced_oversample(
        base_samples,
        bin_width=cfg.federated.fan_age_bin_width,   # 年龄分箱宽度（岁）
        cap=cfg.federated.fan_oversample_cap,        # 单样本最大重复倍数
    )
    aug_ds = OpenMindDataset(
        balanced,
        mri_shape=tuple(cfg.data.mri_shape),         # MRI 目标分辨率
        augment=True,   # 启用 MONAI 3D 增强流水线（每次 __getitem__ 随机变换）
        seed=cfg.training.seed,                      # 增强随机种子（保证可复现）
        augment_crop_ratio=cfg.data.augment_crop_ratio,   # 随机裁剪比例
    )
    if len(aug_ds) == 0:
        return loader   # 数据集为空，返回原始 loader（避免空迭代器错误）
    if len(balanced) != len(base_samples):
        print(f"[FAN] 标签平衡过采样：{len(base_samples)} → {len(balanced)} 个样本条目")
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
        c.build_models(task_specs, random_init_swin=True)

    _train_t0 = time.perf_counter()   # 记录训练总开始时间

    # ------------------------------------------------------------------ #
    # 阶段 1：热身（Warm-up）—— 各客户端在增强数据上独立训练生成和领域模型     #
    # ------------------------------------------------------------------ #
    print("[FAN] === 热身阶段 ===")
    # 为每个客户端创建带 MONAI 数据增强的 DataLoader（随机翻转等 3D 增强）
    aug_loaders = [_augment_loader(c.train_loader, cfg) for c in clients]

    # 为每个客户端创建生成模型（MedMD）和领域模型（SwinMultiTask）的训练器
    client_gen_trainers = []
    client_dom_trainers = []
    for i, c in enumerate(clients):
        g = MedMDTrainer(c.medmd, device=device, lr=cfg.training.lr_medmd,
                         mixed_precision=cfg.training.mixed_precision, use_qlora=False)
        d = SwinTrainer(c.swin, device=device, lr=cfg.training.lr_swin,
                        mixed_precision=cfg.training.mixed_precision, use_qlora=False)
        g.train_epoch(aug_loaders[i])
        d.train_epoch(aug_loaders[i])
        client_gen_trainers.append(g)
        client_dom_trainers.append(d)

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
    # 阶段 3：本地训练（Local Training）—— 合成增强数据，训练领域模型 SwinMultiTask #
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
        merged_loader = make_dataloader(
            ConcatDataset([aug_base, syn_ds]),
            cfg.training.batch_size, shuffle=True, seed=cfg.training.seed,
        )
        for ep in range(cfg.federated.local_rounds):
            loss = client_dom_trainers[i].train_epoch(merged_loader)
            local_losses[f"client_{c.client_id}"].append({"round": ep, "loss": loss})

    record_timing("fan/train", time.perf_counter() - _train_t0)   # 记录训练总耗时

    # ---- 领域模型评估阶段（脑年龄预测：MAE / RMSE / Pearson r / R²） ----
    _eval_t0 = time.perf_counter()
    results = {}
    for c in clients:
        t = SwinTrainer(c.swin, device=device, lr=0, mixed_precision=False)
        y_true, y_pred = t.evaluate(c.test_loader)
        results[f"client_{c.client_id}"] = compute_domain_metrics(y_true, y_pred)
    record_timing("fan/eval", time.perf_counter() - _eval_t0)   # 记录评估耗时

    # 计算跨客户端领域指标的均值和标准差
    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _agg({k: results[k] for k in client_keys})
    results["clients_mean"] = agg["mean"]   # 均值
    results["clients_std"] = agg["std"]     # 标准差

    # ---- 生成模型评估阶段（各客户端 MedMD：PSNR / SSIM / LPIPS / FID / KID） ----
    # FAN 去中心化，无服务器节点：每个客户端各自持有并训练 MedMD，逐客户端评估
    _gen_t0 = time.perf_counter()
    gen_results = {}
    for c in clients:
        gen_results[f"client_{c.client_id}"] = evaluate_generative_model(
            c.medmd, c.test_loader, cfg, device)
    gen_agg = aggregate_metrics({k: gen_results[k] for k in client_keys})
    gen_results["clients_mean"] = gen_agg["mean"]   # 跨客户端均值
    gen_results["clients_std"] = gen_agg["std"]     # 跨客户端标准差
    results["gen"] = gen_results   # 生成指标挂到结果字典（供 run_all 汇总）
    record_timing("fan/gen_eval", time.perf_counter() - _gen_t0)

    # ---- 保存结果到磁盘 ----
    for c in clients:
        key = f"client_{c.client_id}"
        # 联邦循环阶段损失（MedMD 生成模型）与本地训练阶段损失（Swin 领域模型）分开保存
        save_csv(pd.DataFrame(all_losses[key]), os.path.join(out, f"{key}_medmd_losses.csv"))
        save_csv(pd.DataFrame(local_losses[key]), os.path.join(out, f"{key}_swin_losses.csv"))
        save_csv(pd.DataFrame([results[key]]), os.path.join(out, f"{key}_metrics.csv"))
        save_csv(pd.DataFrame([gen_results[key]]),
                 os.path.join(out, f"{key}_gen_metrics.csv"))   # 生成指标 CSV
        if cfg.eval.save_model:
            save_model(c.swin, os.path.join(out, f"{key}_swin.pt"))    # 领域模型权重
            save_model(c.medmd, os.path.join(out, f"{key}_medmd.pt"))  # 生成模型权重

    # 跨客户端汇总 CSV：领域指标、生成指标、逐轮损失的 mean/std
    save_summary(results["clients_mean"], results["clients_std"],
                 os.path.join(out, "clients_domain_summary.csv"))
    save_summary(gen_results["clients_mean"], gen_results["clients_std"],
                 os.path.join(out, "clients_gen_summary.csv"))
    save_loss_summary({k: all_losses[k] for k in client_keys},
                      os.path.join(out, "clients_medmd_loss_summary.csv"))
    save_loss_summary({k: local_losses[k] for k in client_keys},
                      os.path.join(out, "clients_swin_loss_summary.csv"))
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
