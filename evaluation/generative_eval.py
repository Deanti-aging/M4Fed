"""
生成模型（MedMD）质量评估与结果汇总工具。

对比实验中，FAN / GPT-FL / M4Fed 三个方法各自持有生成模型（MedMD）。
本模块提供统一的生成模型评估流程，供这三个方法的评估阶段复用：

  评估方式（img2img 配对合成）：
    对某节点的真实测试体积 x，用该节点的 MedMD 以 x 为源图像、
    以其文本条件做 img2img 合成，得到配对的合成体积 x_hat。
    - 配对指标（PSNR / SSIM / LPIPS）：直接比较 (x, x_hat)
    - 分布指标（FID / KID）           ：取每个体积的中轴切片，比较真实/合成切片集合

  为什么用 img2img 而非纯噪声生成：
    img2img 天然产生一一配对的 (真实, 合成) 体积，使 PSNR/SSIM/LPIPS 这类
    配对指标有明确定义；同时保留真实解剖结构，评估更贴近"该生成模型能否
    忠实重建目标节点的数据分布"这一联邦生成质量问题。

本模块还包含跨节点指标聚合（mean/std）与损失/指标汇总 CSV 的保存函数，
供三个方法统一调用，避免重复实现。
"""
from __future__ import annotations   # 允许注解中使用尚未定义的类名
import numpy as np                   # NumPy：nanmean/nanstd 等统计
import torch                         # PyTorch：推理张量操作
from typing import Dict, List, Optional   # 类型注解工具

from evaluation.metrics import compute_image_metrics, compute_fid_kid   # 图像/分布质量指标


# 生成模型评估的五个指标（顺序固定，供汇总表列顺序参考）
GEN_METRIC_KEYS: List[str] = ["PSNR", "SSIM", "LPIPS", "FID", "KID"]


def nan_generative_metrics() -> Dict[str, float]:
    """返回全 NaN 的生成指标字典（无有效数据或合成失败时的占位）。"""
    return {k: float("nan") for k in GEN_METRIC_KEYS}


@torch.no_grad()   # 评估阶段不需要梯度（节省显存和计算）
def evaluate_generative_model(
    medmd,                                   # 待评估的 MedMD 实例（生成模型）
    loader,                                  # 提供真实源体积的 DataLoader（通常为节点测试集）
    cfg,                                     # 全局 M4FedConfig 配置对象
    device: str,                             # 计算设备（"cuda" 或 "cpu"）
    max_samples: Optional[int] = None,       # 参与评估的最大样本数（None 则读配置）
    noise_strength: Optional[float] = None,  # img2img 噪声强度（None 则读配置）
) -> Dict[str, float]:
    """
    评估单个 MedMD 生成模型的合成质量，返回 PSNR/SSIM/LPIPS/FID/KID。

    流程：
      1. 遍历 loader，取真实体积 x 与文本条件，用 medmd.synthesise 做 img2img 合成
      2. 收集配对的 (真实, 合成) 体积，最多 max_samples 对
      3. compute_image_metrics：配对 PSNR/SSIM/LPIPS
      4. 取每个体积的中轴切片，compute_fid_kid：分布级 FID/KID

    参数：
      medmd          : MedMD 生成模型（需已训练；随机初始化会得到很差的指标）
      loader         : 真实源体积 DataLoader（batch 需含 'mri'/'text'）
      cfg            : 全局配置（读取 ddim_steps、gen_eval_* 等）
      device         : 计算设备
      max_samples    : 评估样本上限（控制耗时）；None 时用 cfg.eval.gen_eval_max_samples
      noise_strength : img2img 噪声强度；None 时用 cfg.eval.gen_eval_noise_strength

    返回：
      {'PSNR': float, 'SSIM': float, 'LPIPS': float, 'FID': float, 'KID': float}
      无有效样本或合成失败时返回全 NaN。
    """
    if medmd is None or loader is None:
        return nan_generative_metrics()   # 缺少模型或数据，直接返回 NaN

    # 读取评估超参数（优先使用显式传入值，否则回退到配置默认）
    if max_samples is None:
        max_samples = getattr(cfg.eval, "gen_eval_max_samples", 16)
    if noise_strength is None:
        noise_strength = getattr(cfg.eval, "gen_eval_noise_strength",
                                 cfg.model.noise_strength_fed)
    ddim_steps = cfg.model.ddim_steps   # DDIM 去噪步数（与训练/联邦阶段一致）

    medmd.eval()   # 切换为推理模式（关闭 Dropout 等）
    real_vols: List[np.ndarray] = []   # 收集真实体积 (D, H, W)
    fake_vols: List[np.ndarray] = []   # 收集对应的合成体积 (D, H, W)

    for batch in loader:
        if len(real_vols) >= max_samples:
            break   # 已达样本上限，停止（控制评估耗时）
        mri = batch["mri"].to(device)   # 真实 MRI (B, 1, D, H, W)，值域 [-1, 1]
        texts = batch["text"]            # 文本条件列表（img2img 的合成条件）
        try:
            # img2img 合成：真实体积 + 文本条件 → 合成体积（保留部分解剖结构）
            syn = medmd.synthesise(
                mri, texts,
                noise_strength=noise_strength,   # 噪声强度（越大越偏离源图像）
                ddim_steps=ddim_steps,           # DDIM 去噪步数
            )
        except Exception as e:
            print(f"[GenEval] 合成失败，跳过该批次：{e}")
            continue   # 单批次合成异常时跳过（健壮性兜底），不中断整体评估
        # 拆解批次，逐样本收集配对体积（去掉通道维：(B,1,D,H,W) → (D,H,W)）
        for i in range(mri.shape[0]):
            if len(real_vols) >= max_samples:
                break   # 批次内达到上限则提前退出
            real_vols.append(mri[i, 0].detach().cpu().numpy())   # 真实体积
            fake_vols.append(syn[i, 0].detach().cpu().numpy())   # 合成体积

    if not real_vols:
        return nan_generative_metrics()   # 无有效样本（如 loader 为空），返回 NaN

    # ---- 配对图像质量指标（PSNR / SSIM / LPIPS） ----
    img_metrics = compute_image_metrics(real_vols, fake_vols)

    # ---- 分布级指标（FID / KID）：取每个体积的中轴切片（深度方向中间层） ----
    real_slices = [v[v.shape[0] // 2] for v in real_vols]   # 真实中轴切片 (H, W)
    fake_slices = [f[f.shape[0] // 2] for f in fake_vols]   # 合成中轴切片 (H, W)
    dist_metrics = compute_fid_kid(real_slices, fake_slices, device=device)

    # 合并五个指标为统一字典（缺失键用 NaN 兜底）
    return {
        "PSNR": img_metrics.get("PSNR", float("nan")),
        "SSIM": img_metrics.get("SSIM", float("nan")),
        "LPIPS": img_metrics.get("LPIPS", float("nan")),
        "FID": dist_metrics.get("FID", float("nan")),
        "KID": dist_metrics.get("KID", float("nan")),
    }


# --------------------------------------------------------------------------- #
#  跨节点指标聚合与汇总 CSV 保存（供各方法评估阶段复用）                            #
# --------------------------------------------------------------------------- #

def aggregate_metrics(metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    对多个节点的指标字典计算逐指标的均值和标准差（忽略 NaN）。

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
    return {
        "mean": {k: float(np.nanmean(v)) for k, v in vals.items()},   # 均值（忽略 NaN）
        "std":  {k: float(np.nanstd(v)) for k, v in vals.items()},    # 标准差（忽略 NaN）
    }


def save_summary(mean: Dict[str, float], std: Dict[str, float], path: str) -> None:
    """
    将跨节点的 mean/std 指标字典保存为两行 CSV（stat 列区分 mean/std）。

    参数：
      mean : {metric_name: 均值}
      std  : {metric_name: 标准差}
      path : CSV 保存路径
    """
    import pandas as pd                       # 局部导入（保存时才需要）
    from utils.io_utils import save_csv       # 统一的 CSV 保存工具
    rows = [
        {"stat": "mean", **mean},   # 第一行：各指标均值
        {"stat": "std", **std},     # 第二行：各指标标准差
    ]
    save_csv(pd.DataFrame(rows), path)


def save_loss_summary(losses_by_client: Dict[str, list], path: str) -> None:
    """
    计算跨客户端的逐轮损失均值和标准差，保存为 CSV。

    输入的每个客户端损失记录为 [{'loss': float, ...}, ...]（等长，按轮次对齐）。
    输出列：[round, loss_mean, loss_std]。若客户端记录长度不一，按位置对齐，
    每个位置只对存在该轮记录的客户端求统计。

    参数：
      losses_by_client : {client_key: [损失记录字典, ...]}
      path             : CSV 保存路径
    """
    import pandas as pd                       # 局部导入
    from utils.io_utils import save_csv       # CSV 保存工具

    # 提取每个客户端的损失序列（跳过空记录的客户端）
    series = []
    for records in losses_by_client.values():
        if records:
            series.append([r.get("loss", float("nan")) for r in records])
    if not series:
        return   # 无任何损失记录，跳过（避免写空文件）

    max_len = max(len(s) for s in series)   # 最长损失序列长度（对齐轮次）
    rows = []
    for i in range(max_len):
        # 第 i 轮：收集所有存在该轮记录的客户端损失
        vals = [s[i] for s in series if i < len(s)]
        rows.append({
            "round": i,
            "loss_mean": float(np.nanmean(vals)),   # 跨客户端均值
            "loss_std": float(np.nanstd(vals)),     # 跨客户端标准差
        })
    save_csv(pd.DataFrame(rows), path)
