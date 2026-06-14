"""
评估指标集合：

  领域模型评估（年龄预测）: MAE、RMSE、Pearson r、R²
  生成质量评估（配对体积）: PSNR、SSIM、LPIPS
  分布级别评估           : FID、KID（基于 torch-fidelity）
  偏差分析               : 年龄分段 MAE、残差图、性别/健康状态误差热图
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch


# --------------------------------------------------------------------------- #
#  领域模型指标（年龄预测）                                                       #
# --------------------------------------------------------------------------- #

def compute_domain_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算年龄预测任务的四个核心指标。

    过滤规则：自动跳过真实值或预测值中的 NaN/Inf（无标签样本）。
    样本数 < 2 时返回全 NaN（无法计算相关系数等统计量）。

    返回：
      {
        'MAE'      : 平均绝对误差（年），越小越好
        'RMSE'     : 均方根误差（年），对大误差更敏感
        'Pearson_r': Pearson 相关系数，[-1, 1]，越接近 1 越好
        'R2'       : R² 决定系数，越接近 1 越好
      }
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # 过滤非有限值（NaN 和 Inf）
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[valid], y_pred[valid]

    if len(yt) < 2:
        # 有效样本太少，无法计算统计量
        return {"MAE": float("nan"), "RMSE": float("nan"),
                "Pearson_r": float("nan"), "R2": float("nan")}

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r, _ = pearsonr(yt, yp)
    r2 = r2_score(yt, yp)
    return {"MAE": mae, "RMSE": rmse, "Pearson_r": r, "R2": r2}


# --------------------------------------------------------------------------- #
#  图像质量指标（配对 3D 体积）                                                   #
# --------------------------------------------------------------------------- #

def compute_image_metrics(
    real_vols: List[np.ndarray],   # 真实体积列表，每个形状 (D, H, W)，值域 [0, 1]
    fake_vols: List[np.ndarray],   # 合成体积列表，形状与真实体积一一对应
) -> Dict[str, float]:
    """
    计算配对 3D 体积的 PSNR、SSIM 和 LPIPS（感知相似度）。

    LPIPS 的计算方式：
      - 取每个体积的中间轴向切片（2D），转为 RGB（三通道复制）
      - 使用 VGG 网络提取感知特征进行比较
      - 需要安装 lpips 库；未安装时返回 NaN

    值域约定：输入体积若值域为 [-1, 1]，函数内部会自动转换到 [0, 1]。

    返回：
      {'PSNR': float, 'SSIM': float, 'LPIPS': float}
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    psnrs, ssims, lpips_vals = [], [], []
    lpips_fn = _get_lpips_fn()

    for real, fake in zip(real_vols, fake_vols):
        real = _clip01(real)
        fake = _clip01(fake)

        # 整体体积的 PSNR 和 SSIM
        psnrs.append(peak_signal_noise_ratio(real, fake, data_range=1.0))
        ssims.append(structural_similarity(real, fake, data_range=1.0))

        if lpips_fn is not None:
            # 取中间轴向切片，扩展为 3 通道 (1, 3, H, W) 以满足 LPIPS 输入格式
            mid = real.shape[0] // 2
            r_slice = np.stack([real[mid]] * 3, axis=0)[None]   # (1, 3, H, W)
            f_slice = np.stack([fake[mid]] * 3, axis=0)[None]
            # LPIPS 期望输入值域 [-1, 1]
            r_t = torch.from_numpy(r_slice * 2 - 1).float()
            f_t = torch.from_numpy(f_slice * 2 - 1).float()
            with torch.no_grad():
                lp = lpips_fn(r_t, f_t).item()
            lpips_vals.append(lp)

    result = {
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
    }
    result["LPIPS"] = float(np.mean(lpips_vals)) if lpips_vals else float("nan")
    return result


def _get_lpips_fn():
    """尝试加载 LPIPS 评估函数（VGG 网络）；失败时返回 None。"""
    try:
        import lpips
        return lpips.LPIPS(net="vgg")
    except Exception:
        return None


def _clip01(arr: np.ndarray) -> np.ndarray:
    """
    将数组值域统一到 [0, 1]。
    若数组最小值 < -0.5（说明是 [-1,1] 值域），先做 (x+1)/2 变换；
    否则直接 clip 到 [0, 1]。
    """
    return np.clip((arr + 1) / 2, 0, 1) if arr.min() < -0.5 else np.clip(arr, 0, 1)


# --------------------------------------------------------------------------- #
#  FID / KID（分布级别指标）                                                     #
# --------------------------------------------------------------------------- #

def compute_fid_kid(
    real_slices: List[np.ndarray],   # 真实 2D 切片列表，(H, W) 灰度图，值域 [0, 1]
    fake_slices: List[np.ndarray],   # 合成切片列表
    device: str = "cpu",
) -> Dict[str, float]:
    """
    使用 torch-fidelity 计算 FID（Fréchet Inception Distance）和
    KID（Kernel Inception Distance）。

    实现细节：
      1. 将切片保存为临时 PNG 文件（torch-fidelity 要求图像文件作为输入）
      2. 调用 torch_fidelity.calculate_metrics 计算 FID 和 KID
      3. 临时文件在 with 块结束时自动清理

    torch-fidelity 未安装时返回全 NaN（不报错）。

    返回：
      {'FID': float, 'KID': float}
    """
    import tempfile, os
    from PIL import Image

    try:
        import torch_fidelity
    except ImportError:
        return {"FID": float("nan"), "KID": float("nan")}

    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, "real")
        fake_dir = os.path.join(tmpdir, "fake")
        os.makedirs(real_dir)
        os.makedirs(fake_dir)

        # 保存真实切片为 PNG（先转 uint8 灰度图，再转 RGB 以满足 Inception V3 要求）
        for i, s in enumerate(real_slices):
            img = Image.fromarray((_clip01(s) * 255).astype(np.uint8)).convert("RGB")
            img.save(os.path.join(real_dir, f"{i:05d}.png"))
        # 保存合成切片
        for i, s in enumerate(fake_slices):
            img = Image.fromarray((_clip01(s) * 255).astype(np.uint8)).convert("RGB")
            img.save(os.path.join(fake_dir, f"{i:05d}.png"))

        try:
            metrics = torch_fidelity.calculate_metrics(
                input1=real_dir,
                input2=fake_dir,
                cuda=(device == "cuda"),
                fid=True,
                kid=True,
                # KID 子集大小不超过实际样本数，避免采样比样本数多的错误
                kid_subset_size=min(50, len(real_slices)),
            )
            return {
                "FID": metrics.get("frechet_inception_distance", float("nan")),
                "KID": metrics.get("kernel_inception_distance_mean", float("nan")),
            }
        except Exception as e:
            print(f"[FID/KID] 计算失败: {e}")
            return {"FID": float("nan"), "KID": float("nan")}


# --------------------------------------------------------------------------- #
#  偏差分析（年龄、性别、健康状态的误差分解）                                      #
# --------------------------------------------------------------------------- #

def compute_bias_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: List[pd.Series],
    age_bin_size: int = 10,
) -> dict:
    """
    对预测误差进行多维度偏差分析，返回可直接传给可视化函数的数据结构。

    分析内容：
      1. 散点图数据：真实年龄 vs 预测年龄
      2. 年龄分段 MAE：按 age_bin_size 年一组统计每段的平均绝对误差
      3. 残差图数据：真实年龄 vs 预测误差（发现系统性偏差）
      4. 性别×年龄误差热图：各 (性别, 年龄段) 组合的平均绝对误差
      5. 健康状态×年龄误差热图

    返回：
      {
        'scatter'      : {'y_true': [...], 'y_pred': [...]},
        'age_bin_mae'  : {'bins': [...], 'mae': [...], 'std': [...]},
        'residuals'    : {'age': [...], 'error': [...]},
        'sex_heatmap'  : pd.DataFrame 或 None（数据不足时为 None）,
        'health_heatmap': pd.DataFrame 或 None,
      }
    """
    errors = y_pred - y_true          # 带符号误差（预测值 - 真实值）
    abs_errors = np.abs(errors)
    ages = y_true

    # ---- 年龄分段 MAE ----
    # 计算覆盖年龄范围的分段边界（向下取整到 age_bin_size 的整数倍）
    bins = list(range(int(ages.min()) - int(ages.min()) % age_bin_size,
                      int(ages.max()) + age_bin_size + 1, age_bin_size))
    bin_labels, bin_maes, bin_stds = [], [], []
    for lo in bins[:-1]:
        hi = lo + age_bin_size
        mask = (ages >= lo) & (ages < hi)
        if mask.sum() == 0:
            continue  # 该年龄段无样本，跳过
        bin_labels.append(f"{lo}-{hi}")
        bin_maes.append(float(np.mean(abs_errors[mask])))
        bin_stds.append(float(np.std(abs_errors[mask])))

    # ---- 因子×年龄误差热图 ----
    sex_age_df = _build_factor_df(ages, errors, metadata, "sex")
    health_age_df = _build_factor_df(ages, errors, metadata, "health_status")

    return {
        "scatter": {"y_true": y_true.tolist(), "y_pred": y_pred.tolist()},
        "age_bin_mae": {"bins": bin_labels, "mae": bin_maes, "std": bin_stds},
        "residuals": {"age": ages.tolist(), "error": errors.tolist()},
        "sex_heatmap": sex_age_df,
        "health_heatmap": health_age_df,
    }


def _build_factor_df(
    ages: np.ndarray,
    errors: np.ndarray,
    metadata: List[pd.Series],
    factor_field: str,
    bin_size: int = 10,
) -> Optional[pd.DataFrame]:
    """
    构建 (年龄分段, 因子取值) → 平均绝对误差 的透视表（pivot table）。

    用于生成性别×年龄、健康状态×年龄等误差热图。
    若元数据中该字段均缺失，则返回 None。

    参数：
      ages         : 真实年龄数组
      errors       : 带符号预测误差数组（预测 - 真实）
      metadata     : 元数据列表（与 ages/errors 等长）
      factor_field : 分析的因子字段名（如 'sex'、'health_status'）
      bin_size     : 年龄分段大小（年）

    返回：
      pd.DataFrame，行为年龄分段，列为因子取值，值为该格中样本的平均绝对误差；
      或 None（若无有效数据）
    """
    rows = []
    for age, err, meta in zip(ages, errors, metadata):
        # 兼容 pd.Series（用 .get()）和 dict（用 .get()）两种元数据格式
        if isinstance(meta, dict):
            val = meta.get(factor_field)
        elif hasattr(meta, "get"):
            val = meta.get(factor_field)
        else:
            continue
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        # 将年龄向下取整到分段起点
        age_bin = int(age // bin_size) * bin_size
        rows.append({"age_bin": age_bin, factor_field: str(val), "abs_error": abs(err)})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # 按 (年龄分段, 因子值) 分组求平均绝对误差，然后转为透视表
    pivot = df.groupby(["age_bin", factor_field])["abs_error"].mean().unstack(factor_field)
    return pivot
