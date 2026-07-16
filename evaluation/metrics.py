"""
评估指标集合：

  领域模型评估（年龄预测）: MAE、RMSE、Pearson r、R²
  生成质量评估（配对体积）: PSNR、SSIM、LPIPS
  分布级别评估           : FID、KID（基于 torch-fidelity）
  偏差分析               : 年龄分段 MAE、残差图、性别/健康状态误差热图
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import numpy as np                    # NumPy：数值计算（数组操作、NaN 过滤、统计）
import pandas as pd                   # Pandas：DataFrame（偏差分析透视表）
from typing import Dict, List, Optional, Tuple   # 类型注解工具
import torch                          # PyTorch（LPIPS 计算时需要 Tensor 操作）


# --------------------------------------------------------------------------- #
#  领域模型指标（年龄预测）                                                       #
# --------------------------------------------------------------------------- #

def compute_domain_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算年龄预测任务的四个核心指标。

    过滤规则：自动跳过真实值或预测值中的 NaN/Inf（主要防模型输出的非有限值；
             age 无标签样本已在建数据集阶段过滤，此处为通用健壮性兜底）。
    样本数 < 2 时返回全 NaN（无法计算相关系数等统计量）。

    参数：
      y_true : 真实年龄数组
      y_pred : 预测年龄数组

    返回：
      {
        'MAE'      : 平均绝对误差（年），越小越好
        'RMSE'     : 均方根误差（年），对大误差更敏感
        'Pearson_r': Pearson 相关系数，[-1, 1]，越接近 1 越好
        'R2'       : R² 决定系数，越接近 1 越好
      }
    """
    from scipy.stats import pearsonr   # SciPy：Pearson 相关系数
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # sklearn 指标

    y_true = np.asarray(y_true, dtype=float)   # 转为 float 数组（兼容列表输入）
    y_pred = np.asarray(y_pred, dtype=float)   # 转为 float 数组

    # 过滤非有限值（NaN 和 Inf）：获得有效样本的布尔掩膜
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[valid], y_pred[valid]   # 取有效样本

    if len(yt) < 2:
        # 有效样本少于 2 个，无法计算相关系数等统计量，返回全 NaN
        return {"MAE": float("nan"), "RMSE": float("nan"),
                "Pearson_r": float("nan"), "R2": float("nan")}

    mae = mean_absolute_error(yt, yp)           # 平均绝对误差（MAE）
    rmse = np.sqrt(mean_squared_error(yt, yp))  # 均方根误差（RMSE = √MSE）
    r, _ = pearsonr(yt, yp)                      # Pearson 相关系数（r）和 p 值（忽略）
    r2 = r2_score(yt, yp)                        # R² 决定系数（解释方差比例）
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
      - 使用 VGG 网络提取感知特征进行比较（越小越相似）
      - 需要安装 lpips 库；未安装时返回 NaN

    值域约定：输入体积若值域为 [-1, 1]，函数内部会自动转换到 [0, 1]。

    参数：
      real_vols : 真实 3D 体积列表（每个形状 (D, H, W)）
      fake_vols : 对应的合成 3D 体积列表（形状相同）

    返回：
      {'PSNR': float, 'SSIM': float, 'LPIPS': float}
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # 图像质量指标

    psnrs, ssims, lpips_vals = [], [], []   # 收集各对体积的指标值
    lpips_fn = _get_lpips_fn()   # 尝试加载 LPIPS 函数（可能为 None）

    for real, fake in zip(real_vols, fake_vols):
        real = _clip01(real)   # 统一值域到 [0, 1]（[-1,1] 自动转换）
        fake = _clip01(fake)   # 统一值域到 [0, 1]

        # 计算整体 3D 体积的 PSNR（峰值信噪比，越高越相似）
        psnrs.append(peak_signal_noise_ratio(real, fake, data_range=1.0))
        # 计算整体 3D 体积的 SSIM（结构相似度，越接近 1 越好）
        ssims.append(structural_similarity(real, fake, data_range=1.0))

        if lpips_fn is not None:
            # 取中间轴向切片（避免处理整个 3D 体积）
            mid = real.shape[0] // 2   # 深度方向的中间层索引
            # 将灰度切片复制为 3 通道 (1, 3, H, W) 以满足 LPIPS 的 RGB 输入格式
            r_slice = np.stack([real[mid]] * 3, axis=0)[None]   # (1, 3, H, W)
            f_slice = np.stack([fake[mid]] * 3, axis=0)[None]   # (1, 3, H, W)
            # LPIPS 期望输入值域 [-1, 1]（将 [0,1] 线性映射到 [-1,1]）
            r_t = torch.from_numpy(r_slice * 2 - 1).float()   # 转为 float 张量
            f_t = torch.from_numpy(f_slice * 2 - 1).float()
            with torch.no_grad():
                lp = lpips_fn(r_t, f_t).item()   # 计算感知距离，.item() 转标量
            lpips_vals.append(lp)   # 记录 LPIPS 值

    result = {
        "PSNR": float(np.mean(psnrs)),   # 平均 PSNR
        "SSIM": float(np.mean(ssims)),   # 平均 SSIM
    }
    # LPIPS：若有值则取均值；否则返回 NaN（lpips 未安装）
    result["LPIPS"] = float(np.mean(lpips_vals)) if lpips_vals else float("nan")
    return result


def _get_lpips_fn():
    """
    尝试加载 LPIPS 感知相似度评估函数（基于 VGG 网络）。
    失败时返回 None（lpips 库未安装时不报错，仅返回 NaN 指标）。
    """
    try:
        import lpips   # LPIPS 感知相似度库（pip install lpips）
        return lpips.LPIPS(net="vgg")   # 使用 VGG 网络作为特征提取器
    except Exception:
        return None   # 加载失败时返回 None，调用方检查后跳过 LPIPS 计算


def _clip01(arr: np.ndarray) -> np.ndarray:
    """
    将数组值域统一到 [0, 1]。

    判断逻辑：
      - 若最小值 < -0.5（说明是 [-1, 1] 值域），做 (x+1)/2 线性映射后 clip
      - 否则（已是 [0, 1] 附近）直接 clip 到 [0, 1]

    参数：
      arr : 输入 NumPy 数组（任意形状）

    返回：
      值域在 [0, 1] 内的数组
    """
    return np.clip((arr + 1) / 2, 0, 1) if arr.min() < -0.5 else np.clip(arr, 0, 1)


# --------------------------------------------------------------------------- #
#  FID / KID（分布级别指标）                                                     #
# --------------------------------------------------------------------------- #

def compute_fid_kid(
    real_slices: List[np.ndarray],   # 真实 2D 切片列表，(H, W) 灰度图，值域 [0, 1]
    fake_slices: List[np.ndarray],   # 合成 2D 切片列表（与 real_slices 数量相同）
    device: str = "cpu",             # 计算设备（"cuda" 时 Inception 推理更快）
) -> Dict[str, float]:
    """
    使用 torch-fidelity 计算 FID（Fréchet Inception Distance）和
    KID（Kernel Inception Distance）。

    实现细节：
      1. 将切片保存为临时 PNG 文件（torch-fidelity 要求图像文件作为输入）
      2. 调用 torch_fidelity.calculate_metrics 计算 FID 和 KID
      3. 临时文件在 with 块结束时自动清理（TemporaryDirectory 上下文管理器）

    torch-fidelity 未安装时返回全 NaN（不抛出异常）。

    参数：
      real_slices : 真实 2D 切片列表（灰度图，值域 [0, 1]）
      fake_slices : 合成 2D 切片列表
      device      : 计算设备（"cuda" 时更快）

    返回：
      {'FID': float, 'KID': float}
    """
    import tempfile, os   # 标准库：临时目录和路径操作
    from PIL import Image  # 图像保存（PNG 格式）

    try:
        import torch_fidelity   # FID/KID 计算库（pip install torch-fidelity）
    except ImportError:
        return {"FID": float("nan"), "KID": float("nan")}   # 未安装时返回 NaN

    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, "real")   # 真实图像临时目录
        fake_dir = os.path.join(tmpdir, "fake")   # 合成图像临时目录
        os.makedirs(real_dir)   # 创建真实图像目录
        os.makedirs(fake_dir)   # 创建合成图像目录

        # 保存真实切片为 PNG（uint8 灰度图 → RGB，满足 Inception V3 输入要求）
        for i, s in enumerate(real_slices):
            # _clip01 确保值域 [0,1]，*255 转 [0,255]，astype uint8，convert RGB 三通道
            img = Image.fromarray((_clip01(s) * 255).astype(np.uint8)).convert("RGB")
            img.save(os.path.join(real_dir, f"{i:05d}.png"))   # 5 位数字命名保证文件顺序
        # 保存合成切片为 PNG（同样处理）
        for i, s in enumerate(fake_slices):
            img = Image.fromarray((_clip01(s) * 255).astype(np.uint8)).convert("RGB")
            img.save(os.path.join(fake_dir, f"{i:05d}.png"))

        try:
            # 调用 torch-fidelity 计算 FID 和 KID
            metrics = torch_fidelity.calculate_metrics(
                input1=real_dir,   # 真实图像目录
                input2=fake_dir,   # 合成图像目录
                cuda=(device == "cuda"),  # 是否用 GPU 加速 Inception 推理
                fid=True,          # 计算 FID
                kid=True,          # 计算 KID
                # KID 子集大小不超过实际样本数（避免采样数多于样本总数的错误）
                kid_subset_size=min(50, len(real_slices)),
            )
            return {
                "FID": metrics.get("frechet_inception_distance", float("nan")),   # FID 值
                "KID": metrics.get("kernel_inception_distance_mean", float("nan")),  # KID 均值
            }
        except Exception as e:
            print(f"[FID/KID] 计算失败: {e}")   # 打印失败原因（如样本量太少）
            return {"FID": float("nan"), "KID": float("nan")}   # 失败时返回 NaN


# --------------------------------------------------------------------------- #
#  偏差分析（年龄、性别、健康状态的误差分解）                                      #
# --------------------------------------------------------------------------- #

def compute_bias_analysis(
    y_true: np.ndarray,         # 真实年龄数组
    y_pred: np.ndarray,         # 预测年龄数组
    metadata: List[pd.Series],  # 元数据列表（与 y_true/y_pred 等长）
    age_bin_size: int = 10,     # 年龄分段大小（年），默认 10 年一组
) -> dict:
    """
    对预测误差进行多维度偏差分析，返回可直接传给可视化函数的数据结构。

    分析内容：
      1. 散点图数据：真实年龄 vs 预测年龄（观察预测精度分布）
      2. 年龄分段 MAE：按 age_bin_size 年一组统计每段的平均绝对误差
      3. 残差图数据：真实年龄 vs 预测误差（发现系统性年龄偏差）
      4. 性别×年龄误差热图：各 (性别, 年龄段) 组合的平均绝对误差
      5. 健康状态×年龄误差热图

    参数：
      y_true      : 真实年龄数组
      y_pred      : 预测年龄数组
      metadata    : 元数据列表（每个元素为 dict 或 pd.Series）
      age_bin_size: 年龄分段大小（年）

    返回：
      {
        'scatter'       : {'y_true': [...], 'y_pred': [...]},
        'age_bin_mae'   : {'bins': [...], 'mae': [...], 'std': [...]},
        'residuals'     : {'age': [...], 'error': [...]},
        'sex_heatmap'   : pd.DataFrame 或 None（数据不足时为 None）,
        'health_heatmap': pd.DataFrame 或 None,
      }
    """
    errors = y_pred - y_true          # 带符号误差：正值=高估，负值=低估
    abs_errors = np.abs(errors)       # 绝对误差（用于 MAE 计算）
    ages = y_true                     # 真实年龄（年龄轴）

    # ---- 年龄分段 MAE 计算 ----
    # 生成覆盖年龄范围的分段边界（向下取整到 age_bin_size 的整数倍）
    bins = list(range(int(ages.min()) - int(ages.min()) % age_bin_size,
                      int(ages.max()) + age_bin_size + 1, age_bin_size))
    bin_labels, bin_maes, bin_stds = [], [], []   # 收集各段的标签、MAE、标准差
    for lo in bins[:-1]:
        hi = lo + age_bin_size   # 该段的上界
        mask = (ages >= lo) & (ages < hi)   # 属于该年龄段的样本掩膜
        if mask.sum() == 0:
            continue   # 该年龄段无样本，跳过（避免空统计）
        bin_labels.append(f"{lo}-{hi}")                      # 年龄段标签（如 "20-30"）
        bin_maes.append(float(np.mean(abs_errors[mask])))    # 该段的平均绝对误差
        bin_stds.append(float(np.std(abs_errors[mask])))     # 该段的标准差

    # ---- 因子×年龄误差热图数据 ----
    sex_age_df = _build_factor_df(ages, errors, metadata, "sex")            # 性别×年龄透视表
    health_age_df = _build_factor_df(ages, errors, metadata, "health_status")  # 健康状态×年龄

    return {
        "scatter": {"y_true": y_true.tolist(), "y_pred": y_pred.tolist()},   # 散点图数据
        "age_bin_mae": {"bins": bin_labels, "mae": bin_maes, "std": bin_stds},  # 年龄段 MAE
        "residuals": {"age": ages.tolist(), "error": errors.tolist()},        # 残差图数据
        "sex_heatmap": sex_age_df,      # 性别×年龄误差热图（可能为 None）
        "health_heatmap": health_age_df,  # 健康状态×年龄误差热图（可能为 None）
    }


def _build_factor_df(
    ages: np.ndarray,             # 真实年龄数组
    errors: np.ndarray,           # 带符号预测误差数组
    metadata: List[pd.Series],    # 元数据列表（与 ages/errors 等长）
    factor_field: str,            # 分析的因子字段名（如 'sex'、'health_status'）
    bin_size: int = 10,           # 年龄分段大小（年）
) -> Optional[pd.DataFrame]:
    """
    构建 (年龄分段, 因子取值) → 平均绝对误差 的透视表（pivot table）。

    用于生成性别×年龄、健康状态×年龄等误差热图。
    若元数据中该字段均缺失（全部为 None/NaN），则返回 None。

    参数：
      ages         : 真实年龄数组
      errors       : 带符号预测误差数组（预测 - 真实）
      metadata     : 元数据列表（每个元素为 dict 或 pd.Series）
      factor_field : 要分析的因子字段名
      bin_size     : 年龄分段大小（年）

    返回：
      pd.DataFrame（行为年龄分段，列为因子取值，值为平均绝对误差）；
      或 None（若无有效数据）
    """
    rows = []   # 收集有效的 (age_bin, factor_value, abs_error) 记录
    for age, err, meta in zip(ages, errors, metadata):
        # 兼容 dict 和 pd.Series 两种元数据格式（都支持 .get() 方法）
        if isinstance(meta, dict):
            val = meta.get(factor_field)        # dict：直接 get
        elif hasattr(meta, "get"):
            val = meta.get(factor_field)        # pd.Series 也支持 get
        else:
            continue   # 不支持的类型，跳过
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue   # 字段值缺失，跳过（不参与热图统计）
        # 将年龄向下取整到分段起点（如 25 → 20，属于 20-30 段）
        age_bin = int(age // bin_size) * bin_size
        rows.append({"age_bin": age_bin, factor_field: str(val), "abs_error": abs(err)})

    if not rows:
        return None   # 无有效数据（该字段在所有元数据中均缺失），返回 None

    df = pd.DataFrame(rows)   # 将记录列表转为 DataFrame
    # 按 (年龄分段, 因子值) 分组求平均绝对误差，再转为透视表（行=年龄段，列=因子取值）
    pivot = df.groupby(["age_bin", factor_field])["abs_error"].mean().unstack(factor_field)
    return pivot   # 返回透视表（pd.DataFrame）
