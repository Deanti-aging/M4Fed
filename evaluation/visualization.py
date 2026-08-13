"""
可视化函数集合。

所有图表使用 Carlito 字体（不可用时退回 DejaVu Sans / Arial）
和项目预定义调色板，保存为 SVG 或 PDF。
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import os                              # 标准库：路径操作（makedirs、path.join）
import numpy as np                     # NumPy：数组计算（arange、mean 等）
import pandas as pd                    # Pandas：DataFrame（误差热图的透视表）
from typing import Dict, List, Optional, Tuple   # 类型注解工具

# 使用非交互式后端（适合服务器/无显示器环境，不弹出窗口）
import matplotlib   # Matplotlib 图形库
matplotlib.use("Agg")   # 设置非交互式后端（必须在 import pyplot 之前调用）
import matplotlib.pyplot as plt                # 绘图接口
import matplotlib.patches as mpatches          # 图例色块（Patch）
import seaborn as sns                          # Seaborn：热图绘制

# 字体候选列表：Carlito 优先（论文指定字体），后备字体依次降级
_FONT_CANDIDATES = ["Carlito", "DejaVu Sans", "Arial"]

# 论文颜色调色板（6 种颜色，对应 6 个比较方法）
PALETTE = ["#8172B3", "#4874CB", "#EE822F", "#75BD42", "#D64A4A", "#2AB7A9"]

# 各基线的显示名称（与调色板颜色一一对应，用于图例标注）
BASELINE_NAMES = ["FedAvg", "FedProx", "FAN", "GPT-FL", "M4Fed-clients", "M4Fed-server"]


def _setup_font() -> None:
    """
    设置 matplotlib 字体：按优先级尝试 _FONT_CANDIDATES 中的字体，
    若均不可用（未安装）则退回系统默认 sans-serif 字体。
    在每个绘图函数开头调用，确保字体设置正确。
    """
    from matplotlib import font_manager   # 字体管理器
    available = {f.name for f in font_manager.fontManager.ttflist}   # 获取所有已安装字体名集合
    for font in _FONT_CANDIDATES:
        if font in available:
            plt.rcParams["font.family"] = font   # 设置为第一个可用的候选字体
            return   # 找到可用字体后立即返回
    plt.rcParams["font.family"] = "sans-serif"   # 所有候选字体均不可用时退回系统默认


def save_figure(fig: plt.Figure, path: str, fmt: str = "svg", dpi: int = 300) -> None:
    """
    保存图表到文件并关闭 figure（释放内存）。

    自动创建父目录（若不存在）；若路径末尾未含扩展名则自动追加。
    dpi=300 适合论文印刷质量（300 DPI 是 IEEE/Springer 等期刊的标准要求）。

    参数：
      fig  : 待保存的 matplotlib Figure 对象
      path : 保存路径（不含扩展名时自动追加 .{fmt}）
      fmt  : 文件格式（默认 "svg"，亦支持 "pdf"、"png"）
      dpi  : 分辨率（像素/英寸，矢量格式 svg/pdf 时此参数无效）
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)   # 自动创建父目录
    if not path.endswith(f".{fmt}"):
        path = f"{path}.{fmt}"   # 自动追加文件扩展名
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")   # 保存（tight 避免标签被裁剪）
    plt.close(fig)   # 关闭 figure 释放内存（批量保存时避免内存溢出）


# --------------------------------------------------------------------------- #
#  损失曲线图                                                                    #
# --------------------------------------------------------------------------- #

def plot_loss_curves(
    losses: Dict[str, Dict[str, list]],   # {基线名: {'mean': [损失值...], 'std': [标准差...]}}
    xlabel: str = "Round",                 # x 轴标签
    ylabel: str = "Loss",                  # y 轴标签
    palette: Optional[List[str]] = None,   # 颜色列表（默认用 PALETTE）
    baseline_names: Optional[List[str]] = None,  # 显示名称列表（默认用字典键名）
    save_path: Optional[str] = None,       # 图表保存路径（None 则不保存）
    fmt: str = "svg",                      # 保存格式（"svg"、"pdf"、"png"）
) -> plt.Figure:
    """
    绘制多个方法的训练损失曲线（带置信区间）。

    每条曲线：实线表示均值，半透明区域表示 ±1 标准差范围（观察稳定性）。
    图例放在图区右上角，避免遮挡曲线主体。

    参数：
      losses : 各方法的损失字典，键为方法名，值包含 'mean' 列表和可选的 'std' 列表

    返回：
      matplotlib Figure 对象（可进一步修改）
    """
    _setup_font()   # 设置字体（确保论文字体一致性）
    palette = palette or PALETTE   # 使用自定义调色板或默认 PALETTE
    bnames = baseline_names or list(losses.keys())   # 显示名称列表
    fig, ax = plt.subplots(figsize=(8, 5))   # 创建 8×5 英寸的图表

    handles = []   # 收集图例句柄
    for i, (bname, vals) in enumerate(losses.items()):
        color = palette[i % len(palette)]          # 循环使用调色板颜色
        mean = np.array(vals["mean"])              # 均值损失曲线（numpy 数组）
        std = np.array(vals.get("std", np.zeros_like(mean)))   # 标准差（默认为 0）
        x = np.arange(1, len(mean) + 1)           # x 轴：轮次编号（1-indexed）
        ax.plot(x, mean, color=color, linewidth=2, label=bname)   # 绘制均值曲线
        # 绘制置信区间（均值 ± 标准差，半透明填充）
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        handles.append(mpatches.Patch(color=color, label=bname))   # 添加图例色块

    # 设置图例（右上角，半透明背景）
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
              framealpha=0.85, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11)   # x 轴标签
    ax.set_ylabel(ylabel, fontsize=11)   # y 轴标签
    ax.grid(True, alpha=0.3)             # 网格线（轻微透明）
    fig.tight_layout()                   # 自动调整子图间距（防止标签被截断）
    if save_path:
        save_figure(fig, save_path, fmt=fmt)   # 保存图表
    return fig   # 返回 Figure 对象（供进一步修改）


# --------------------------------------------------------------------------- #
#  指标柱状图                                                                    #
# --------------------------------------------------------------------------- #

def plot_metric_bars(
    metrics: Dict[str, Dict[str, float]],              # {基线名: {指标名: 值}}
    stds: Optional[Dict[str, Dict[str, float]]] = None,  # 误差棒（标准差，可选）
    metric_name: str = "MAE",               # 要展示的指标名称
    palette: Optional[List[str]] = None,    # 颜色列表
    save_path: Optional[str] = None,        # 保存路径
    fmt: str = "svg",                       # 保存格式
) -> plt.Figure:
    """
    绘制各方法在指定指标上的柱状对比图（带误差棒）。

    参数：
      metrics    : 各方法指标字典（{方法名: {指标名: 值}}）
      stds       : 各方法标准差字典（用于误差棒，可选）
      metric_name: 要展示的指标名称（如 'MAE', 'RMSE', 'R2'）

    返回：
      matplotlib Figure 对象
    """
    _setup_font()   # 设置字体
    palette = palette or PALETTE   # 颜色调色板
    bnames = list(metrics.keys())   # 方法名列表（作为 x 轴刻度）
    vals = [metrics[b].get(metric_name, float("nan")) for b in bnames]   # 各方法的指标值
    errs = None   # 误差棒（默认不显示）
    if stds is not None:
        errs = [stds[b].get(metric_name, 0.0) for b in bnames]   # 从 stds 中提取误差

    colors = [palette[i % len(palette)] for i in range(len(bnames))]   # 各柱的颜色
    fig, ax = plt.subplots(figsize=(7, 4))   # 创建 7×4 英寸图表
    x = np.arange(len(bnames))               # x 轴位置（整数数组）
    # 绘制柱状图（含误差棒，capsize 为端帽大小）
    ax.bar(x, vals, yerr=errs, color=colors, width=0.6, capsize=5,
           error_kw={"elinewidth": 1.5})   # 误差棒线宽
    ax.set_xticks(x)   # 设置 x 轴刻度位置
    ax.set_xticklabels(bnames, rotation=20, ha="right", fontsize=9)   # x 轴标签（旋转防重叠）
    ax.set_ylabel(metric_name, fontsize=11)   # y 轴标签（指标名）
    ax.grid(axis="y", alpha=0.3)   # 水平网格线（辅助读数）
    handles = [mpatches.Patch(color=c, label=n) for c, n in zip(colors, bnames)]   # 图例句柄
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
              framealpha=0.85, fontsize=9)
    fig.tight_layout()   # 自动调整间距
    if save_path:
        save_figure(fig, save_path, fmt=fmt)   # 保存图表
    return fig   # 返回 Figure 对象


# --------------------------------------------------------------------------- #
#  偏差分析系列图                                                                 #
# --------------------------------------------------------------------------- #

def plot_scatter(
    y_true: List[float],           # 真实年龄列表
    y_pred: List[float],           # 预测年龄列表
    save_path: Optional[str] = None,   # 保存路径
    fmt: str = "svg",                  # 保存格式
) -> plt.Figure:
    """
    绘制真实年龄 vs 预测年龄的散点图（评估预测精度的标准图）。

    包含 y=x 参考线（理想预测应沿此线分布）。
    点的透明度设为 0.5 以便在样本密集区域观察分布。

    参数：
      y_true : 真实年龄列表
      y_pred : 对应的预测年龄列表

    返回：
      matplotlib Figure 对象
    """
    _setup_font()   # 设置字体
    fig, ax = plt.subplots(figsize=(5, 5))   # 正方形图（保证 x/y 轴比例一致）
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=PALETTE[4])   # 散点图（半透明小点）
    lo, hi = min(y_true + y_pred), max(y_true + y_pred)   # x/y 轴范围（统一）
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="y=x")   # 理想对角线（y=x）
    ax.set_xlabel("True Age", fontsize=11)       # x 轴：真实年龄
    ax.set_ylabel("Predicted Age", fontsize=11)  # y 轴：预测年龄
    ax.legend(fontsize=9)    # 显示图例（y=x 参考线说明）
    ax.grid(True, alpha=0.3)   # 网格线
    fig.tight_layout()   # 调整间距
    if save_path:
        save_figure(fig, save_path, fmt=fmt)   # 保存图表
    return fig   # 返回 Figure 对象


def plot_age_bin_mae(
    bins: List[str],                    # 年龄段标签列表（如 ["20-30", "30-40", ...]）
    mae: List[float],                   # 各年龄段的 MAE 列表
    std: Optional[List[float]] = None,  # 各年龄段的 MAE 标准差（用于误差棒，可选）
    save_path: Optional[str] = None,    # 保存路径
    fmt: str = "svg",                   # 保存格式
) -> plt.Figure:
    """
    绘制按年龄分段统计的 MAE 柱状图（带标准差误差棒）。

    用于分析模型在不同年龄段的预测精度差异
    （通常高龄群体预测误差更大，因为训练数据中高龄样本较少）。

    参数：
      bins : 年龄段标签列表（来自 compute_bias_analysis 的 age_bin_mae['bins']）
      mae  : 对应的 MAE 列表
      std  : 对应的 MAE 标准差列表（可选）

    返回：
      matplotlib Figure 对象
    """
    _setup_font()   # 设置字体
    fig, ax = plt.subplots(figsize=(8, 4))   # 8×4 英寸（宽图，适合多个年龄段）
    x = np.arange(len(bins))   # x 轴位置
    ax.bar(x, mae, yerr=std, color=PALETTE[4], width=0.6, capsize=4)   # 柱状图 + 误差棒
    ax.set_xticks(x)   # 设置 x 轴刻度
    ax.set_xticklabels(bins, rotation=30, ha="right", fontsize=9)   # x 轴年龄段标签
    ax.set_ylabel("MAE (years)", fontsize=11)   # y 轴：MAE（年）
    ax.grid(axis="y", alpha=0.3)                 # 水平网格线
    fig.tight_layout()   # 调整间距
    if save_path:
        save_figure(fig, save_path, fmt=fmt)   # 保存图表
    return fig   # 返回 Figure 对象


def plot_residuals(
    ages: List[float],                # 真实年龄列表（x 轴）
    errors: List[float],              # 带符号预测误差列表（y 轴，预测 - 真实）
    save_path: Optional[str] = None,  # 保存路径
    fmt: str = "svg",                 # 保存格式
) -> plt.Figure:
    """
    绘制残差图：真实年龄（x 轴）vs 预测误差（y 轴）。

    用于诊断系统性年龄偏差：
      - 理想情况下残差随机分布在 y=0 水平线两侧（无系统性偏差）
      - 若存在年龄相关趋势（如年轻组正残差/高估，年老组负残差/低估），
        说明模型存在"向均值回归"的年龄偏差（常见于回归任务）

    参数：
      ages   : 真实年龄列表（x 轴）
      errors : 带符号误差列表（y 轴，正=高估，负=低估）

    返回：
      matplotlib Figure 对象
    """
    _setup_font()   # 设置字体
    fig, ax = plt.subplots(figsize=(6, 4))   # 6×4 英寸图表
    ax.scatter(ages, errors, alpha=0.5, s=20, color=PALETTE[5])   # 散点图
    ax.axhline(0, color="k", linewidth=1.5, linestyle="--")   # y=0 参考线（理想残差基准）
    ax.set_xlabel("True Age", fontsize=11)                           # x 轴：真实年龄
    ax.set_ylabel("Prediction Error (Pred - True)", fontsize=11)     # y 轴：预测误差
    ax.grid(True, alpha=0.3)           # 网格线
    fig.tight_layout()   # 调整间距
    if save_path:
        save_figure(fig, save_path, fmt=fmt)   # 保存图表
    return fig   # 返回 Figure 对象


def plot_error_heatmap(
    pivot: pd.DataFrame,                        # 透视表（行=年龄段，列=因子取值，值=平均绝对误差）
    xlabel: str = "Category",                   # x 轴标签（因子类别名称）
    ylabel: str = "Age bin",                    # y 轴标签
    save_path: Optional[str] = None,            # 保存路径
    fmt: str = "svg",                           # 保存格式
) -> plt.Figure:
    """
    绘制误差热图：行为年龄分段，列为因子取值（性别/健康状态），
    单元格颜色深浅表示该组合的平均绝对误差大小。

    使用 YlOrRd 色系（从黄色到红色）：误差越大，颜色越深（越红）。
    每个单元格内标注具体的 MAE 数值（2 位小数）。

    参数：
      pivot  : 透视表 DataFrame（由 _build_factor_df 生成）
      xlabel : x 轴标签（通常为因子名称，如 "Sex"）
      ylabel : y 轴标签（通常为 "Age Bin"）

    返回：
      matplotlib Figure 对象
    """
    _setup_font()   # 设置字体
    # 图表尺寸随列数（因子取值数）和行数（年龄分段数）动态调整
    fig, ax = plt.subplots(figsize=(max(5, pivot.shape[1] * 1.2),   # 宽度随列数增加
                                   max(4, pivot.shape[0] * 0.5)))    # 高度随行数增加
    # seaborn 热图：annot 显示数值，fmt=".2f" 保留 2 位小数，YlOrRd 颜色方案
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "MAE (years)"})
    ax.set_xlabel(xlabel, fontsize=11)   # x 轴标签
    ax.set_ylabel(ylabel, fontsize=11)   # y 轴标签
    fig.tight_layout()   # 调整间距
    if save_path:
        save_figure(fig, save_path, fmt=fmt)   # 保存图表
    return fig   # 返回 Figure 对象


# --------------------------------------------------------------------------- #
#  3D MRI 体积可视化：正中三视图 2D 切片                                          #
# --------------------------------------------------------------------------- #

def plot_mri_slices(
    volume: np.ndarray,                 # 3D MRI 体积，形状 (D, H, W) 或 (1, D, H, W)
    save_path: str,                     # 保存路径（不含扩展名时自动追加）
    fmt: str = "png",                   # 保存格式（默认 "png"，图像类内容不适合矢量格式）
    title: Optional[str] = None,        # 图表总标题（如模型/方法名）
) -> plt.Figure:
    """
    保存 3D MRI 体积的正中三视图（轴状/冠状/矢状）2D 切片图。

    分别沿深度/高度/宽度三个轴取正中间的一张 2D 切片，用灰度图并排展示，
    三张切片共享同一灰度范围（体积的全局 min/max），便于直观比较。

    参数：
      volume    : 3D MRI 体积数组，值域通常为 [-1, 1]（preprocess_mri 处理后）
      save_path : 保存路径
      fmt       : 图片格式（"png"、"jpg" 等，非矢量格式）
      title     : 可选的图表总标题

    返回：
      matplotlib Figure 对象
    """
    _setup_font()   # 设置字体
    vol = np.asarray(volume)
    if vol.ndim == 4:   # (1, D, H, W) → (D, H, W)：去掉通道维
        vol = vol[0]
    d, h, w = vol.shape
    slices = [vol[d // 2, :, :], vol[:, h // 2, :], vol[:, :, w // 2]]   # 正中三个切面
    names = ["Axial", "Coronal", "Sagittal"]
    vmin, vmax = float(vol.min()), float(vol.max())   # 三张切片共享灰度范围，便于比较

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, sl, name in zip(axes, slices, names):
        ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)   # 灰度显示 MRI 切片
        ax.set_title(name, fontsize=11)
        ax.axis("off")   # 隐藏坐标轴（图像类展示不需要）
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_figure(fig, save_path, fmt=fmt)
    return fig


# --------------------------------------------------------------------------- #
#  便捷函数：一次保存某方法的全部偏差分析图                                        #
# --------------------------------------------------------------------------- #

def save_all_bias_figures(
    bias_data: dict,       # compute_bias_analysis 的返回值
    out_dir: str,          # 图表保存目录
    method_name: str,      # 方法名（作为文件名前缀，如 "m4fed"）
    metadata: list,        # 原始元数据列表（此处接口预留，bias_data 已包含所需数据）
    fmt: str = "svg",      # 保存格式（"svg"、"pdf"、"png"）
) -> None:
    """
    依次生成并保存偏差分析的全部标准图表：
      - 散点图（真实 vs 预测年龄）
      - 年龄分段 MAE 柱状图
      - 残差图
      - 性别×年龄误差热图（若数据足够）
      - 健康状态×年龄误差热图（若数据足够）

    图表文件名格式：{method_name}_{图表类型}.{fmt}

    参数：
      bias_data   : compute_bias_analysis() 的返回字典
      out_dir     : 图表保存目录（不存在时自动创建）
      method_name : 方法名前缀（文件名用）
      metadata    : 元数据列表（接口预留，bias_data 中已包含必要信息）
      fmt         : 图表文件格式
    """
    os.makedirs(out_dir, exist_ok=True)   # 创建输出目录（若不存在）
    d = bias_data   # 简化变量名

    # 散点图：真实年龄 vs 预测年龄
    plot_scatter(d["scatter"]["y_true"], d["scatter"]["y_pred"],
                 save_path=os.path.join(out_dir, f"{method_name}_scatter"), fmt=fmt)

    # 年龄分段 MAE 柱状图
    plot_age_bin_mae(d["age_bin_mae"]["bins"], d["age_bin_mae"]["mae"],
                     std=d["age_bin_mae"].get("std"),     # 标准差（可选）
                     save_path=os.path.join(out_dir, f"{method_name}_age_bin_mae"), fmt=fmt)

    # 残差图：真实年龄 vs 预测误差
    plot_residuals(d["residuals"]["age"], d["residuals"]["error"],
                   save_path=os.path.join(out_dir, f"{method_name}_residuals"), fmt=fmt)

    # 性别×年龄误差热图（若 sex_heatmap 非 None）
    if d.get("sex_heatmap") is not None:
        plot_error_heatmap(d["sex_heatmap"],
                           xlabel="Sex", ylabel="Age Bin",
                           save_path=os.path.join(out_dir, f"{method_name}_sex_heatmap"), fmt=fmt)

    # 健康状态×年龄误差热图（若 health_heatmap 非 None）
    if d.get("health_heatmap") is not None:
        plot_error_heatmap(d["health_heatmap"],
                           xlabel="Health Status", ylabel="Age Bin",
                           save_path=os.path.join(out_dir, f"{method_name}_health_heatmap"), fmt=fmt)
