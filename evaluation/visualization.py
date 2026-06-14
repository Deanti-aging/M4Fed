"""
可视化函数集合。

所有图表使用 Carlito 字体（不可用时退回 DejaVu Sans / Arial）
和项目预定义调色板，保存为 SVG 或 PDF。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# 使用非交互式后端（适合服务器环境，不弹出窗口）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# 字体候选列表：Carlito 优先，后备字体依次降级
_FONT_CANDIDATES = ["Carlito", "DejaVu Sans", "Arial"]

# 论文颜色调色板（6 种颜色，对应 6 个方法）
PALETTE = ["#8172B3", "#4874CB", "#EE822F", "#75BD42", "#D64A4A", "#2AB7A9"]

# 各基线的显示名称（与调色板颜色一一对应）
BASELINE_NAMES = ["FedAvg", "FedProx", "FAN", "GPT-FL", "M4Fed-clients", "M4Fed-server"]


def _setup_font() -> None:
    """
    设置 matplotlib 字体：按优先级尝试 _FONT_CANDIDATES，
    若均不可用则退回系统默认 sans-serif。
    """
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in _FONT_CANDIDATES:
        if font in available:
            plt.rcParams["font.family"] = font
            return
    plt.rcParams["font.family"] = "sans-serif"


def save_figure(fig: plt.Figure, path: str, fmt: str = "svg", dpi: int = 300) -> None:
    """
    保存图表到文件并关闭 figure（释放内存）。

    自动创建父目录；若路径末尾未含扩展名则自动追加。
    dpi=300 适合论文印刷质量。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not path.endswith(f".{fmt}"):
        path = f"{path}.{fmt}"
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  损失曲线图                                                                    #
# --------------------------------------------------------------------------- #

def plot_loss_curves(
    losses: Dict[str, Dict[str, list]],   # {基线名: {'mean': [损失值...], 'std': [标准差...]}}
    title: str = "Training Loss",
    xlabel: str = "Round",
    ylabel: str = "Loss",
    palette: Optional[List[str]] = None,
    baseline_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    fmt: str = "svg",
) -> plt.Figure:
    """
    绘制多个方法的训练损失曲线（带置信区间）。

    每条曲线：实线表示均值，半透明区域表示 ±1 标准差范围。
    图例放在图区右上角，避免遮挡曲线。

    参数：
      losses         : 各方法的损失字典，键为方法名，值包含 'mean' 和可选的 'std'
      save_path      : 若不为 None，将图表保存到该路径
    """
    _setup_font()
    palette = palette or PALETTE
    bnames = baseline_names or list(losses.keys())
    fig, ax = plt.subplots(figsize=(8, 5))

    handles = []
    for i, (bname, vals) in enumerate(losses.items()):
        color = palette[i % len(palette)]
        mean = np.array(vals["mean"])
        std = np.array(vals.get("std", np.zeros_like(mean)))
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, color=color, linewidth=2, label=bname)
        # 置信区间（填充区域）
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        handles.append(mpatches.Patch(color=color, label=bname))

    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
              framealpha=0.85, fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, fmt=fmt)
    return fig


# --------------------------------------------------------------------------- #
#  指标柱状图                                                                    #
# --------------------------------------------------------------------------- #

def plot_metric_bars(
    metrics: Dict[str, Dict[str, float]],  # {基线名: {指标名: 值}}
    stds: Optional[Dict[str, Dict[str, float]]] = None,  # 误差棒（标准差）
    metric_name: str = "MAE",
    title: Optional[str] = None,
    palette: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    fmt: str = "svg",
) -> plt.Figure:
    """
    绘制各方法在指定指标上的柱状对比图（带误差棒）。

    参数：
      metrics    : 各方法指标字典
      stds       : 各方法标准差字典（用于误差棒，可选）
      metric_name: 要展示的指标名称（如 'MAE', 'RMSE', 'R2'）
    """
    _setup_font()
    palette = palette or PALETTE
    bnames = list(metrics.keys())
    vals = [metrics[b].get(metric_name, float("nan")) for b in bnames]
    errs = None
    if stds is not None:
        errs = [stds[b].get(metric_name, 0.0) for b in bnames]

    colors = [palette[i % len(palette)] for i in range(len(bnames))]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bnames))
    # 柱状图，capsize 为误差棒端帽大小
    ax.bar(x, vals, yerr=errs, color=colors, width=0.6, capsize=5,
           error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(bnames, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title or f"{metric_name} 对比", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    handles = [mpatches.Patch(color=c, label=n) for c, n in zip(colors, bnames)]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
              framealpha=0.85, fontsize=9)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, fmt=fmt)
    return fig


# --------------------------------------------------------------------------- #
#  偏差分析系列图                                                                 #
# --------------------------------------------------------------------------- #

def plot_scatter(
    y_true: List[float],
    y_pred: List[float],
    title: str = "True Age vs Predicted Age",
    save_path: Optional[str] = None,
    fmt: str = "svg",
) -> plt.Figure:
    """
    绘制真实年龄 vs 预测年龄的散点图。

    包含 y=x 参考线（理想预测应落在该线上）。
    点的透明度设为 0.5 以便观察密度分布。
    """
    _setup_font()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=PALETTE[4])
    lo, hi = min(y_true + y_pred), max(y_true + y_pred)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="y=x")  # 理想对角线
    ax.set_xlabel("True Age", fontsize=11)
    ax.set_ylabel("Predicted Age", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, fmt=fmt)
    return fig


def plot_age_bin_mae(
    bins: List[str],
    mae: List[float],
    std: Optional[List[float]] = None,
    title: str = "Age-bin MAE",
    save_path: Optional[str] = None,
    fmt: str = "svg",
) -> plt.Figure:
    """
    绘制按年龄分段统计的 MAE 柱状图（带标准差误差棒）。

    用于分析模型在不同年龄段的预测精度差异（高龄群体误差通常更大）。
    """
    _setup_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(bins))
    ax.bar(x, mae, yerr=std, color=PALETTE[4], width=0.6, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MAE (years)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, fmt=fmt)
    return fig


def plot_residuals(
    ages: List[float],
    errors: List[float],
    title: str = "Residual Plot",
    save_path: Optional[str] = None,
    fmt: str = "svg",
) -> plt.Figure:
    """
    绘制残差图：真实年龄（x 轴）vs 预测误差（y 轴）。

    理想情况下残差应随机分布在 y=0 水平线两侧（无系统性偏差）。
    若存在明显的年龄相关趋势（如年轻组高估、年老组低估），说明模型存在年龄偏差。
    """
    _setup_font()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ages, errors, alpha=0.5, s=20, color=PALETTE[5])
    ax.axhline(0, color="k", linewidth=1.5, linestyle="--")  # y=0 参考线
    ax.set_xlabel("True Age", fontsize=11)
    ax.set_ylabel("Prediction Error (Pred - True)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, fmt=fmt)
    return fig


def plot_error_heatmap(
    pivot: pd.DataFrame,
    xlabel: str = "Category",
    ylabel: str = "Age bin",
    title: str = "Error Heatmap",
    save_path: Optional[str] = None,
    fmt: str = "svg",
) -> plt.Figure:
    """
    绘制误差热图：行为年龄分段，列为因子取值（性别/健康状态），
    单元格颜色深浅表示该组合的平均绝对误差大小。

    使用 YlOrRd 色系（从黄色到红色，误差越大颜色越深）。
    """
    _setup_font()
    # 图表尺寸随列数（因子取值数）和行数（年龄分段数）动态调整
    fig, ax = plt.subplots(figsize=(max(5, pivot.shape[1] * 1.2),
                                   max(4, pivot.shape[0] * 0.5)))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "MAE (years)"})
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, fmt=fmt)
    return fig


# --------------------------------------------------------------------------- #
#  便捷函数：一次保存某方法的全部偏差分析图                                        #
# --------------------------------------------------------------------------- #

def save_all_bias_figures(
    bias_data: dict,
    out_dir: str,
    method_name: str,
    metadata: list,
    fmt: str = "svg",
) -> None:
    """
    依次生成并保存偏差分析的全部标准图表：
      - 散点图（真实 vs 预测年龄）
      - 年龄分段 MAE 柱状图
      - 残差图
      - 性别×年龄误差热图（若数据足够）
      - 健康状态×年龄误差热图（若数据足够）

    图表文件名以 method_name 为前缀，保存在 out_dir 目录下。
    """
    os.makedirs(out_dir, exist_ok=True)
    d = bias_data

    plot_scatter(d["scatter"]["y_true"], d["scatter"]["y_pred"],
                 title=f"True vs Predicted Age ({method_name})",
                 save_path=os.path.join(out_dir, f"{method_name}_scatter"), fmt=fmt)

    plot_age_bin_mae(d["age_bin_mae"]["bins"], d["age_bin_mae"]["mae"],
                     std=d["age_bin_mae"].get("std"),
                     title=f"Age-bin MAE ({method_name})",
                     save_path=os.path.join(out_dir, f"{method_name}_age_bin_mae"), fmt=fmt)

    plot_residuals(d["residuals"]["age"], d["residuals"]["error"],
                   title=f"Residuals ({method_name})",
                   save_path=os.path.join(out_dir, f"{method_name}_residuals"), fmt=fmt)

    if d.get("sex_heatmap") is not None:
        plot_error_heatmap(d["sex_heatmap"],
                           xlabel="Sex", ylabel="Age Bin",
                           title=f"Age×Sex Error Heatmap ({method_name})",
                           save_path=os.path.join(out_dir, f"{method_name}_sex_heatmap"), fmt=fmt)

    if d.get("health_heatmap") is not None:
        plot_error_heatmap(d["health_heatmap"],
                           xlabel="Health Status", ylabel="Age Bin",
                           title=f"Age×Health Error Heatmap ({method_name})",
                           save_path=os.path.join(out_dir, f"{method_name}_health_heatmap"), fmt=fmt)
