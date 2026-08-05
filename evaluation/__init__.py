# evaluation 包的公开接口：将指标计算和可视化函数直接暴露到 evaluation 命名空间
from .metrics import (
    compute_domain_metrics,   # 计算领域预测指标：MAE / RMSE / Pearson r / R²
    compute_image_metrics,    # 计算图像质量指标：PSNR / SSIM / LPIPS（评估合成 MRI 质量）
    compute_fid_kid,          # 计算 FID / KID（基于 torch-fidelity，评估合成分布质量）
    compute_bias_analysis,    # 按年龄段/性别分组计算误差（偏差分析，用于公平性评估）
)
from .generative_eval import (
    evaluate_generative_model,   # 评估单个 MedMD 生成模型：PSNR/SSIM/LPIPS/FID/KID
    nan_generative_metrics,      # 全 NaN 生成指标占位（无数据/合成失败时）
    aggregate_metrics,           # 跨节点指标聚合（mean/std，忽略 NaN）
    save_summary,                # 保存跨节点 mean/std 指标为 CSV
    save_loss_summary,           # 保存跨客户端逐轮损失 mean/std 为 CSV
    GEN_METRIC_KEYS,             # 生成指标名列表
)
from .visualization import (
    plot_loss_curves,         # 绘制各方法训练损失曲线（含 ±1 std 阴影区域）
    plot_metric_bars,         # 绘制各方法指标对比柱状图（含误差棒）
    plot_scatter,             # 绘制真实年龄 vs 预测年龄散点图（含 y=x 参考线）
    plot_age_bin_mae,         # 绘制各年龄段的 MAE 柱状图（用于年龄相关偏差可视化）
    plot_residuals,           # 绘制预测残差图（误差 vs 真实年龄，用于系统性偏差检测）
    plot_error_heatmap,       # 绘制分组误差热图（seaborn heatmap，年龄段 × 其他因素）
    save_figure,              # 统一的图表保存函数（自动追加后缀、tight_layout、关闭图）
)

# 声明 `from evaluation import *` 时可以导入哪些名称
__all__ = [
    "compute_domain_metrics", "compute_image_metrics",
    "compute_fid_kid", "compute_bias_analysis",
    "evaluate_generative_model", "nan_generative_metrics",
    "aggregate_metrics", "save_summary", "save_loss_summary",
    "GEN_METRIC_KEYS",
    "plot_loss_curves", "plot_metric_bars",
    "plot_scatter", "plot_age_bin_mae", "plot_residuals",
    "plot_error_heatmap", "save_figure",
]
