# evaluation 包的公开接口：将指标计算和可视化函数直接暴露到 evaluation 命名空间
from .metrics import (
    compute_domain_metrics, compute_image_metrics,
    compute_fid_kid, compute_bias_analysis
)
from .visualization import (
    plot_loss_curves, plot_metric_bars,
    plot_scatter, plot_age_bin_mae, plot_residuals,
    plot_error_heatmap, save_figure
)

__all__ = [
    "compute_domain_metrics", "compute_image_metrics",
    "compute_fid_kid", "compute_bias_analysis",
    "plot_loss_curves", "plot_metric_bars",
    "plot_scatter", "plot_age_bin_mae", "plot_residuals",
    "plot_error_heatmap", "save_figure",
]
