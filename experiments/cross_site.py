"""
跨站点泛化实验（Cross-Site Generalisation）。

使用 M4Fed 协议训练完成的服务器端 NeuroFM 模型，
在 5 个跨站点子数据集上评估脑年龄预测性能。

跨站点数据集（cross_dataset）与训练集不重叠，
代表真实世界中模型迁移到未见过的采集中心的场景。

评估指标：MAE、RMSE、Pearson r、R²，
并生成偏差分析图（年龄段/性别分组误差）。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

# 将项目根目录加入路径（使实验脚本可以直接运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import load_config
from utils.seed_utils import set_seed
from utils.io_utils import ensure_dir, save_csv, load_model
from data.openmind_dataset import OpenMindDataset, make_dataloader
from models import NeuroFMMultiTask
from training.neurofm_trainer import NeuroFMTrainer
from evaluation.metrics import compute_domain_metrics, compute_bias_analysis
from evaluation.visualization import save_all_bias_figures, plot_metric_bars


def run_cross_site(args, cfg, cross_dataset: OpenMindDataset, results_dir: str,
                   model_path: str = None) -> dict:
    """
    在跨站点数据集上评估服务器端 NeuroFM 的泛化性能。

    参数：
      args          : 命令行参数
      cfg           : M4FedConfig 配置对象
      cross_dataset : 跨站点测试集（与训练集无重叠的子数据集）
      results_dir   : 结果保存根目录
      model_path    : 已训练 NeuroFM 权重路径；
                      若为 None 或文件不存在，则使用随机初始化权重进行基线测试

    返回：
      {'metrics': 指标字典, 'bias': 偏差分析结果}
    """
    device = cfg.training.device
    out = ensure_dir(os.path.join(results_dir, "cross_site"))
    set_seed(cfg.training.seed)

    # 构建 NeuroFM 模型（仅回归头，预测年龄）
    task_specs = {"age": {"type": "regression", "n_classes": 1}}
    model = NeuroFMMultiTask(
        repo_id=cfg.model.neurofm_repo,
        task_specs=task_specs,
        device=device,
        random_init=False,   # 默认从预训练权重初始化骨干网络
    ).to(device)

    # 若提供了已训练权重，则加载（strict=False 允许部分权重匹配）
    if model_path and os.path.exists(model_path):
        load_model(model, model_path, strict=False)
        print(f"[Cross-site] 成功加载模型权重：{model_path}")
    else:
        print("[Cross-site] 未找到模型路径，使用随机初始化权重（基线对照）。")

    # 构建跨站点测试 DataLoader（shuffle=False，保持顺序确保指标一致）
    loader = make_dataloader(cross_dataset, batch_size=cfg.training.batch_size,
                             shuffle=False, seed=cfg.training.seed)

    # 使用 NeuroFMTrainer 在测试集上推理（lr=0 表示纯推理模式）
    trainer = NeuroFMTrainer(model, device=device, lr=0, mixed_precision=False)
    y_true, y_pred = trainer.evaluate(loader)

    # 计算领域评估指标
    metrics = compute_domain_metrics(y_true, y_pred)
    print(f"[Cross-site] MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}"
          f"  r={metrics['Pearson_r']:.3f}  R²={metrics['R2']:.3f}")

    # 从数据集中提取 metadata（用于偏差分析：按年龄段/性别分组）
    metadata = [s.get("metadata", {}) for s in cross_dataset.samples]
    bias_data = compute_bias_analysis(y_true, y_pred, metadata)

    # ---- 保存结果 ----
    save_csv(pd.DataFrame([metrics]), os.path.join(out, "cross_site_metrics.csv"))
    # 生成摘要行（与 run_all_baselines 的汇总格式兼容）
    save_csv(pd.DataFrame([{"method": "M4Fed-server",
                            "MAE_mean": metrics["MAE"], "MAE_std": 0,
                            "RMSE_mean": metrics["RMSE"], "RMSE_std": 0,
                            "Pearson_r_mean": metrics["Pearson_r"], "Pearson_r_std": 0,
                            "R2_mean": metrics["R2"], "R2_std": 0}]),
             os.path.join(out, "cross_site_metrics_summary.csv"))

    # 生成偏差分析图（年龄段误差热图、残差图等）
    save_all_bias_figures(bias_data, out, method_name="cross_site",
                          metadata=metadata, fmt=cfg.eval.figure_format)
    print("[Cross-site] 结果已保存至", out)
    return {"metrics": metrics, "bias": bias_data}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m4fed.yaml")
    p.add_argument("--dummy", action="store_true", help="使用虚拟数据快速验证")
    p.add_argument("--model_path", default=None, help="NeuroFM 权重路径")
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    cfg = load_config(args.config)
    mri_shape = (16, 16, 16) if args.dummy else tuple(cfg.data.mri_shape)

    if args.dummy:
        # 虚拟模式：直接生成随机假样本
        from data.openmind_dataset import _make_dummy_samples
        cross_samples = _make_dummy_samples(10, mri_shape=mri_shape, seed=99)
    else:
        # 正式模式：从 build_node_datasets 中提取跨站点数据集
        from data.openmind_dataset import build_node_datasets
        all_ds_ids = [f"ds{i:06d}" for i in range(100)]
        nd = build_node_datasets(all_ds_ids, cfg.data.n_cross_ds,
                                 cfg.data.n_private_ds_per_node, 2,
                                 cfg.data.train_split, mri_shape, cfg.data.seed,
                                 dummy_mode=True, n_dummy_per_ds=5)
        cross_samples = nd["cross"]["all"].samples

    cross_ds = OpenMindDataset(cross_samples, mri_shape)
    # 默认在 M4Fed 结果目录中查找服务器 NeuroFM 权重
    model_path = args.model_path or os.path.join(args.results_dir, "m4fed", "server_neurofm.pt")
    run_cross_site(args, cfg, cross_ds, args.results_dir, model_path)
