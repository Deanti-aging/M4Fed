"""
跨站点泛化实验（Cross-Site Generalisation）。

使用 M4Fed 协议训练完成的服务器端 SwinMultiTask 模型，
在 5 个跨站点子数据集上评估脑年龄预测性能。

跨站点数据集（cross_dataset）与训练集不重叠，
代表真实世界中模型迁移到未见过的采集中心的场景。

评估指标：MAE、RMSE、Pearson r、R²，
并生成偏差分析图（年龄段/性别分组误差）。
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import os, sys                        # os：路径操作；sys：路径修改
import numpy as np                    # NumPy（统计计算，此处用于结果对比）
import pandas as pd                   # Pandas（保存结果汇总表格）

# 将项目根目录加入 Python 路径（使实验脚本可以直接运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import load_config                           # 加载 YAML 配置文件
from utils.seed_utils import set_seed                     # 随机种子设置（保证可复现）
from utils.io_utils import ensure_dir, save_csv, load_model  # 目录创建、CSV 保存、模型加载
from data.openmind_dataset import OpenMindDataset, make_dataloader  # 数据集和 DataLoader
from models import SwinMultiTask                           # SwinUNETR 多任务模型（骨干 + 任务头）
from training.swin_trainer import SwinTrainer              # SwinMultiTask 训练/推理封装器
from evaluation.metrics import compute_domain_metrics, compute_bias_analysis
# compute_domain_metrics : 计算 MAE/RMSE/Pearson r/R²
# compute_bias_analysis  : 按年龄段/性别分组计算误差（偏差分析）
from evaluation.visualization import save_all_bias_figures, plot_metric_bars
# save_all_bias_figures  : 生成并保存全套偏差分析图表
# plot_metric_bars       : 生成指标柱状图
from utils.timing_utils import time_block                 # 代码块耗时计时器（上下文管理器）


def run_cross_site(args, cfg, cross_dataset: OpenMindDataset, results_dir: str,
                   model_path: str = None) -> dict:
    """
    在跨站点数据集上评估服务器端 SwinMultiTask 的泛化性能。

    评估流程：
      1. 构建 SwinMultiTask（仅年龄回归头）
      2. 加载已训练的领域模型权重（若路径有效）
      3. 构建跨站点测试 DataLoader（shuffle=False 保持样本顺序）
      4. 使用 SwinTrainer.evaluate 进行前向推理（无梯度更新）
      5. 计算 MAE/RMSE/Pearson r/R² 指标
      6. 进行偏差分析并保存结果（CSV + 图表）

    参数：
      args          : 命令行参数对象
      cfg           : M4FedConfig 配置对象（含模型和训练参数）
      cross_dataset : 跨站点测试集（OpenMindDataset，与训练集无重叠的子数据集）
      results_dir   : 结果保存根目录（cross_site/ 子目录将在此创建）
      model_path    : 已训练 SwinMultiTask 权重路径（.pt 文件）；
                      若为 None 或文件不存在，则使用随机初始化权重进行基线对照

    返回：
      {'metrics': 指标字典, 'bias': 偏差分析结果字典}
    """
    device = cfg.training.device   # 计算设备（"cuda" 或 "cpu"）
    out = ensure_dir(os.path.join(results_dir, "cross_site"))  # 创建 cross_site 结果子目录
    set_seed(cfg.training.seed)    # 重置随机种子（保证可复现）

    # ------------------------------------------------------------------
    # 步骤 1：构建 SwinMultiTask（仅包含年龄回归头）
    # random_init=False：骨干从 MONAI SwinViT 预训练权重初始化（后续会被 load_model 覆盖）
    # ------------------------------------------------------------------
    task_specs = {
        "age":           {"type": "regression",     "n_classes": 1},
        "sex":           {"type": "classification", "n_classes": 2},
        "health_status": {"type": "classification", "n_classes": 2},
        "race":          {"type": "classification", "n_classes": 5},
    }
    model = SwinMultiTask(
        pretrain_url=cfg.model.swinunetr_pretrain_url,   # MONAI SwinViT 预训练权重 URL
        img_size=tuple(cfg.data.mri_shape),              # 输入 MRI 空间尺寸
        feature_size=cfg.model.swinunetr_feature_size,   # 基础特征维度
        task_specs=task_specs,                           # 任务规格：年龄回归 + 健康状态/种族分类
        device=device,                                   # 计算设备
        random_init=False,                               # 使用预训练骨干（下载 MONAI 权重）
    ).to(device)                                         # 将整个模型移到计算设备

    # ------------------------------------------------------------------
    # 步骤 2：加载 M4Fed 训练完成的领域模型权重（覆盖预训练骨干权重）
    # strict=False：允许部分权重匹配（若任务头结构有细微差异时不报错）
    # ------------------------------------------------------------------
    if model_path and os.path.exists(model_path):
        load_model(model, model_path, strict=False)   # 加载已训练权重
        print(f"[Cross-site] 成功加载模型权重：{model_path}")
    else:
        # 路径无效时：使用预训练骨干权重作为基线（对照组，无 M4Fed 微调）
        print("[Cross-site] 未找到模型路径，使用预训练骨干权重（基线对照）。")

    # ------------------------------------------------------------------
    # 步骤 3：构建跨站点测试 DataLoader
    # shuffle=False：保持样本顺序，使评估结果与样本索引对应（偏差分析时需要）
    # ------------------------------------------------------------------
    loader = make_dataloader(
        cross_dataset,                        # 跨站点测试集
        batch_size=cfg.training.batch_size,   # 批大小（通常为 1，MRI 体积较大）
        shuffle=False,                        # 不打乱（保证评估顺序一致性）
        seed=cfg.training.seed,               # DataLoader Generator 随机种子
    )

    trainer = SwinTrainer(
        model,
        device=device,
        lr=0,
        mixed_precision=False,
    )
    with time_block("cross_site/eval"):   # 计时：跨站点评估阶段
        y_true, y_pred = trainer.evaluate(loader)   # 前向推理，返回 numpy 真值和预测值数组

    # ------------------------------------------------------------------
    # 步骤 5：计算评估指标（MAE/RMSE/Pearson r/R²）
    # compute_domain_metrics 内部自动过滤 NaN 值
    # ------------------------------------------------------------------
    metrics = compute_domain_metrics(y_true, y_pred)   # 返回指标字典
    print(f"[Cross-site] MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}"
          f"  r={metrics['Pearson_r']:.3f}  R²={metrics['R2']:.3f}")

    # ------------------------------------------------------------------
    # 步骤 6：偏差分析（按年龄段 / 性别分组查看误差分布）
    # metadata 列表与 y_true/y_pred 一一对应（通过 shuffle=False 保证）
    # ------------------------------------------------------------------
    metadata = [s.get("metadata", {}) for s in cross_dataset.samples]  # 提取各样本元数据
    bias_data = compute_bias_analysis(y_true, y_pred, metadata)         # 计算分组误差

    # ------------------------------------------------------------------
    # 步骤 7：保存结果（CSV 文件 + 偏差分析图表）
    # ------------------------------------------------------------------
    # 保存单行指标结果（用于事后比较不同实验配置）
    save_csv(pd.DataFrame([metrics]),
             os.path.join(out, "cross_site_metrics.csv"))

    # 保存汇总格式的 CSV（与 run_all_baselines 的汇总格式兼容，方便合并表格）
    save_csv(pd.DataFrame([{
        "method": "M4Fed-server",                      # 方法名（M4Fed 服务器端模型）
        "MAE_mean": metrics["MAE"], "MAE_std": 0,     # 单次运行无标准差
        "RMSE_mean": metrics["RMSE"], "RMSE_std": 0,
        "Pearson_r_mean": metrics["Pearson_r"], "Pearson_r_std": 0,
        "R2_mean": metrics["R2"], "R2_std": 0,
    }]), os.path.join(out, "cross_site_metrics_summary.csv"))

    # 生成并保存全套偏差分析图（年龄段误差热图 / 残差图 / 散点图等）
    save_all_bias_figures(
        bias_data,                         # 偏差分析结果（年龄段/性别分组误差）
        out,                               # 图表保存目录
        method_name="cross_site",          # 图标题前缀（用于标注方法名）
        metadata=metadata,                 # 样本元数据（含性别、年龄段等分组字段）
        fmt=cfg.eval.figure_format,        # 图表文件格式（如 "svg" 或 "pdf"）
    )
    print("[Cross-site] 结果已保存至", out)
    return {"metrics": metrics, "bias": bias_data}   # 返回指标和偏差分析结果字典


if __name__ == "__main__":
    import argparse   # 命令行参数解析
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m4fed.yaml")   # 配置文件路径
    p.add_argument("--model_path", default=None, help="SwinMultiTask 权重路径")     # 模型路径
    p.add_argument("--results_dir", default="results")          # 结果保存目录
    args = p.parse_args()   # 解析参数

    cfg = load_config(args.config)   # 加载 M4FedConfig 配置
    mri_shape = tuple(cfg.data.mri_shape)

    from data.openmind_dataset import (build_node_datasets,
                                       list_openmind_ds_ids,
                                       make_openmind_sample_loader,
                                       compute_ds_quality_stats)
    # 枚举 HuggingFace 上所有子数据集 ID（如 ds000001、ds000002 ...）
    all_ds_ids = list_openmind_ds_ids(cfg.data.hf_dataset_id)
    # 各 ds_id 的 age/sex/health_status 缺失率统计，用于挑选 cross 数据集时二次把关
    # （缺失率过高的 ds_id 会被跳过），避免偏差分析热图因样本量过少而不可靠；
    # 仅在能读取本地 local_metadata_csv 时可用，否则跳过质量过滤（None）
    quality_stats = None
    if cfg.data.local_metadata_csv and os.path.exists(cfg.data.local_metadata_csv):
        quality_stats = compute_ds_quality_stats(cfg.data.local_metadata_csv)
    # 创建样本加载函数（从 HuggingFace 惰性下载 NIfTI + CSV 元数据）
    sample_loader_fn = make_openmind_sample_loader(
        cfg.data.hf_dataset_id,                  # HuggingFace 仓库 ID
        cache_dir=cfg.data.cache_dir or None,    # 本地缓存目录
        max_cache_gb=cfg.data.max_cache_gb,      # LRU 磁盘缓存上限（GB）
    )
    # 构建完整节点数据集（包含 cross 子集）
    nd, split_ids = build_node_datasets(
        all_ds_ids,                              # 所有子数据集 ID
        cfg.data.n_cross_ds,                     # 划分给跨站点的数据集数量
        cfg.data.n_private_ds_per_node,          # 每节点私有数据集数量
        2,                                       # 客户端数量
        cfg.data.train_split,                    # 训练/测试比例
        mri_shape,                               # MRI 目标分辨率
        cfg.data.seed,                           # 随机种子
        sample_loader_fn=sample_loader_fn,       # 真实数据加载函数
        quality_stats=quality_stats,             # ds_id → {字段: 缺失率}，None 表示不过滤
        max_missing_ratio=cfg.data.cross_max_missing_ratio,
        return_split_ids=True,
    )
    cross_samples = nd["cross"]["all"].samples   # 取出跨站点评估集的样本列表

    # 报告 cross 实际使用了哪些 ds_id，便于核对偏差分析的数据来源
    from utils.io_utils import save_csv, ensure_dir
    cross_report_dir = ensure_dir(os.path.join(args.results_dir, "cross_site"))
    save_csv({"ds_id": split_ids["cross"]}, os.path.join(cross_report_dir, "cross_ds_ids.csv"))
    print(f"[Cross-site] cross 数据集（{len(split_ids['cross'])} 个 ds_id）已记录至 "
          f"{os.path.join(cross_report_dir, 'cross_ds_ids.csv')}")

    # 构建跨站点 OpenMindDataset（不做数据增强）
    cross_ds = OpenMindDataset(cross_samples, mri_shape)
    model_path = args.model_path or os.path.join(args.results_dir, "m4fed", "server_swin.pt")
    run_cross_site(args, cfg, cross_ds, args.results_dir, model_path)
