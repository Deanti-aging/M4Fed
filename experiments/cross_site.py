"""
跨站点泛化实验（Cross-Site Generalisation）。

使用 M4Fed 协议训练完成的服务器端 NeuroFM 模型，
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
from models import NeuroFMMultiTask                       # NeuroFM 多任务模型（骨干 + 任务头）
from training.neurofm_trainer import NeuroFMTrainer       # NeuroFM 训练/推理封装器
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
    在跨站点数据集上评估服务器端 NeuroFM 的泛化性能。

    评估流程：
      1. 构建 NeuroFMMultiTask（仅年龄回归头）
      2. 加载已训练的 NeuroFM 权重（若路径有效）
      3. 构建跨站点测试 DataLoader（shuffle=False 保持样本顺序）
      4. 使用 NeuroFMTrainer.evaluate 进行前向推理（无梯度更新）
      5. 计算 MAE/RMSE/Pearson r/R² 指标
      6. 进行偏差分析并保存结果（CSV + 图表）

    参数：
      args          : 命令行参数对象（含 .dummy 等标志）
      cfg           : M4FedConfig 配置对象（含模型和训练参数）
      cross_dataset : 跨站点测试集（OpenMindDataset，与训练集无重叠的子数据集）
      results_dir   : 结果保存根目录（cross_site/ 子目录将在此创建）
      model_path    : 已训练 NeuroFM 权重路径（.pt 文件）；
                      若为 None 或文件不存在，则使用随机初始化权重进行基线对照

    返回：
      {'metrics': 指标字典, 'bias': 偏差分析结果字典}
    """
    device = cfg.training.device   # 计算设备（"cuda" 或 "cpu"）
    out = ensure_dir(os.path.join(results_dir, "cross_site"))  # 创建 cross_site 结果子目录
    set_seed(cfg.training.seed)    # 重置随机种子（保证可复现）

    # ------------------------------------------------------------------
    # 步骤 1：构建 NeuroFMMultiTask（仅包含年龄回归头）
    # random_init=False：骨干从 NeuroFM 预训练权重初始化（后续会被 load_model 覆盖）
    # ------------------------------------------------------------------
    task_specs = {"age": {"type": "regression", "n_classes": 1}}   # 仅年龄回归任务
    model = NeuroFMMultiTask(
        repo_id=cfg.model.neurofm_repo,   # NeuroFM HuggingFace 仓库 ID（含预训练骨干权重）
        task_specs=task_specs,            # 任务规格：只有年龄回归头
        device=device,                    # 计算设备
        random_init=False,                # 使用预训练骨干（从 HuggingFace 下载权重）
    ).to(device)                          # 将整个模型移到计算设备

    # ------------------------------------------------------------------
    # 步骤 2：加载 M4Fed 训练完成的 NeuroFM 权重（覆盖预训练骨干权重）
    # strict=False：允许部分权重匹配（若任务头结构有细微差异时不报错）
    # ------------------------------------------------------------------
    if model_path and os.path.exists(model_path):
        load_model(model, model_path, strict=False)   # 加载已训练权重
        print(f"[Cross-site] 成功加载模型权重：{model_path}")
    else:
        # 路径无效时：使用当前预训练骨干权重作为基线（对照组，无 M4Fed 微调）
        print("[Cross-site] 未找到模型路径，使用随机初始化权重（基线对照）。")

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

    # ------------------------------------------------------------------
    # 步骤 4：使用 NeuroFMTrainer.evaluate 进行前向推理
    # lr=0：纯推理模式（不创建优化器或训练步骤）
    # mixed_precision=False：评估阶段关闭 FP16（减少精度损失，提高评估准确性）
    # ------------------------------------------------------------------
    trainer = NeuroFMTrainer(
        model,                           # 已加载权重的 NeuroFM 模型
        device=device,                   # 计算设备
        lr=0,                            # 学习率为 0（表示纯推理，不更新参数）
        mixed_precision=False,           # 评估时使用 FP32（避免 FP16 截断误差影响指标）
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
    p.add_argument("--dummy", action="store_true", help="使用虚拟数据快速验证")   # 虚拟模式
    p.add_argument("--model_path", default=None, help="NeuroFM 权重路径")          # 模型路径
    p.add_argument("--results_dir", default="results")          # 结果保存目录
    args = p.parse_args()   # 解析参数

    cfg = load_config(args.config)   # 加载 M4FedConfig 配置
    # 虚拟模式用 16³ 分辨率（快速验证），正式模式用配置中的分辨率
    mri_shape = (16, 16, 16) if args.dummy else tuple(cfg.data.mri_shape)

    if args.dummy:
        # 虚拟模式：直接生成 10 个随机假样本（跳过 HuggingFace 下载）
        from data.openmind_dataset import _make_dummy_samples
        cross_samples = _make_dummy_samples(10, mri_shape=mri_shape, seed=99)
    else:
        # 正式模式：从 HuggingFace 枚举真实子数据集 ID，构建跨站点数据集
        from data.openmind_dataset import (build_node_datasets,
                                           list_openmind_ds_ids,
                                           make_openmind_sample_loader)
        # 枚举 HuggingFace 上所有子数据集 ID（如 ds000001、ds000002 ...）
        all_ds_ids = list_openmind_ds_ids(cfg.data.hf_dataset_id)
        # 创建样本加载函数（从 HuggingFace 惰性下载 NIfTI + CSV 元数据）
        sample_loader_fn = make_openmind_sample_loader(
            cfg.data.hf_dataset_id,                  # HuggingFace 仓库 ID
            cache_dir=cfg.data.cache_dir or None,    # 本地缓存目录
            max_cache_gb=cfg.data.max_cache_gb,      # LRU 磁盘缓存上限（GB）
        )
        # 构建完整节点数据集（包含 cross 子集）
        nd = build_node_datasets(
            all_ds_ids,                              # 所有子数据集 ID
            cfg.data.n_cross_ds,                     # 划分给跨站点的数据集数量
            cfg.data.n_private_ds_per_node,          # 每节点私有数据集数量
            2,                                       # 客户端数量
            cfg.data.train_split,                    # 训练/测试比例
            mri_shape,                               # MRI 目标分辨率
            cfg.data.seed,                           # 随机种子
            sample_loader_fn=sample_loader_fn,       # 真实数据加载函数
            dummy_mode=False,                        # 正式模式（不使用虚拟数据）
            n_dummy_per_ds=5,                        # 此参数在正式模式下无效
            n_public_max=cfg.data.n_public_ds_max,  # 公开数据集上限
        )
        cross_samples = nd["cross"]["all"].samples   # 取出跨站点评估集的样本列表

    # 构建跨站点 OpenMindDataset（不做数据增强）
    cross_ds = OpenMindDataset(cross_samples, mri_shape)
    # 若未指定 model_path，默认在 M4Fed 结果目录中查找服务器 NeuroFM 权重
    model_path = args.model_path or os.path.join(args.results_dir, "m4fed", "server_neurofm.pt")
    run_cross_site(args, cfg, cross_ds, args.results_dir, model_path)
