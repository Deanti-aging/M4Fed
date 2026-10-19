"""
M4Fed 补充实验：在 experiments/healthy_subset_registered.py（健康子集 + 配准到标准模板，
轮数 1 / 3 / 3）的基础上，额外对领域模型的年龄训练标签做 z-score 标准化。

与 healthy_subset_registered.py 的唯一区别：
  run_m4fed(age_zscore=True)——每个节点用「自身真实训练集」的年龄均值 / 标准差把
  SwinMultiTask 的年龄目标变为 (age - mean) / std（服务器：公开 + 私有训练集；客户端：
  私有训练集；各节点独立统计，不共享标签统计量），评估时把输出反变换回原始年龄再计算
  MAE / RMSE / r / R2，因此指标与原实验口径一致、可直接对比。统计量保存在
  m4fed/age_zscore_stats.json。MedMD 文本条件、TabPFN 补全、合成样本标签、数据划分、
  超参数全部不变。

训练结束后自动：
  1. 从保存的 m4fed/<node>_swin.pt 重新推理测试集，导出逐样本预测 CSV 与多节点散点图
     （experiments/export_healthy_subset_predictions.py，与复现 healthy_subset_registered_133
     用的是同一段代码）；
  2. 若 --baseline_dir 下已有导出的 m4fed/test_predictions.csv，再画一张坐标轴统一的
     并排对比散点图 figures/test_scatter_compare.<fmt>。

用法：
  python experiments/healthy_subset_registered_age_zscore.py --config configs/m4fed.yaml \
      --results_dir results/healthy_subset_registered_age_zscore \
      --baseline_dir results/healthy_subset_registered_133
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.env_utils import load_local_env
load_local_env()

from configs import load_config
from utils.log_utils import setup_run_logger
from experiments.healthy_subset import run_healthy_subset
from experiments.export_healthy_subset_predictions import export_predictions, compare_with


def parse_args():
    p = argparse.ArgumentParser(description="M4Fed 补充实验：健康子集 + 配准 + 年龄标签 z-score")
    p.add_argument("--config", default="configs/m4fed.yaml", help="YAML 配置文件路径")
    p.add_argument("--results_dir", default="results/healthy_subset_registered_age_zscore",
                   help="结果保存目录")
    p.add_argument("--baseline_dir", default="results/healthy_subset_registered_133",
                   help="未做 z-score 的对照实验目录（用于画对比散点图，不存在则跳过）")
    p.add_argument("--batch_size", type=int, default=4, help="导出预测时的推理批大小")
    return p.parse_args()


def main():
    args = parse_args()
    log_path = setup_run_logger(args.results_dir)
    print(f"[healthy-registered-zscore] 终端日志将同步保存至 {log_path}")
    cfg = load_config(args.config)
    if not cfg.eval.save_model:
        cfg.eval.save_model = True   # 导出测试集预测依赖保存的 checkpoint
    run_healthy_subset(args, cfg, rounds_override=(1, 3, 3), register_to_template=True,
                       age_zscore=True)

    # 用重新加载的配置导出，避免训练过程对内存中 cfg 的改动（如 device 降级）影响推理
    cfg = load_config(args.config)
    all_df = export_predictions(cfg, args.results_dir, register_to_template=True,
                                batch_size=args.batch_size)
    if args.baseline_dir:
        compare_with(args.results_dir, all_df, [args.baseline_dir], cfg.eval.figure_format)


if __name__ == "__main__":
    main()
