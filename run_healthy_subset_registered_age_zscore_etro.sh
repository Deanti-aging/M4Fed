#!/bin/bash
#SBATCH --job-name=m4fed_healthy_reg_zscore
#SBATCH --partition=FARM
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=results/healthy_subset_registered_age_zscore/slurm_%j.log
set -euo pipefail
source /FARM/xwang/miniconda3/etc/profile.d/conda.sh
conda activate m4fed
cd /FARM/xwang/M4Fed
mkdir -p results/healthy_subset_registered_age_zscore

# 与 run_healthy_subset_registered_etro.sh 相同（健康子集 + 配准 + 轮数 1/3/3），
# 唯一差异：领域模型的年龄训练标签做 z-score 标准化（各节点用自身训练集统计量）。
# 训练结束后自动导出测试集逐样本预测 CSV + 多节点散点图；若对照组
# results/healthy_subset_registered_133 已用 run_export_healthy_subset_registered_133_etro.sh
# 导出过 m4fed/test_predictions.csv，还会额外画一张并排对比散点图。
CACHE_DIR=data/mri_registered_cache
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo "[run_healthy_subset_registered_age_zscore] 警告：$CACHE_DIR 为空，尚未预先配准，" >&2
    echo "[run_healthy_subset_registered_age_zscore] 训练时会现场配准，速度会很慢。" >&2
fi

python experiments/healthy_subset_registered_age_zscore.py --config configs/m4fed.yaml \
    --results_dir results/healthy_subset_registered_age_zscore \
    --baseline_dir results/healthy_subset_registered_133 \
    --batch_size 4
