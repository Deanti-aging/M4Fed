#!/bin/bash
#SBATCH --job-name=m4fed_central_reg_zscore
#SBATCH --partition=FARM
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-20:00:00
#SBATCH --output=results/centralized_healthy_registered_age_zscore/slurm_%j.log
set -euo pipefail
source /FARM/xwang/miniconda3/etc/profile.d/conda.sh
conda activate m4fed
cd /FARM/xwang/M4Fed
mkdir -p results/centralized_healthy_registered_age_zscore

# 中心化对照实验：健康子集 + 配准 + 年龄标签 z-score，不做联邦学习。
# 按 results/healthy_subset_registered_133/healthy_split_assignment.csv 还原受试者级划分，
# 把所有节点的训练数据合并到一个节点，对加载了 MONAI SwinViT 预训练权重的 SwinMultiTask
# 全参数微调 10 轮；测试集 = 联邦实验三个节点测试集的并集（逐样本一致）。
# 每轮保存测试集逐样本预测 CSV、MAE/RMSE/r/R2、train/test 损失；最后一轮额外保存
# 散点图、年龄 × 性别误差热图、损失曲线和 centralized/swin_final.pt。
CACHE_DIR=data/mri_registered_cache
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo "[run_centralized_healthy_registered_age_zscore] 警告：$CACHE_DIR 为空，尚未预先配准，" >&2
    echo "[run_centralized_healthy_registered_age_zscore] 训练时会现场配准，速度会很慢。" >&2
fi

python experiments/centralized_healthy_registered_age_zscore.py --config configs/m4fed.yaml \
    --results_dir results/centralized_healthy_registered_age_zscore \
    --split_dir results/healthy_subset_registered_133 \
    --epochs 10 \
    --eval_batch_size 4
