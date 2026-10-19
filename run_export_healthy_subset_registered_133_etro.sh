#!/bin/bash
#SBATCH --job-name=m4fed_export_healthy133
#SBATCH --partition=FARM
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/healthy_subset_registered_133/slurm_export_preds_%j.log
set -euo pipefail
source /FARM/xwang/miniconda3/etc/profile.d/conda.sh
conda activate m4fed
cd /FARM/xwang/M4Fed

# 不重新训练：加载 results/healthy_subset_registered_133/m4fed/<node>_swin.pt（未做年龄
# z-score 的对照组），按 healthy_split_assignment.csv 还原训练时的划分，在 server /
# client_0 / client_1 测试集上重新推理，保存原始逐样本预测 CSV 与多节点散点图到该目录。
# --config 必须与训练时一致（train_split / seed / mri_shape）。
# 若 z-score 实验已经跑完，顺带画并排对比图（目录不存在时自动跳过）。
python experiments/export_healthy_subset_predictions.py --config configs/m4fed.yaml \
    --results_dir results/healthy_subset_registered_133 \
    --batch_size 4 \
    --compare_with results/healthy_subset_registered_age_zscore
