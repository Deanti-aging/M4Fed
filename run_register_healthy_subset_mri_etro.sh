#!/bin/bash
#SBATCH --job-name=m4fed_register_healthy_mri
#SBATCH --partition=FARM
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/register_healthy_subset_mri/slurm_%j.log
set -euo pipefail
source /FARM/xwang/miniconda3/etc/profile.d/conda.sh
conda activate m4fed
cd /FARM/xwang/M4Fed
mkdir -p results/register_healthy_subset_mri

# 纯 CPU 批处理（把健康受试者子集的全部 T1w 扫描配准到 MNI152 标准模板并落盘
# 缓存），不需要 GPU，故不请求 --gres，避免空占 A100。
# 必须在提交 run_healthy_subset_registered_etro.sh 之前跑完本脚本，否则训练脚本
# 会在 GPU 作业里现场配准未缓存的样本，把 CPU 密集步骤和 GPU 训练绑在一起、很慢。
python experiments/register_healthy_subset_mri.py --config configs/m4fed.yaml \
    --results_dir results/register_healthy_subset_mri --workers 16
