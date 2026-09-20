#!/bin/bash
#SBATCH --job-name=m4fed_healthy_low_noise
#SBATCH --partition=FARM
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=results/healthy_subset_low_noise/slurm_%j.log
set -euo pipefail
source /FARM/xwang/miniconda3/etc/profile.d/conda.sh
conda activate m4fed
cd /FARM/xwang/M4Fed
mkdir -p results/healthy_subset_low_noise
python experiments/healthy_subset_low_noise.py --config configs/m4fed.yaml \
    --results_dir results/healthy_subset_low_noise
