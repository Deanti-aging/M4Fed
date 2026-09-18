#!/bin/bash
#SBATCH --job-name=m4fed_visualize
#SBATCH --partition=FARM
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/main/slurm_visualize_%j.log
set -euo pipefail
source /FARM/xwang/miniconda3/etc/profile.d/conda.sh
conda activate m4fed
cd /FARM/xwang/M4Fed

# 生成样本可视化复用 run_main.sh（--mode run_all）训练出的 FAN/M4Fed 权重，
# 权重固定读取自 <results_dir>/fan/client_0_medmd.pt 与 <results_dir>/m4fed/server_medmd.pt，
# 因此这里的 --results_dir 必须和 run_main.sh 里的一致（results/main）。
python main.py --mode visualize --config configs/m4fed.yaml --results_dir results/main
