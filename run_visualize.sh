#!/bin/bash
#SBATCH --job-name=m4fed_visualize
#SBATCH --partition=ada_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/main/slurm_visualize_%j.log
source ~/.bashrc
conda activate m4fed
cd /scratch/brussel/113/vsc11375/M4Fed

# 生成样本可视化复用 run_main.sh（--mode run_all）训练出的 FAN/M4Fed 权重，
# 权重固定读取自 <results_dir>/fan/client_0_medmd.pt 与 <results_dir>/m4fed/server_medmd.pt，
# 因此这里的 --results_dir 必须和 run_main.sh 里的一致（results/main）。
python main.py --mode visualize --config configs/m4fed.yaml --results_dir results/main
