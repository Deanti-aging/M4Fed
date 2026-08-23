#!/bin/bash
#SBATCH --job-name=m4fed_test_visualize
#SBATCH --clusters=anansi
#SBATCH --partition=ada_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/test_visualize/slurm_%j.log
source ~/.bashrc
conda activate m4fed
cd /scratch/brussel/113/vsc11375/M4Fed

# 生成样本可视化 suite 复用 run_test_baseline.sh（--suite baseline）训练出的
# FAN/M4Fed 权重，权重固定读取自 <results_dir>/fan/client_0_medmd.pt 与
# <results_dir>/m4fed/server_medmd.pt，因此这里的 --results_dir 必须和
# run_test_baseline.sh 里的一致（results/test_baseline）。
python tests/unit_test.py --suite visualization --results_dir results/test_baseline
