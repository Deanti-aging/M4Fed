#!/bin/bash
#SBATCH --job-name=m4fed_cross_site
#SBATCH --clusters=hydra
#SBATCH --partition=ampere_gpu,hopper_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/main/slurm_cross_%j.log
source ~/.bashrc
conda activate m4fed
cd /scratch/brussel/113/vsc11375/M4Fed

# cross_site 复用 run_main.sh（--mode run_all）已训练好的 M4Fed 权重，
# 权重固定读取自 <results_dir>/m4fed/server_swin.pt，因此这里的 --results_dir
# 必须和 run_main.sh 里的一致（results/main），否则会用随机初始化权重做基线对照。
CKPT=results/main/m4fed/server_swin.pt
if [ ! -f "$CKPT" ]; then
    echo "[run_cross_site] 未找到 $CKPT，请先跑完 run_main.sh（--mode run_all）再提交本脚本。" >&2
    exit 1
fi

python main.py --mode cross_site --config configs/m4fed.yaml --results_dir results/main
