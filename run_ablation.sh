#!/bin/bash
#SBATCH --job-name=m4fed_ablation
#SBATCH --partition=ada_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --output=results/ablation/slurm_%j.log
source ~/.bashrc
conda activate m4fed
cd /scratch/brussel/113/vsc11375/M4Fed
python main.py --mode ablation --config configs/m4fed.yaml --results_dir results/ablation
