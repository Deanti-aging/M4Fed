#!/bin/bash
#SBATCH --job-name=m4fed_sensitivity
#SBATCH --partition=ada_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=90:00:00
#SBATCH --output=results/sensitivity/slurm_%j.log
source ~/.bashrc
conda activate m4fed
cd /scratch/brussel/113/vsc11375/M4Fed
python main.py --mode sensitivity --config configs/m4fed.yaml --results_dir results/sensitivity
