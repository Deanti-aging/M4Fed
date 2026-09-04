#!/bin/bash
#SBATCH --job-name=m4fed_healthy_subset
#SBATCH --clusters=hydra
#SBATCH --partition=ampere_gpu,hopper_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/healthy_subset/slurm_%j.log
source ~/.bashrc
conda activate m4fed
cd /scratch/brussel/113/vsc11375/M4Fed
mkdir -p results/healthy_subset
python experiments/healthy_subset.py --config configs/m4fed.yaml --results_dir results/healthy_subset
