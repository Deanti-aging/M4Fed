#!/bin/bash
#SBATCH --job-name=m4fed_sensitivity
#SBATCH --clusters=hydra
#SBATCH --partition=ampere_gpu,hopper_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/sensitivity/slurm_%j.log
source ~/.bashrc
conda activate m4fed
cd /scratch/brussel/113/vsc11375/M4Fed
python main.py --mode sensitivity --config configs/m4fed.yaml --results_dir results/sensitivity
