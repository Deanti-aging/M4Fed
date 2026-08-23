#!/bin/bash
#SBATCH --job-name=m4fed_test_cross_site
#SBATCH --clusters=anansi
#SBATCH --partition=ada_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/test_cross_site/slurm_%j.log
source ~/.bashrc
conda activate m4fed
cd /scratch/brussel/113/vsc11375/M4Fed
python tests/unit_test.py --suite cross_site --results_dir results/test_cross_site
