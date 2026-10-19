#!/bin/bash
#SBATCH --job-name=m4fed_export_preds
#SBATCH --partition=FARM
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/main/slurm_export_preds_%j.log
set -euo pipefail
source /FARM/xwang/miniconda3/etc/profile.d/conda.sh
conda activate m4fed
cd /FARM/xwang/M4Fed

# 不重新训练：加载 run_main_etro.sh（--mode run_all）留下的 <method>/<node>_swin.pt，
# 在与 run_all 相同的节点测试集上重新推理，补全逐样本 真实/预测年龄 CSV 与按节点着色的散点图。
# --config / --results_dir 必须与 run_main_etro.sh 一致，才能重建出相同的数据划分。
python experiments/export_test_predictions.py --config configs/m4fed.yaml \
    --results_dir results/main --batch_size 4
