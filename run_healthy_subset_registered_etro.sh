#!/bin/bash
#SBATCH --job-name=m4fed_healthy_registered
#SBATCH --partition=FARM
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=results/healthy_subset_registered/slurm_%j.log
set -euo pipefail
source /FARM/xwang/miniconda3/etc/profile.d/conda.sh
conda activate m4fed
cd /FARM/xwang/M4Fed
mkdir -p results/healthy_subset_registered

# 依赖 run_register_healthy_subset_mri_etro.sh 已经把健康受试者子集的配准结果
# 缓存到 data/mri_registered_cache/；未缓存的样本训练时会现场配准（兜底逻辑见
# data/openmind_dataset.py::OpenMindDataset._read_volume），但那样会让本该跑
# GPU 训练的作业去等 CPU 配准，只是提醒，不阻塞本脚本运行。
CACHE_DIR=data/mri_registered_cache
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo "[run_healthy_subset_registered] 警告：$CACHE_DIR 为空，尚未预先配准。" >&2
    echo "[run_healthy_subset_registered] 建议先提交 run_register_healthy_subset_mri_etro.sh，" >&2
    echo "[run_healthy_subset_registered] 否则本作业会在训练过程中现场配准，速度会很慢。" >&2
fi

python experiments/healthy_subset_registered.py --config configs/m4fed.yaml \
    --results_dir results/healthy_subset_registered
