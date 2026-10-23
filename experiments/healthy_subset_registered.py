"""
M4Fed 补充实验：在 experiments/healthy_subset.py（健康受试者子集）的基础上，
额外验证「配准到标准模板」这一 MRI 预处理步骤的影响。

与 healthy_subset.py 的区别仅一处（预热轮数 / 联邦循环 / 本地训练轮数同为
baseline 的 1 / 3 / 3，其余训练/评估/可视化流程完全一致，直接复用
healthy_subset.run_healthy_subset，只保留“是否配准”这一个变量）：
  - 每张 MRI 在 preprocess_mri（裁剪/归一化/插值）之前，先仿射配准到
     MNI152 标准模板空间（见 data/registration.py），配准结果落盘缓存于
     data/mri_registered_cache/。

运行前置条件：
  配准非常耗时（单张约 10~30 秒，健康子集约 7000+ 张 T1w 扫描），必须先跑
  完 experiments/register_healthy_subset_mri.py 把全部样本的配准结果缓存
  好（纯 CPU 批处理，可独立于本训练脚本单独提交/重跑）。本脚本不强制校验
  缓存是否齐全——未命中缓存的样本会在训练时现场配准（兜底，见
  data/openmind_dataset.py::OpenMindDataset._read_volume），但那样会让
  GPU 训练作业等待 CPU 配准，体验很差，不建议依赖这条兜底路径。

用法：
  python experiments/register_healthy_subset_mri.py --config configs/m4fed.yaml   # 先跑这个
  python experiments/healthy_subset_registered.py --config configs/m4fed.yaml \
      --results_dir results/healthy_subset_registered
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import argparse                        # 命令行参数解析
import os                              # 路径操作
import sys                             # sys.path 修改

# 将项目根目录加入 Python 路径（使脚本可直接运行，不依赖 package install）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.env_utils import load_local_env   # 从 .env.local 加载本地密钥
load_local_env()   # 必须在可能触发 tabpfn 等库读取环境变量的导入之前执行

from configs import load_config             # 加载 YAML 配置
from utils.log_utils import setup_run_logger  # 终端输出同步落盘
from experiments.healthy_subset import run_healthy_subset


def parse_args():
    p = argparse.ArgumentParser(description="M4Fed 补充实验：健康受试者子集 + 配准到标准模板")
    p.add_argument("--config", default="configs/m4fed.yaml", help="YAML 配置文件路径")
    p.add_argument("--results_dir", default="results/healthy_subset_registered",
                    help="结果保存目录")
    return p.parse_args()


def main():
    args = parse_args()
    log_path = setup_run_logger(args.results_dir)
    print(f"[healthy-registered] 终端日志将同步保存至 {log_path}")
    cfg = load_config(args.config)
    # 轮数改回 baseline 的 1/3/3；额外开启配准到标准模板——与 healthy_subset.py
    # 唯一的两处差异，其余数据筛选/划分/超参/评估/可视化逻辑完全共用。
    run_healthy_subset(args, cfg, rounds_override=(1, 3, 3), register_to_template=True)


if __name__ == "__main__":
    main()
