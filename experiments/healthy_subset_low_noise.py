"""
M4Fed 补充实验：在 experiments/healthy_subset.py（健康受试者子集）的基础上，
额外验证「降低 MedMD 合成数据噪声强度」这一因素的影响。

与 healthy_subset.py 的区别仅一处（预热轮数 / 联邦循环 / 本地训练轮数同为
baseline 的 1 / 3 / 3，其余数据筛选/划分/超参/评估/可视化流程完全一致，直接
复用 healthy_subset.run_healthy_subset）：
  - 联邦循环阶段与本地训练阶段合成 MedMD 数据时的 img2img 噪声强度，由
     baseline 的 0.75（federated/methods/m4fed.py 联邦循环阶段固定值）与
     np.random.uniform(0.35, 0.70)（本地训练阶段随机采样区间）均降低为
     0.05，只保留“合成噪声强度高低”这一个变量。

用法：
  python experiments/healthy_subset_low_noise.py --config configs/m4fed.yaml \
      --results_dir results/healthy_subset_low_noise
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
    p = argparse.ArgumentParser(description="M4Fed 补充实验：健康受试者子集 + 降低合成噪声强度")
    p.add_argument("--config", default="configs/m4fed.yaml", help="YAML 配置文件路径")
    p.add_argument("--results_dir", default="results/healthy_subset_low_noise",
                    help="结果保存目录")
    return p.parse_args()


def main():
    args = parse_args()
    log_path = setup_run_logger(args.results_dir)
    print(f"[healthy-low-noise] 终端日志将同步保存至 {log_path}")
    cfg = load_config(args.config)
    # 预热/联邦循环/本地训练轮数为 baseline 的 1/3/3；联邦循环 + 本地训练阶段的
    # 合成噪声强度均降至 0.05——与 healthy_subset.py 唯一的差异，其余
    # 数据筛选/划分/超参/评估/可视化逻辑完全共用。
    run_healthy_subset(args, cfg, rounds_override=(1, 3, 3),
                        noise_override=(0.05, 0.05, 0.05))


if __name__ == "__main__":
    main()
