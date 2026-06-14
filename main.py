"""
M4Fed 统一命令行入口。

使用示例：
  # 运行所有基线（完整模式，需安装所有依赖）：
  python main.py --mode run_all --config configs/m4fed.yaml

  # 快速冒烟测试（使用极小的虚拟数据）：
  python main.py --mode test

  # 单独运行各实验：
  python main.py --mode ablation    --config configs/m4fed.yaml --dummy
  python main.py --mode sensitivity --config configs/m4fed.yaml --dummy
  python main.py --mode cross_site  --config configs/m4fed.yaml --dummy
"""
from __future__ import annotations
import argparse
import os
import sys

# 将当前文件所在目录加入模块搜索路径，使包内相对导入正常工作
sys.path.insert(0, os.path.dirname(__file__))

from utils.seed_utils import set_seed


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="M4Fed 联邦学习框架")
    # --mode：选择要执行的实验模式
    p.add_argument("--mode", choices=["run_all", "test", "ablation",
                                       "sensitivity", "cross_site"],
                   default="test", help="要运行的实验类型")
    # --config：YAML 配置文件路径
    p.add_argument("--config", default="configs/m4fed.yaml",
                   help="YAML 配置文件路径")
    # --results_dir：结果保存目录
    p.add_argument("--results_dir", default="results",
                   help="结果保存目录")
    # --dummy：是否使用极小的虚拟数据进行快速冒烟测试
    p.add_argument("--dummy", action="store_true",
                   help="使用极小的虚拟数据进行快速冒烟测试")
    return p.parse_args()


def main():
    """主函数：解析参数、加载配置、分发到各实验模块。"""
    args = parse_args()

    # 加载 YAML 配置（不存在时退回默认值）
    from configs import load_config
    cfg = load_config(args.config)

    # 固定全局随机种子，保证可复现性
    set_seed(cfg.training.seed)

    # 根据 --dummy 参数决定 MRI 体积尺寸和数据集 ID 列表
    if args.dummy:
        # 虚拟模式：使用极小体积（16³）和 100 个假数据集 ID
        mri_shape = (16, 16, 16)
        all_ds_ids = [f"ds{i:06d}" for i in range(100)]
    else:
        # 正式模式：从配置文件读取体积尺寸，从 HuggingFace 加载真实数据集 ID
        mri_shape = tuple(cfg.data.mri_shape)
        all_ds_ids = _load_ds_ids(cfg)

    # 确保结果目录存在
    os.makedirs(args.results_dir, exist_ok=True)

    # ---------- 分发到各实验模式 ----------

    if args.mode == "test":
        # 运行单元测试（通过子进程执行 tests/unit_test.py）
        print("=== 运行单元测试 ===")
        import subprocess
        result = subprocess.run([sys.executable, "tests/unit_test.py"],
                                capture_output=False)
        sys.exit(result.returncode)

    elif args.mode == "run_all":
        # 运行全部 5 个联邦基线，并生成对比图表
        from experiments.run_all_baselines import main as run_all_main
        # 将参数通过 sys.argv 传递给子模块（子模块有自己的 parse_args）
        sys.argv = [sys.argv[0],
                    "--config", args.config,
                    "--results_dir", args.results_dir]
        if args.dummy:
            sys.argv.append("--dummy")
        run_all_main()

    elif args.mode == "ablation":
        # 消融实验：测试 M4Fed 各组件的贡献
        from experiments.ablation import run_ablation
        task_specs = {"age": {"type": "regression", "n_classes": 1}}
        run_ablation(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)

    elif args.mode == "sensitivity":
        # 超参数敏感性分析：扫描合成数据比例、学习率、Non-IID 程度
        from experiments.sensitivity import run_sensitivity
        task_specs = {"age": {"type": "regression", "n_classes": 1}}
        run_sensitivity(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)

    elif args.mode == "cross_site":
        # 跨站点泛化实验：用服务器训练好的 NeuroFM 在跨站点数据上评估
        from experiments.cross_site import run_cross_site
        from data.openmind_dataset import build_node_datasets, OpenMindDataset

        # 构建节点数据集（取其中的 cross 子集作为跨站点评估集）
        nd = build_node_datasets(
            all_ds_ids=all_ds_ids,
            n_cross=cfg.data.n_cross_ds,
            n_private_per_node=cfg.data.n_private_ds_per_node,
            n_clients=2,
            train_split=cfg.data.train_split,
            mri_shape=mri_shape,
            seed=cfg.data.seed,
            dummy_mode=args.dummy,
            n_dummy_per_ds=5,
        )
        # cross["all"] 包含所有 5 个跨站点子数据集的样本
        cross_ds = nd["cross"]["all"]
        # 期望 M4Fed 训练完成后的服务器模型路径
        model_path = os.path.join(args.results_dir, "m4fed", "server_neurofm.pt")
        run_cross_site(args, cfg, cross_ds, args.results_dir, model_path)


def _load_ds_ids(cfg) -> list:
    """
    从 HuggingFace 流式加载 OpenMind 数据集的子数据集 ID 列表。
    如果加载失败（无网络、未安装 datasets 库等），则退回 100 个虚拟 ID。
    """
    try:
        from datasets import load_dataset
        print("[main] 正在流式加载 OpenMind 数据集 ID...")
        ds = load_dataset(
            cfg.data.hf_dataset_id,
            split="train",
            streaming=True,       # 流式模式：不下载全部数据
            trust_remote_code=True,
        )
        ids = set()
        for sample in ds:
            # 尝试从样本中读取数据集 ID 字段（两种可能的字段名）
            ds_id = sample.get("dataset_id") or sample.get("ds_id") or "unknown"
            ids.add(str(ds_id))
            if len(ids) >= 300:   # 最多收集 300 个不同 ID
                break
        return sorted(ids)
    except Exception as e:
        print(f"[main] HuggingFace 加载失败: {e}。改用虚拟 ID。")
        return [f"ds{i:06d}" for i in range(100)]


if __name__ == "__main__":
    main()
