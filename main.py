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
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import argparse                       # 命令行参数解析（标准库）
import os                             # 操作系统路径操作
import sys                            # sys.path 修改和 sys.exit

# 将当前文件所在目录加入模块搜索路径，使包内相对导入正常工作
# （在从任意工作目录运行 main.py 时保证 `from configs import ...` 等能找到包）
sys.path.insert(0, os.path.dirname(__file__))

from utils.seed_utils import set_seed   # 全局随机种子设置（保证可复现）


def parse_args():
    """
    解析命令行参数，返回 Namespace 对象。

    可用参数：
      --mode       : 实验模式（test/run_all/ablation/sensitivity/cross_site）
      --config     : YAML 配置文件路径（不存在时退回默认配置）
      --results_dir: 结果保存目录（自动创建）
      --dummy      : 是否使用极小虚拟数据进行快速冒烟测试（跳过 HuggingFace 下载）
    """
    p = argparse.ArgumentParser(description="M4Fed 联邦学习框架")
    # --mode：选择要执行的实验模式
    p.add_argument("--mode", choices=["run_all", "test", "ablation",
                                       "sensitivity", "cross_site"],
                   default="test",              # 默认运行单元测试
                   help="要运行的实验类型")
    # --config：YAML 配置文件路径（支持相对路径）
    p.add_argument("--config", default="configs/m4fed.yaml",
                   help="YAML 配置文件路径")
    # --results_dir：所有结果的保存根目录（子实验会在此创建对应子目录）
    p.add_argument("--results_dir", default="results",
                   help="结果保存目录")
    # --dummy：开关标志，启用后跳过真实数据下载，用随机噪声体积替代
    p.add_argument("--dummy", action="store_true",
                   help="使用极小的虚拟数据进行快速冒烟测试")
    return p.parse_args()   # 解析并返回命令行参数


def main():
    """
    主函数：解析参数、加载配置、分发到各实验模块。

    分发逻辑：
      - test       : 调用 tests/unit_test.py（端对端冒烟测试，通过子进程，退出码传递）
      - run_all    : 调用 experiments/run_all_baselines.main()
      - ablation   : 调用 experiments/ablation.run_ablation()
      - sensitivity: 调用 experiments/sensitivity.run_sensitivity()
      - cross_site : 调用 experiments/cross_site.run_cross_site()
    """
    args = parse_args()   # 解析命令行参数

    # 加载 YAML 配置（若 YAML 文件不存在则退回默认 M4FedConfig）
    from configs import load_config
    cfg = load_config(args.config)   # 返回 M4FedConfig 对象

    # 固定全局随机种子（PyTorch / NumPy / random 等），保证可复现性
    set_seed(cfg.training.seed)

    # 根据 --dummy 参数决定 MRI 体积尺寸和数据集 ID 列表
    if args.dummy:
        # 虚拟模式：使用极小体积（16³ 体素）和 100 个假数据集 ID
        mri_shape = (16, 16, 16)                           # 16³ 体素（远小于真实 96³）
        all_ds_ids = [f"ds{i:06d}" for i in range(100)]   # 100 个假 ID（ds000000~ds000099）
    else:
        # 正式模式：从配置文件读取体积尺寸，从 HuggingFace 枚举真实数据集 ID
        mri_shape = tuple(cfg.data.mri_shape)   # 目标 MRI 分辨率（如 (96, 96, 96)）
        all_ds_ids = _load_ds_ids(cfg)          # 调用 HuggingFace API 列出真实 ID

    # 确保结果根目录存在（exist_ok=True：目录已存在时不报错）
    os.makedirs(args.results_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 根据 --mode 分发到对应实验函数                                         #
    # ------------------------------------------------------------------ #

    if args.mode == "test":
        # 运行冒烟测试（通过子进程执行 tests/unit_test.py，透传退出码）
        print("=== 运行冒烟测试 ===")
        import subprocess
        # capture_output=False：将 unit_test.py 的输出直接打印到终端（方便查看失败信息）
        result = subprocess.run([sys.executable, "tests/unit_test.py"],
                                capture_output=False)
        sys.exit(result.returncode)   # 透传退出码（0=通过，非0=失败）

    elif args.mode == "run_all":
        # 运行全部 5 个联邦学习基线（FedAvg/FedProx/FAN/GPT-FL/M4Fed），生成对比图表
        from experiments.run_all_baselines import main as run_all_main
        # 通过修改 sys.argv 将参数传给子模块（子模块有自己的 parse_args）
        sys.argv = [sys.argv[0],
                    "--config", args.config,       # 传入配置文件路径
                    "--results_dir", args.results_dir]  # 传入结果目录
        if args.dummy:
            sys.argv.append("--dummy")   # 若开启虚拟模式则追加 --dummy 标志
        run_all_main()   # 调用 run_all_baselines 的 main 函数

    elif args.mode == "ablation":
        # 消融实验：逐一移除 M4Fed 的各关键组件，验证每个组件的贡献
        from experiments.ablation import run_ablation
        task_specs = {"age": {"type": "regression", "n_classes": 1}}   # 年龄回归任务
        run_ablation(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)

    elif args.mode == "sensitivity":
        # 超参数敏感性分析：扫描合成比例/MedMD 学习率/Non-IID 程度
        from experiments.sensitivity import run_sensitivity
        task_specs = {"age": {"type": "regression", "n_classes": 1}}   # 年龄回归任务
        run_sensitivity(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)

    elif args.mode == "cross_site":
        # 跨站点泛化实验：在未见过的采集中心数据上评估服务器端 NeuroFM
        from experiments.cross_site import run_cross_site
        from data.openmind_dataset import build_node_datasets, OpenMindDataset, make_openmind_sample_loader

        # 非 dummy 模式：创建真实数据加载函数（从 HuggingFace 惰性下载 NIfTI + 元数据）
        sample_loader_fn = None if args.dummy else make_openmind_sample_loader(
            cfg.data.hf_dataset_id,                  # HuggingFace 数据集仓库 ID
            cache_dir=cfg.data.cache_dir or None,    # 本地缓存目录
            max_cache_gb=cfg.data.max_cache_gb,      # LRU 缓存容量上限（GB）
        )
        # 构建完整节点数据集（服务器 + 2 客户端 + 跨站点评估集）
        nd = build_node_datasets(
            all_ds_ids=all_ds_ids,                              # 所有数据集 ID
            n_cross=cfg.data.n_cross_ds,                         # 跨站点数据集数量
            n_private_per_node=cfg.data.n_private_ds_per_node,  # 每节点私有数据集数量
            n_clients=2,                                         # 客户端数量
            train_split=cfg.data.train_split,                    # 训练/测试比例
            mri_shape=mri_shape,                                 # MRI 目标分辨率
            seed=cfg.data.seed,                                  # 数据划分随机种子
            sample_loader_fn=sample_loader_fn,                   # 样本加载函数
            dummy_mode=args.dummy,                               # 是否虚拟模式
            n_dummy_per_ds=5,                                    # 虚拟样本数/数据集
            n_public_max=cfg.data.n_public_ds_max,              # 公开数据集上限
        )
        # nd["cross"]["all"] 包含所有跨站点子数据集的样本（不分 train/test，全部用于评估）
        cross_ds = nd["cross"]["all"]
        # 期望的 M4Fed 服务器端 NeuroFM 权重路径（M4Fed 训练完成后生成）
        model_path = os.path.join(args.results_dir, "m4fed", "server_neurofm.pt")
        run_cross_site(args, cfg, cross_ds, args.results_dir, model_path)

    # ------------------------------------------------------------------
    # 保存全局运行耗时分布（网络下载/各阶段训练/评估等）
    # 注：test 模式在上面已通过 sys.exit() 提前退出，不会执行到这里
    # ------------------------------------------------------------------
    from utils.timing_utils import save_timing_csv
    save_timing_csv(os.path.join(args.results_dir, "timing_breakdown.csv"))


def _load_ds_ids(cfg) -> list:
    """
    获取 OpenMind 有效子数据集 ID 列表（T1w 模态 + age 非空）。

    优先级：
      1. 本地 openneuro_metadata.csv（cfg.data.local_metadata_csv），覆盖全量数据集，无网络开销
      2. HuggingFace 元数据接口（hf_hub_download openneuro_metadata.csv），需网络
      3. 100 个虚拟 ID（离线调试用，不可用于正式实验）

    参数：
      cfg : M4FedConfig（读取 local_metadata_csv、hf_dataset_id、cache_dir 等参数）

    返回：
      子数据集 ID 字符串列表（如 ["ds000001", "ds000002", ...]）
    """
    # 1. 优先读取本地 CSV（全量 ds_id，无需 HuggingFace API 请求）
    local_csv = cfg.data.local_metadata_csv
    if local_csv and os.path.exists(local_csv):
        from data.openmind_dataset import list_ds_ids_from_metadata
        print(f"[main] 从本地 {local_csv} 读取子数据集 ID...")
        return list_ds_ids_from_metadata(local_csv)
    # 2. 本地文件不存在，通过 HuggingFace 下载元数据 CSV
    try:
        from data.openmind_dataset import _load_openmind_metadata
        print("[main] 本地 CSV 不存在，从 HuggingFace 下载元数据...")
        df = _load_openmind_metadata(
            cfg.data.hf_dataset_id, cfg.data.cache_dir or None, "T1w"
        )
        df = df[df["age"].notna()]
        return sorted(df["ds_id"].unique().tolist())
    except Exception as e:
        # 任何异常（网络错误、未安装 huggingface_hub 等）都退回虚拟 ID
        print(f"[main] HuggingFace 加载失败: {e}。改用虚拟 ID。")
        return [f"ds{i:06d}" for i in range(100)]   # 100 个虚拟 ID


if __name__ == "__main__":
    main()   # 程序入口（直接运行 main.py 时调用）
