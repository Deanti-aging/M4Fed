"""
M4Fed 统一命令行入口。

使用示例：
  # 运行所有基线（完整模式，需安装所有依赖）：
  python main.py --mode run_all --config configs/m4fed.yaml

  # 快速冒烟测试（使用极小的虚拟数据）：
  python main.py --mode test

  # 单独运行各实验：
  python main.py --mode ablation    --config configs/m4fed.yaml
  python main.py --mode sensitivity --config configs/m4fed.yaml
  python main.py --mode cross_site  --config configs/m4fed.yaml

  # 生成样本可视化（独立实验，依赖 run_all 训练出的 FAN/M4Fed 权重，需先跑完 run_all）：
  python main.py --mode visualize   --config configs/m4fed.yaml --results_dir results/main
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import argparse                       # 命令行参数解析（标准库）
import os                             # 操作系统路径操作
import sys                            # sys.path 修改和 sys.exit

# 将当前文件所在目录加入模块搜索路径，使包内相对导入正常工作
# （在从任意工作目录运行 main.py 时保证 `from configs import ...` 等能找到包）
sys.path.insert(0, os.path.dirname(__file__))

from utils.env_utils import load_local_env   # 从 .env.local 加载本地密钥（如 TABPFN_TOKEN）
load_local_env()   # 必须在其他模块（可能间接触发 tabpfn 等库读取环境变量）导入前执行

from utils.seed_utils import set_seed   # 全局随机种子设置（保证可复现）


def parse_args():
    """
    解析命令行参数，返回 Namespace 对象。

    可用参数：
      --mode       : 实验模式（test/run_all/ablation/sensitivity/cross_site/visualize）
      --config     : YAML 配置文件路径（不存在时退回默认配置）
      --results_dir: 结果保存目录（自动创建）
    """
    p = argparse.ArgumentParser(description="M4Fed 联邦学习框架")
    # --mode：选择要执行的实验模式
    p.add_argument("--mode", choices=["run_all", "test", "ablation",
                                       "sensitivity", "cross_site", "visualize"],
                   default="test",              # 默认运行单元测试
                   help="要运行的实验类型")
    # --config：YAML 配置文件路径（支持相对路径）
    p.add_argument("--config", default="configs/m4fed.yaml",
                   help="YAML 配置文件路径")
    # --results_dir：所有结果的保存根目录（子实验会在此创建对应子目录）
    p.add_argument("--results_dir", default="results",
                   help="结果保存目录")
    # --m4fed_results_dir：ablation 模式下变体 D 的 mixed_train_* 对照读取
    # M4Fed 完整训练服务器模型（m4fed/server_swin.pt）的目录，默认指向 main 的输出
    p.add_argument("--m4fed_results_dir", default="results/main",
                   help="ablation 变体 D mixed_train_* 对照读取的 M4Fed 完整训练结果目录")
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
      - visualize  : 调用 evaluation/generative_eval.visualize_fan_and_m4fed_samples()
                     （独立实验，依赖 run_all 训练出的 FAN/M4Fed 权重）
    """
    args = parse_args()   # 解析命令行参数

    # 尽早重定向 stdout/stderr：此后所有终端输出（含本函数下方的打印和各实验模块的打印）
    # 都会同步保存到 {results_dir}/run.log，终端显示效果不变，方便训练结束后回看
    from utils.log_utils import setup_run_logger
    log_path = setup_run_logger(args.results_dir)
    print(f"[main] 终端日志将同步保存至 {log_path}")

    # 加载 YAML 配置（若 YAML 文件不存在则退回默认 M4FedConfig）
    from configs import load_config
    cfg = load_config(args.config)   # 返回 M4FedConfig 对象

    # 固定全局随机种子（PyTorch / NumPy / random 等），保证可复现性
    set_seed(cfg.training.seed)

    mri_shape = tuple(cfg.data.mri_shape)   # 目标 MRI 分辨率（如 (96, 96, 96)）
    # visualize 模式只读取已训练好的 FAN/M4Fed 权重做生成可视化，不需要数据集 ID 列表，
    # 跳过该 HuggingFace API 调用可避免不必要的网络依赖
    all_ds_ids = None if args.mode == "visualize" else _load_ds_ids(cfg)

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

    elif args.mode == "visualize":
        # 独立的生成样本可视化实验：加载 run_all（--mode run_all）留下的 FAN（client_0）
        # 与 M4Fed（server）已训练完成的生成模型，各随机生成一个 3D MRI 并保存 2D 切片。
        # 权重文件不存在时（尚未跑过 run_all，或该次 run_all 未训练该方法）会跳过对应
        # 方法并打印提示，不影响退出码。--results_dir 需与对应 run_all 一致。
        print("=== 运行生成样本可视化 ===")
        from evaluation.generative_eval import visualize_fan_and_m4fed_samples
        visualize_fan_and_m4fed_samples(cfg, args.results_dir, cfg.training.device, mri_shape=mri_shape)
        return

    elif args.mode == "run_all":
        # 运行全部 5 个联邦学习基线（FedAvg/FedProx/FAN/GPT-FL/M4Fed），生成对比图表
        from experiments.run_all_baselines import main as run_all_main
        # 通过修改 sys.argv 将参数传给子模块（子模块有自己的 parse_args）
        sys.argv = [sys.argv[0],
                    "--config", args.config,       # 传入配置文件路径
                    "--results_dir", args.results_dir]  # 传入结果目录
        run_all_main()   # 调用 run_all_baselines 的 main 函数

    elif args.mode == "ablation":
        # 消融实验：逐一移除 M4Fed 的各关键组件，验证每个组件的贡献
        from experiments.ablation import run_ablation
        task_specs = {
            "age":           {"type": "regression",     "n_classes": 1},
            "sex":           {"type": "classification", "n_classes": 2},
            "health_status": {"type": "classification", "n_classes": 2},  # run_ablation 内更新 n_classes
            "race":          {"type": "classification", "n_classes": 5},  # run_ablation 内更新 n_classes
        }
        run_ablation(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)

    elif args.mode == "sensitivity":
        # 超参数敏感性分析：扫描合成比例/MedMD 学习率/Non-IID 程度
        from experiments.sensitivity import run_sensitivity
        task_specs = {
            "age":           {"type": "regression",     "n_classes": 1},
            "sex":           {"type": "classification", "n_classes": 2},
            "health_status": {"type": "classification", "n_classes": 2},  # run_sensitivity 内更新 n_classes
            "race":          {"type": "classification", "n_classes": 5},  # run_sensitivity 内更新 n_classes
        }
        run_sensitivity(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)

    elif args.mode == "cross_site":
        # 跨站点泛化实验：在未见过的采集中心数据上评估服务器端领域模型（SwinMultiTask）
        from experiments.cross_site import run_cross_site
        from data.openmind_dataset import build_node_datasets, OpenMindDataset, make_openmind_sample_loader
        from utils.io_utils import save_csv, ensure_dir

        sample_loader_fn = make_openmind_sample_loader(
            cfg.data.hf_dataset_id,                  # HuggingFace 数据集仓库 ID
            cache_dir=cfg.data.cache_dir or None,    # 本地缓存目录
            max_cache_gb=cfg.data.max_cache_gb,      # LRU 缓存容量上限（GB）
        )
        # 各 ds_id 的 age/sex/health_status 缺失率统计，用于挑选 cross 数据集时二次把关
        # （缺失率过高的 ds_id 会被跳过），避免偏差分析热图因样本量过少而不可靠
        quality_stats = _load_ds_quality_stats(cfg)
        # 构建完整节点数据集（服务器 + 2 客户端 + 跨站点评估集）
        nd, split_ids = build_node_datasets(
            all_ds_ids=all_ds_ids,                              # 所有数据集 ID
            n_cross=cfg.data.n_cross_ds,                         # 跨站点数据集数量
            n_private_per_node=cfg.data.n_private_ds_per_node,  # 每节点私有数据集数量
            n_clients=2,                                         # 客户端数量
            train_split=cfg.data.train_split,                    # 训练/测试比例
            mri_shape=mri_shape,                                 # MRI 目标分辨率
            seed=cfg.data.seed,                                  # 数据划分随机种子
            sample_loader_fn=sample_loader_fn,                   # 样本加载函数
            quality_stats=quality_stats,                         # ds_id → {字段: 缺失率}，None 表示不过滤
            max_missing_ratio=cfg.data.cross_max_missing_ratio,
            n_public_max=cfg.data.n_public_ds_max,               # server_public 数据集数量上限
            return_split_ids=True,
        )
        # nd["cross"]["all"] 包含所有跨站点子数据集的样本（不分 train/test，全部用于评估）
        cross_ds = nd["cross"]["all"]

        # 报告 cross 实际使用了哪些 ds_id，便于核对偏差分析的数据来源
        cross_report_dir = ensure_dir(os.path.join(args.results_dir, "cross_site"))
        save_csv({"ds_id": split_ids["cross"]}, os.path.join(cross_report_dir, "cross_ds_ids.csv"))
        print(f"[main] cross 数据集（{len(split_ids['cross'])} 个 ds_id）已记录至 "
              f"{os.path.join(cross_report_dir, 'cross_ds_ids.csv')}")

        model_path = os.path.join(args.results_dir, "m4fed", "server_swin.pt")
        run_cross_site(args, cfg, cross_ds, args.results_dir, model_path,
                       sample_loader_fn=sample_loader_fn)

    # ------------------------------------------------------------------
    # 保存全局运行耗时分布（网络下载/各阶段训练/评估等）
    # 注：test 模式在上面已通过 sys.exit() 提前退出，不会执行到这里
    # ------------------------------------------------------------------
    from utils.timing_utils import save_timing_csv
    save_timing_csv(os.path.join(args.results_dir, "timing_breakdown.csv"))

    # 保存被判定为不可用而排除的样本清单（重新下载后仍损坏），用于统计真实数据可用率
    from utils.data_quality_utils import save_unusable_csv
    save_unusable_csv(os.path.join(args.results_dir, "unusable_samples.csv"))


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


def _load_ds_quality_stats(cfg):
    """
    获取各 ds_id 的 age/sex/health_status 缺失率统计，供 cross 数据集质量筛选使用
    （build_node_datasets 的 quality_stats 参数）。优先级与 _load_ds_ids 一致，
    保证统计的 ds_id 集合与 all_ds_ids 口径对齐。

    返回：
      {ds_id: {field: missing_ratio}} 或 None（无法计算时，退化为不做质量过滤）
    """
    local_csv = cfg.data.local_metadata_csv
    if local_csv and os.path.exists(local_csv):
        from data.openmind_dataset import compute_ds_quality_stats
        return compute_ds_quality_stats(local_csv)
    try:
        from data.openmind_dataset import _load_openmind_metadata, _missing_ratio_by_ds_id
        df = _load_openmind_metadata(cfg.data.hf_dataset_id, cfg.data.cache_dir or None, "T1w")
        return _missing_ratio_by_ds_id(df, ("age", "sex", "health_status"))
    except Exception as e:
        print(f"[main] 无法计算 cross 数据集质量统计: {e}，跳过质量过滤。")
        return None


if __name__ == "__main__":
    main()   # 程序入口（直接运行 main.py 时调用）
