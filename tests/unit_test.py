"""
冒烟测试（Smoke Tests）。

运行方式：python tests/unit_test.py                          # 全部 4 个基础 suite（不含 visualization）
         python tests/unit_test.py --suite baseline          # 仅 5 个联邦基线方法
         python tests/unit_test.py --suite ablation          # 仅消融研究
         python tests/unit_test.py --suite sensitivity       # 仅敏感性分析
         python tests/unit_test.py --suite cross_site        # 仅跨站点泛化
         python tests/unit_test.py --suite visualization     # 仅 FAN/M4Fed 生成样本可视化
         python tests/unit_test.py --suite baseline --results_dir results/test_baseline
         python main.py --mode test

与 main.py 实验范围一致，包含 5 个 suite、共 9 个冒烟测试：
  - baseline     : 5 个联邦基线方法（FedAvg/FedProx/FAN/M4Fed/GPT-FL）端对端完整流程，
                   M4Fed 需先于 GPT-FL 运行以提供预训练 MedMD
  - ablation     : 消融研究（4 个变体，各自运行完整 M4Fed）
  - sensitivity  : 敏感性分析（9 个超参数扫描点）
  - cross_site   : 跨站点泛化实验（单次 SwinMultiTask 推理评估）。会依次尝试从本次
                   {results_dir}/m4fed/server_swin.pt 与 results/test_baseline/m4fed/
                   server_swin.pt 加载 baseline suite 训练好的权重，都找不到才退化为
                   预训练骨干权重基线对照
  - visualization: FAN/M4Fed 生成样本可视化，独立 suite。依赖 baseline suite 训练留下的
                   FAN/M4Fed 权重（{results_dir}/fan/client_0_medmd.pt、
                   {results_dir}/m4fed/server_medmd.pt），需先跑完 baseline suite
                   （--results_dir 与本次一致）再运行本 suite

--suite all 只运行前 4 个基础 suite（8 个冒烟测试），不含 visualization——
后者依赖 baseline 训练出的权重，需单独运行。

所有测试使用真实 OpenMind 数据，MRI 裁剪为 32×32×32 加速推理。
训练轮次均为 2（warm / federated / local rounds），目的是快速验证代码可运行
而非结果正确性。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse

from utils.env_utils import load_local_env   # 从 .env.local 加载本地密钥（如 TABPFN_TOKEN）
load_local_env()   # 必须在其他模块导入前执行（本文件可独立于 main.py 运行）

import traceback
import torch

from utils.seed_utils import set_seed
set_seed(42)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MRI_SHAPE = (32, 32, 32)

# 结果保存目录：默认 "results/test"，可通过 --results_dir 覆盖（如按 suite 拆分作业时
# 各自使用独立目录，避免并行提交的多个 SLURM 作业写入同一份 run.log 导致内容交叉）。
RESULTS_DIR = "results/test"

# baseline suite 独立跑（--results_dir results/test_baseline，见文件头部用法说明）时的
# 默认结果目录。cross_site suite 通常单独提交作业、使用自己的 --results_dir，
# 因此其结果目录下不会有 m4fed/server_swin.pt；退回到这里查找 baseline 训练好的
# SwinMultiTask 权重，避免冒烟测试退化为"预训练骨干权重基线对照"。
BASELINE_RESULTS_DIR = "results/test_baseline"


class _Args:
    """模拟 argparse.Namespace，供实验函数使用（真实数据模式）。"""


def _smoke_cfg():
    from configs.base_config import (
        M4FedConfig, DataConfig, ModelConfig, FederatedConfig,
        TrainingConfig, EvalConfig,
    )
    cfg = M4FedConfig()
    cfg.data = DataConfig(
        mri_shape=list(MRI_SHAPE),
        n_cross_ds=1,
        n_private_ds_per_node=2,
        train_split=0.75,
        seed=42,
    )
    cfg.model = ModelConfig(
        unet_in_channels=8,
        unet_out_channels=8,
        unet_num_channels=[8, 8],
        unet_attention_levels=[False, True],
        unet_norm_num_groups=4,
        unet_cross_attention_dim=64,
        diffusion_timesteps=10,
        ddim_steps=3,
        noise_strength_fed=0.75,
        noise_strength_local_min=0.35,
        noise_strength_local_max=0.70,
        vae_scale_factor=1.0,
    )
    cfg.federated = FederatedConfig(
        warm_rounds=2,
        federated_rounds=2,
        local_rounds=2,
        synth_ratio_fed=0.5,
        synth_ratio_local=0.5,
        qlora_r=2,
        qlora_alpha=4,
    )
    cfg.training = TrainingConfig(
        batch_size=1,
        lr_medmd=1e-3,
        lr_swin=1e-3,
        mixed_precision=False,
        gradient_checkpointing=False,
        device=DEVICE,
        num_workers=0,
        seed=42,
    )
    cfg.eval = EvalConfig(
        save_model=False,
        results_dir=RESULTS_DIR,
    )
    return cfg


def _load_real_ids(cfg):
    """从本地 openneuro_metadata.csv 读取真实子数据集 ID 列表。"""
    from data.openmind_dataset import list_ds_ids_from_metadata
    csv_path = cfg.data.local_metadata_csv
    if not (csv_path and os.path.exists(csv_path)):
        raise FileNotFoundError(
            f"冒烟测试需要本地元数据 CSV：{csv_path!r} 不存在，"
            "请先下载 openneuro_metadata.csv 到项目根目录。"
        )
    return list_ds_ids_from_metadata(csv_path)


def _load_real_ids_health_first(cfg):
    """
    读取真实子数据集 ID 列表，并把 health_status 有标注的 ds_id 排到最前面。

    test_ablation 需要把 ID 池截断到很小的数量（见其内部注释），若直接取
    排序后的前 N 个（_load_real_ids 的顺序），可能整批落在没有 health_status
    标注的子数据集上——ds000001~ds000011 这批就恰好如此，health_status 全部
    缺失，导致变体 D（只用健康人训练）的训练集过滤和健康状态子集评估都在空
    样本上运行，产出灾难性/全 NaN 的结果。这里用 compute_ds_quality_stats
    统计各 ds_id 的 health_status 缺失率，把缺失率低的排到前面，保证截断后
    的池子里确实包含健康/患病标注。
    """
    from data.openmind_dataset import list_ds_ids_from_metadata, compute_ds_quality_stats
    csv_path = cfg.data.local_metadata_csv
    if not (csv_path and os.path.exists(csv_path)):
        raise FileNotFoundError(
            f"冒烟测试需要本地元数据 CSV：{csv_path!r} 不存在，"
            "请先下载 openneuro_metadata.csv 到项目根目录。"
        )
    ids = list_ds_ids_from_metadata(csv_path)
    quality = compute_ds_quality_stats(csv_path, fields=("health_status",))
    ids.sort(key=lambda d: quality.get(d, {}).get("health_status", 1.0))  # 缺失率低的排前面（稳定排序，同缺失率维持原有字典序）
    return ids


def _build_datasets(cfg, n_clients=2):
    """使用真实 OpenMind 数据（MRI 裁剪为 32³）构建节点数据集，同时返回 sample_fn（含 label_maps）。"""
    from data.openmind_dataset import build_node_datasets, make_openmind_sample_loader
    all_ids = _load_real_ids(cfg)
    # 截断可用 ID 池，使 assign_ds_splits 分配后剩余的 public（即 server_public）
    # 数量正好等于 n_private_ds_per_node，与 server_private 数量保持一致，
    # 避免冒烟测试里 server_public 远大于其余角色导致规模失衡。
    n_needed = cfg.data.n_cross_ds + cfg.data.n_private_ds_per_node * (2 + n_clients)
    all_ids = all_ids[:n_needed]
    sample_fn = make_openmind_sample_loader(
        repo_id=cfg.data.hf_dataset_id,
        cache_dir=cfg.data.cache_dir or None,
        max_cache_gb=cfg.data.max_cache_gb,
    )
    nd = build_node_datasets(
        all_ds_ids=all_ids,
        n_cross=cfg.data.n_cross_ds,
        n_private_per_node=cfg.data.n_private_ds_per_node,
        n_clients=n_clients,
        train_split=cfg.data.train_split,
        mri_shape=MRI_SHAPE,
        seed=cfg.data.seed,
        sample_loader_fn=sample_fn,
    )
    return nd, sample_fn


def _build_task_specs(sample_fn):
    """从 sample_fn.n_classes 构建完整任务规格字典（含正确的 n_classes）。"""
    nc = getattr(sample_fn, "n_classes", {})
    return {
        "age":           {"type": "regression",     "n_classes": 1},
        "sex":           {"type": "classification", "n_classes": 2},
        "health_status": {"type": "classification", "n_classes": max(2, nc.get("health_status", 2))},
        "race":          {"type": "classification", "n_classes": max(2, nc.get("race", 5))},
    }


# ============================================================= #
#  1. 基线方法端对端冒烟测试                                         #
# ============================================================= #

def test_fed_fedavg():
    """FedAvg 完整联邦学习流程（3 客户端，2 轮）。"""
    print("SMOKE: FedAvg end-to-end (3 clients, 2 rounds) ... ", end="")
    cfg = _smoke_cfg()
    nd, sample_fn = _build_datasets(cfg, n_clients=3)
    from federated.server import Server
    from federated.client import Client
    from federated.methods.fedavg import run_fedavg
    server  = Server(cfg, nd["server_private"], nd["server_public"], DEVICE)
    clients = [Client(i, cfg, nd[f"client_{i}"], DEVICE) for i in range(3)]
    task_specs = _build_task_specs(sample_fn)
    res = run_fedavg(server, clients, cfg, task_specs, RESULTS_DIR)
    assert "clients_mean" in res
    print("PASS")


def test_fed_fedprox():
    """FedProx 完整联邦学习流程（3 客户端，2 轮）。"""
    print("SMOKE: FedProx end-to-end (3 clients, 2 rounds) ... ", end="")
    cfg = _smoke_cfg()
    nd, sample_fn = _build_datasets(cfg, n_clients=3)
    from federated.server import Server
    from federated.client import Client
    from federated.methods.fedprox import run_fedprox
    server  = Server(cfg, nd["server_private"], nd["server_public"], DEVICE)
    clients = [Client(i, cfg, nd[f"client_{i}"], DEVICE) for i in range(3)]
    task_specs = _build_task_specs(sample_fn)
    res = run_fedprox(server, clients, cfg, task_specs, RESULTS_DIR)
    assert "clients_mean" in res
    print("PASS")


def test_fed_fan():
    """FAN 完整联邦学习流程（3 客户端，2 轮）。

    开启 save_model，使 client_0 的 MedMD 权重（client_0_medmd.pt）落盘，
    供末尾的可视化功能加载生成随机 MRI 样本。
    """
    print("SMOKE: FAN end-to-end (3 clients, 2 rounds) ... ", end="")
    cfg = _smoke_cfg()
    cfg.eval.save_model = True
    nd, sample_fn = _build_datasets(cfg, n_clients=3)
    from federated.server import Server
    from federated.client import Client
    from federated.methods.fan import run_fan
    server  = Server(cfg, nd["server_private"], nd["server_public"], DEVICE)
    clients = [Client(i, cfg, nd[f"client_{i}"], DEVICE) for i in range(3)]
    task_specs = _build_task_specs(sample_fn)
    res = run_fan(server, clients, cfg, task_specs, RESULTS_DIR)
    assert "clients_mean" in res
    print("PASS")


def test_fed_m4fed():
    """M4Fed 三阶段完整联邦学习流程（2 客户端，2 轮）。

    先于 GPT-FL 运行并开启 save_model，使服务器热身阶段训练好的 MedMD
    权重（server_medmd.pt）落盘，供 test_fed_gptfl 复用为生成模型。
    """
    print("SMOKE: M4Fed end-to-end (2 clients, 2 rounds) ... ", end="")
    cfg = _smoke_cfg()
    cfg.eval.save_model = True
    nd, sample_fn = _build_datasets(cfg, n_clients=2)
    from federated.server import Server
    from federated.client import Client
    from federated.methods.m4fed import run_m4fed
    server  = Server(cfg, nd["server_private"], nd["server_public"], DEVICE)
    clients = [Client(i, cfg, nd[f"client_{i}"], DEVICE) for i in range(2)]
    task_specs = _build_task_specs(sample_fn)
    res = run_m4fed(server, clients, cfg, task_specs, RESULTS_DIR)
    assert "clients_mean" in res
    print("PASS")


def test_fed_gptfl():
    """GPT-FL 完整联邦学习流程（3 客户端，2 轮）。

    复用 test_fed_m4fed 热身阶段保存的服务器 MedMD 权重作为生成模型
    （与 experiments/run_all_baselines.py 中 m4fed → gptfl 的接力方式一致），
    因此必须在 test_fed_m4fed 之后运行。
    """
    print("SMOKE: GPT-FL end-to-end (3 clients, 2 rounds) ... ", end="")
    cfg = _smoke_cfg()
    nd, sample_fn = _build_datasets(cfg, n_clients=3)
    from federated.server import Server
    from federated.client import Client
    from federated.methods.gptfl import run_gptfl
    server  = Server(cfg, nd["server_private"], nd["server_public"], DEVICE)
    clients = [Client(i, cfg, nd[f"client_{i}"], DEVICE) for i in range(3)]
    task_specs = _build_task_specs(sample_fn)
    m4fed_medmd_path = os.path.join(RESULTS_DIR, "m4fed", "server_medmd.pt")
    res = run_gptfl(server, clients, cfg, task_specs, RESULTS_DIR,
                    pretrained_medmd_path=m4fed_medmd_path)
    assert "clients_mean" in res
    print("PASS")


# ============================================================= #
#  2. 消融研究冒烟测试                                              #
# ============================================================= #

def test_ablation():
    """消融实验：4 个变体（无公开数据 / 无 TabPFN / 随机初始化 SwinUNETR / 训练数据仅用健康人）。"""
    print("SMOKE: Ablation study (4 variants) ... ", end="")
    cfg = _smoke_cfg()
    # 用 health_status 覆盖率优先排序的 ID 池（而非 _load_real_ids 的原始字典序），
    # 保证下面截断后变体 D（只用健康人训练）能筛出非空训练集，健康状态子集评估
    # 也能拆出非空的 ill_only/healthy_only 子集，而不是全部落在无标注的数据集上
    all_ids = _load_real_ids_health_first(cfg)
    # 截断可用 ID 池，使 assign_ds_splits 分配后剩余的 public（即 server_public）
    # 数量正好等于 n_private_ds_per_node（与 _build_datasets 的截断逻辑一致），
    # 否则 run_ablation 内部的 build_node_datasets 会把全部剩余数据集（约 300 个）
    # 都分给 server_public，导致变体 B/C/D 热身阶段需要下载整个 OpenMind 语料库。
    n_needed = cfg.data.n_cross_ds + cfg.data.n_private_ds_per_node * (2 + 2)  # 消融实验固定 2 客户端
    all_ids = all_ids[:n_needed]
    from experiments.ablation import run_ablation
    task_specs = {
        "age":           {"type": "regression",     "n_classes": 1},
        "sex":           {"type": "classification", "n_classes": 2},
        "health_status": {"type": "classification", "n_classes": 2},
        "race":          {"type": "classification", "n_classes": 5},
    }
    res = run_ablation(_Args(), cfg, all_ids, MRI_SHAPE, task_specs, RESULTS_DIR)
    assert len(res) == 4
    assert "health_subset_eval" in res["only_healthy_train"]
    print("PASS")


# ============================================================= #
#  3. 敏感性分析冒烟测试                                            #
# ============================================================= #

def test_sensitivity():
    """敏感性分析：扫描合成比例 / MedMD 学习率 / Dirichlet α，共 9 个配置点。"""
    print("SMOKE: Sensitivity analysis (9 configs) ... ", end="")
    cfg = _smoke_cfg()
    all_ids = _load_real_ids(cfg)
    # 截断可用 ID 池，原因同 test_ablation（避免 server_public 吃下整个语料库）
    n_needed = cfg.data.n_cross_ds + cfg.data.n_private_ds_per_node * (2 + 2)  # 敏感性分析固定 2 客户端
    all_ids = all_ids[:n_needed]
    from experiments.sensitivity import run_sensitivity
    task_specs = {
        "age":           {"type": "regression",     "n_classes": 1},
        "sex":           {"type": "classification", "n_classes": 2},
        "health_status": {"type": "classification", "n_classes": 2},
        "race":          {"type": "classification", "n_classes": 5},
    }
    res = run_sensitivity(_Args(), cfg, all_ids, MRI_SHAPE, task_specs, RESULTS_DIR)
    assert len(res) == 9
    print("PASS")


# ============================================================= #
#  4. 跨站点泛化实验冒烟测试                                         #
# ============================================================= #

def _find_baseline_swin_path():
    """查找 baseline 冒烟测试（test_fed_m4fed）训练留下的 server_swin.pt。

    依次尝试当前 RESULTS_DIR 与 BASELINE_RESULTS_DIR 下的 m4fed/server_swin.pt，
    都不存在则返回 None（run_cross_site 会退化为预训练骨干权重基线对照）。
    """
    for results_dir in (RESULTS_DIR, BASELINE_RESULTS_DIR):
        candidate = os.path.join(results_dir, "m4fed", "server_swin.pt")
        if os.path.exists(candidate):
            return candidate
    return None


def test_cross_site():
    """跨站点泛化：在真实跨站点数据上评估 SwinMultiTask 端对端推理流程。

    优先复用 baseline suite（test_fed_m4fed）训练好的 server_swin.pt 权重，
    找不到时才退化为预训练骨干权重基线对照。
    """
    print("SMOKE: Cross-site generalisation ... ", end="")
    cfg = _smoke_cfg()
    nd, sample_fn = _build_datasets(cfg, n_clients=2)
    cross_ds = nd["cross"]["all"]
    from experiments.cross_site import run_cross_site
    model_path = _find_baseline_swin_path()
    res = run_cross_site(_Args(), cfg, cross_ds, RESULTS_DIR, model_path=model_path)
    assert "metrics" in res
    print("PASS")


# ============================================================= #
#  5. 生成样本可视化（FAN / M4Fed）                                  #
# ============================================================= #

def visualize_generated_samples():
    """
    加载 FAN（client_0）与 M4Fed（server）baseline suite 中训练完成的生成模型，
    各随机生成一个 3D MRI 并保存 2D 切片到 {results_dir}/visualization/。

    独立 suite，依赖 baseline suite 训练留下的权重，需先跑完 baseline
    （--results_dir 与本次一致）再运行本函数/suite。
    """
    print("SMOKE: FAN / M4Fed 生成样本可视化 ... ", end="")
    from evaluation.generative_eval import visualize_fan_and_m4fed_samples
    cfg = _smoke_cfg()
    visualize_fan_and_m4fed_samples(cfg, RESULTS_DIR, DEVICE, mri_shape=MRI_SHAPE)
    print("PASS")


BASELINE_TESTS = [
    test_fed_fedavg,
    test_fed_fedprox,
    test_fed_fan,
    test_fed_m4fed,   # 需先于 test_fed_gptfl 运行（提供预训练 MedMD）
    test_fed_gptfl,
]
ABLATION_TESTS     = [test_ablation]
SENSITIVITY_TESTS  = [test_sensitivity]
CROSS_SITE_TESTS   = [test_cross_site]
VISUALIZATION_TESTS = [visualize_generated_samples]

# suite 名称 -> 测试列表，与 main.py 的 --mode 命名风格保持一致
SUITES = {
    "baseline":      BASELINE_TESTS,
    "ablation":      ABLATION_TESTS,
    "sensitivity":   SENSITIVITY_TESTS,
    "cross_site":    CROSS_SITE_TESTS,
    "visualization": VISUALIZATION_TESTS,
}
# "all" 只汇总前 4 个基础 suite，不含 visualization——
# 后者依赖 baseline 训练出的权重，需单独运行（见文件头部说明）。
SUITES["all"] = BASELINE_TESTS + ABLATION_TESTS + SENSITIVITY_TESTS + CROSS_SITE_TESTS


def parse_args():
    """解析命令行参数，返回 Namespace 对象（风格与 main.py 的 parse_args 保持一致）。

    可用参数：
      --suite      : 要运行的冒烟测试子集（all/baseline/ablation/sensitivity/cross_site/visualization）
      --results_dir: 结果保存目录（自动创建），默认 "results/test"
    """
    p = argparse.ArgumentParser(description="M4Fed 冒烟测试")
    p.add_argument("--suite", choices=list(SUITES.keys()), default="all",
                    help="要运行的冒烟测试子集")
    p.add_argument("--results_dir", default="results/test",
                    help="结果保存目录")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    RESULTS_DIR = args.results_dir
    tests = SUITES[args.suite]

    # 终端输出（含各冒烟测试的 PASS/FAIL 及异常回溯）同步保存到 {RESULTS_DIR}/run.log，
    # 便于事后查看失败原因（终端显示效果不变）
    from utils.log_utils import setup_run_logger
    log_path = setup_run_logger(RESULTS_DIR)
    print(f"[unit_test] suite={args.suite}，终端日志将同步保存至 {log_path}")

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {e})")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"结果：{passed} 通过，{failed} 失败，共 {len(tests)} 个冒烟测试。")

    if failed == 0:
        print("所有冒烟测试通过！")
    else:
        sys.exit(1)
