"""
冒烟测试（Smoke Tests）。

运行方式：python tests/unit_test.py
         python main.py --mode test

与 main.py 实验范围一致，包含 8 个冒烟测试：
  - 5 个联邦基线方法（FedAvg/FedProx/FAN/M4Fed/GPT-FL）端对端完整流程，
    M4Fed 需先于 GPT-FL 运行以提供预训练 MedMD
  - 消融研究（4 个变体，各自运行完整 M4Fed）
  - 敏感性分析（9 个超参数扫描点）
  - 跨站点泛化实验（单次 SwinMultiTask 推理评估）

所有测试使用真实 OpenMind 数据，MRI 裁剪为 32×32×32 加速推理。
训练轮次均为 2（warm / federated / local rounds），目的是快速验证代码可运行
而非结果正确性。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.env_utils import load_local_env   # 从 .env.local 加载本地密钥（如 TABPFN_TOKEN）
load_local_env()   # 必须在其他模块导入前执行（本文件可独立于 main.py 运行）

import traceback
import torch

from utils.seed_utils import set_seed
set_seed(42)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MRI_SHAPE = (32, 32, 32)


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
        results_dir="results/test",
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


def _build_datasets(cfg, n_clients=2):
    """使用真实 OpenMind 数据（MRI 裁剪为 32³）构建节点数据集，同时返回 sample_fn（含 label_maps）。"""
    from data.openmind_dataset import build_node_datasets, make_openmind_sample_loader
    all_ids = _load_real_ids(cfg)
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
    res = run_fedavg(server, clients, cfg, task_specs, "results/test")
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
    res = run_fedprox(server, clients, cfg, task_specs, "results/test")
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
    res = run_fan(server, clients, cfg, task_specs, "results/test")
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
    res = run_m4fed(server, clients, cfg, task_specs, "results/test")
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
    m4fed_medmd_path = os.path.join("results/test", "m4fed", "server_medmd.pt")
    res = run_gptfl(server, clients, cfg, task_specs, "results/test",
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
    all_ids = _load_real_ids(cfg)
    from experiments.ablation import run_ablation
    task_specs = {
        "age":           {"type": "regression",     "n_classes": 1},
        "sex":           {"type": "classification", "n_classes": 2},
        "health_status": {"type": "classification", "n_classes": 2},
        "race":          {"type": "classification", "n_classes": 5},
    }
    res = run_ablation(_Args(), cfg, all_ids, MRI_SHAPE, task_specs, "results/test")
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
    from experiments.sensitivity import run_sensitivity
    task_specs = {
        "age":           {"type": "regression",     "n_classes": 1},
        "sex":           {"type": "classification", "n_classes": 2},
        "health_status": {"type": "classification", "n_classes": 2},
        "race":          {"type": "classification", "n_classes": 5},
    }
    res = run_sensitivity(_Args(), cfg, all_ids, MRI_SHAPE, task_specs, "results/test")
    assert len(res) == 9
    print("PASS")


# ============================================================= #
#  4. 跨站点泛化实验冒烟测试                                         #
# ============================================================= #

def test_cross_site():
    """跨站点泛化：在真实跨站点数据上评估 SwinMultiTask 端对端推理流程。"""
    print("SMOKE: Cross-site generalisation ... ", end="")
    cfg = _smoke_cfg()
    nd, sample_fn = _build_datasets(cfg, n_clients=2)
    cross_ds = nd["cross"]["all"]
    from experiments.cross_site import run_cross_site
    res = run_cross_site(_Args(), cfg, cross_ds, "results/test", model_path=None)
    assert "metrics" in res
    print("PASS")


ALL_TESTS = [
    test_fed_fedavg,
    test_fed_fedprox,
    test_fed_fan,
    test_fed_m4fed,
    test_fed_gptfl,
    test_ablation,
    test_sensitivity,
    test_cross_site,
]


# ============================================================= #
#  5. 生成样本可视化（FAN / M4Fed）                                  #
# ============================================================= #

def visualize_generated_samples():
    """
    加载 FAN（client_0）与 M4Fed（server）冒烟测试中训练完成的生成模型，
    各随机生成一个 3D MRI 并保存 2D 切片到 results/test/visualization/。
    """
    print("SMOKE: FAN / M4Fed 生成样本可视化 ... ", end="")
    from evaluation.generative_eval import visualize_fan_and_m4fed_samples
    cfg = _smoke_cfg()
    visualize_fan_and_m4fed_samples(cfg, "results/test", DEVICE, mri_shape=MRI_SHAPE)
    print("PASS")


if __name__ == "__main__":
    # 终端输出（含各冒烟测试的 PASS/FAIL 及异常回溯）同步保存到 results/test/run.log，
    # 便于事后查看失败原因（终端显示效果不变）
    from utils.log_utils import setup_run_logger
    log_path = setup_run_logger("results/test")
    print(f"[unit_test] 终端日志将同步保存至 {log_path}")

    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {e})")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"结果：{passed} 通过，{failed} 失败，共 {len(ALL_TESTS)} 个冒烟测试。")

    # 可视化功能不计入冒烟测试通过/失败统计（依赖前面测试留下的模型权重，
    # 权重缺失或生成异常不应影响冒烟测试本身的退出码）
    try:
        visualize_generated_samples()
    except Exception as e:
        print(f"FAIL ({type(e).__name__}: {e})")
        traceback.print_exc()

    if failed == 0:
        print("所有冒烟测试通过！")
    else:
        sys.exit(1)
