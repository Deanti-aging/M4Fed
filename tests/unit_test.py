"""
单元/冒烟测试（Unit / Smoke Tests）。

运行方式：python tests/unit_test.py

所有测试使用极小虚拟数据（MRI 尺寸 8×8×8，4 个样本）验证整个流水线
能够在不报错的情况下端对端运行。

包含 17 个测试函数，覆盖：
  - 数据预处理（test_preprocessing）
  - 表格文本化（test_table_to_text）
  - 数据集加载（test_dataset_loading）
  - VAE 编解码器（test_vae_wrapper）
  - BioMedCLIP 条件编码器（test_conditioning）
  - MedMD 前向传播（test_medmd_forward）
  - MedMD 合成推理（test_medmd_synthesise）
  - NeuroFM 前向传播（test_neurofm_forward）
  - NeuroFM 多任务损失（test_neurofm_loss）
  - TabPFN 缺失值填充（test_tabpfn_imputer）
  - MedMDTrainer 训练 epoch（test_medmd_trainer）
  - NeuroFMTrainer 训练+评估（test_neurofm_trainer）
  - 领域指标计算（test_metrics）
  - FedAvg 完整流程（test_fed_fedavg）
  - FedProx 完整流程（test_fed_fedprox）
  - M4Fed 完整流程（test_fed_m4fed）
  - 可视化函数（test_visualisation）
"""
from __future__ import annotations
import sys, os
# 将项目根目录加入路径（使测试脚本可以直接运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import pandas as pd
import traceback

from utils.seed_utils import set_seed
set_seed(42)   # 全局固定随机种子，确保测试结果可复现

# 测试全局常量（极小配置，确保测试在几秒内完成）
DEVICE = "cpu"
MRI_SHAPE = (8, 8, 8)   # 极小 MRI 体积（用于速度优先的冒烟测试）
N_SAMPLES = 4           # 每个虚拟数据集的样本数


def _dummy_cfg():
    """
    构建用于测试的极小配置对象。

    所有数值均设置为最小可运行值：
      - U-Net 通道数极少（8）
      - 扩散时间步仅 10 步
      - DDIM 推理仅 3 步
      - 联邦/本地训练仅 1 轮
      - 批量大小为 1
    """
    from configs.base_config import (
        M4FedConfig, DataConfig, ModelConfig, FederatedConfig,
        TrainingConfig, EvalConfig
    )
    cfg = M4FedConfig()
    cfg.data = DataConfig(
        mri_shape=list(MRI_SHAPE), n_cross_ds=1, n_private_ds_per_node=2,
        train_split=0.75, seed=42,
    )
    cfg.model = ModelConfig(
        unet_in_channels=2,
        unet_out_channels=2,
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
        warm_rounds=1, federated_rounds=1, local_rounds=1,
        synth_ratio_fed=0.5, synth_ratio_local=0.5,
        qlora_r=2, qlora_alpha=4,
    )
    cfg.training = TrainingConfig(
        batch_size=1, lr_medmd=1e-3, lr_neurofm=1e-3,
        mixed_precision=False, gradient_checkpointing=False,
        device=DEVICE, num_workers=0, seed=42,
    )
    cfg.eval = EvalConfig(save_model=False, results_dir="results/test")
    return cfg


def _dummy_samples(n=N_SAMPLES, mri_shape=MRI_SHAPE):
    """
    生成 n 个虚拟样本字典列表。

    每个样本格式：{'mri_array': ndarray, 'table_text': str, 'age': float, 'metadata': dict}
    """
    from data.openmind_dataset import _make_dummy_samples
    return _make_dummy_samples(n, mri_shape=mri_shape, seed=0)


def _dummy_datasets(cfg, n_clients=2):
    """
    构建用于测试的虚拟节点数据集（dummy_mode=True）。

    参数：
      cfg       : 测试配置（来自 _dummy_cfg()）
      n_clients : 客户端数量（FedAvg/FedProx/FAN 需要 3 个）
    """
    from data.openmind_dataset import build_node_datasets
    all_ids = [f"ds{i:04d}" for i in range(20)]
    return build_node_datasets(
        all_ds_ids=all_ids,
        n_cross=cfg.data.n_cross_ds,
        n_private_per_node=cfg.data.n_private_ds_per_node,
        n_clients=n_clients,
        train_split=cfg.data.train_split,
        mri_shape=MRI_SHAPE,
        seed=cfg.data.seed,
        dummy_mode=True,
        n_dummy_per_ds=3,
    )


# ============================================================= #
#  测试函数                                                       #
# ============================================================= #

def test_preprocessing():
    """验证 MRI 体积预处理（裁剪/归一化/插值/缩放）和表格预处理。"""
    print("TEST: preprocessing ... ", end="")
    from data.preprocessing import preprocess_mri, preprocess_table
    # 生成随机体积并预处理
    vol = np.random.randn(*MRI_SHAPE).astype(np.float32)
    t = preprocess_mri(vol, MRI_SHAPE)
    # 验证输出形状（通道维度为 1）和值域（[-1.1, 1.1] 宽松容差）
    assert t.shape == (1, *MRI_SHAPE), f"形状不匹配: {t.shape}"
    assert t.min() >= -1.1 and t.max() <= 1.1
    # 验证表格预处理（含 NaN 的 DataFrame）
    df = pd.DataFrame({"age": [25.0, None, 30.0], "sex": ["M", "F", None]})
    df2 = preprocess_table(df)
    assert "age" in df2.columns
    print("PASS")


def test_table_to_text():
    """验证元数据行转文本模板（table_row_to_text）。"""
    print("TEST: table_to_text ... ", end="")
    from data.table_to_text import table_row_to_text
    row = pd.Series({"age": 32, "sex": "female", "diagnosis": "healthy"})
    text = table_row_to_text(row)
    # 验证年龄字段出现在生成文本中
    assert "age is 32" in text
    print("PASS")


def test_dataset_loading():
    """验证 build_node_datasets 能正确构建节点数据集字典并返回有效样本。"""
    print("TEST: dataset loading ... ", end="")
    cfg = _dummy_cfg()
    nd = _dummy_datasets(cfg)
    assert "server_private" in nd
    assert "server_public" in nd
    assert "client_0" in nd
    tr = nd["server_private"]["train"]
    assert len(tr) > 0
    # 验证样本格式（包含 'mri' 键，通道维度为 1）
    batch = tr[0]
    assert "mri" in batch
    assert batch["mri"].shape[0] == 1
    print("PASS")


def test_vae_wrapper():
    """验证 VAE 封装器能正常加载并对 3D MRI 体积执行 encode。"""
    print("TEST: VAE wrapper ... ", end="")
    from models.medmd.vae_wrapper import load_vae
    vae = load_vae(device=DEVICE, latent_channels=2)
    x = torch.randn(1, 1, *MRI_SHAPE)
    z = vae.encode(x)
    # encode 可能返回张量或 (张量, log_var) 元组
    if isinstance(z, tuple):
        z = z[0]
    assert z.shape[0] == 1   # 批量维度不变
    print("PASS")


def test_conditioning():
    """验证 BioMedCLIP 文本编码器能处理文本列表并返回三维嵌入。"""
    print("TEST: BioMedCLIP conditioning ... ", end="")
    from models.medmd.conditioning import BioMedCLIPEncoder
    enc = BioMedCLIPEncoder(device=DEVICE)
    out = enc(["age is 30, sex is male"])
    # 期望输出形状：(batch, seq_len, hidden_dim)
    assert out.ndim == 3
    print("PASS")


def test_medmd_forward():
    """验证 MedMD 前向传播（扩散损失）能返回有限值损失。"""
    print("TEST: MedMD forward (train loss) ... ", end="")
    cfg = _dummy_cfg()
    from models.medmd.medmd import MedMD
    model = MedMD(cfg.model, device=DEVICE)
    x = torch.randn(1, 1, *MRI_SHAPE)
    loss = model(x, ["age is 25"])
    assert torch.isfinite(loss), f"损失为非有限值: {loss}"
    print(f"PASS (loss={loss.item():.4f})")


def test_medmd_synthesise():
    """验证 MedMD img2img 合成推理（DDIM 采样）的输出形状正确。"""
    print("TEST: MedMD synthesise (img2img) ... ", end="")
    cfg = _dummy_cfg()
    from models.medmd.medmd import MedMD
    model = MedMD(cfg.model, device=DEVICE)
    x = torch.randn(1, 1, *MRI_SHAPE)
    with torch.no_grad():
        syn = model.synthesise(x, ["age is 40"], noise_strength=0.5, ddim_steps=2)
    assert syn.shape == (1, 1, *MRI_SHAPE), f"形状不匹配: {syn.shape}"
    print("PASS")


def test_neurofm_forward():
    """验证 NeuroFM 前向传播（随机初始化）能返回年龄预测张量。"""
    print("TEST: NeuroFM forward ... ", end="")
    from models.neurofm.neurofm_multitask import NeuroFMMultiTask
    task_specs = {"age": {"type": "regression", "n_classes": 1}}
    # random_init=True：跳过 HuggingFace 权重下载，使用随机骨干
    model = NeuroFMMultiTask(task_specs=task_specs, device=DEVICE, random_init=True)
    x = torch.randn(2, 1, *MRI_SHAPE)
    preds = model(x)
    assert "age" in preds
    assert preds["age"].shape[0] == 2   # 批量维度正确
    print("PASS")


def test_neurofm_loss():
    """验证 NeuroFM 多任务损失计算能返回有限值损失。"""
    print("TEST: NeuroFM multi-task loss ... ", end="")
    from models.neurofm.neurofm_multitask import NeuroFMMultiTask
    task_specs = {"age": {"type": "regression", "n_classes": 1}}
    model = NeuroFMMultiTask(task_specs=task_specs, device=DEVICE, random_init=True)
    x = torch.randn(2, 1, *MRI_SHAPE)
    preds = model(x)
    targets = {"age": torch.tensor([25.0, 45.0])}
    loss = model.compute_loss(preds, targets)
    assert torch.isfinite(loss)
    print(f"PASS (loss={loss.item():.4f})")


def test_tabpfn_imputer():
    """验证 TabPFNImputer 能填充 NaN（回退到 median/mode 策略时也应通过）。"""
    print("TEST: TabPFN imputer (fallback median) ... ", end="")
    from models.tabpfn_wrapper import TabPFNImputer
    df = pd.DataFrame({
        "age": [25.0, None, 30.0, None, 22.0],
        "sex": ["M", "F", None, "M", "F"],
    })
    imp = TabPFNImputer(device=DEVICE)
    df2 = imp.fit_transform(df)
    # 验证 age 列中没有剩余 NaN
    assert df2["age"].isna().sum() == 0, "age 列仍有 NaN 未填充"
    print("PASS")


def test_medmd_trainer():
    """验证 MedMDTrainer.train_epoch 能完成一个 epoch 并返回有限损失。"""
    print("TEST: MedMDTrainer.train_epoch ... ", end="")
    from data.openmind_dataset import OpenMindDataset, make_dataloader
    from models.medmd.medmd import MedMD
    from training.medmd_trainer import MedMDTrainer
    cfg = _dummy_cfg()
    samples = _dummy_samples()
    ds = OpenMindDataset(samples, MRI_SHAPE)
    loader = make_dataloader(ds, batch_size=1, shuffle=False)
    model = MedMD(cfg.model, device=DEVICE)
    trainer = MedMDTrainer(model, device=DEVICE, lr=1e-3, mixed_precision=False)
    loss = trainer.train_epoch(loader)
    assert np.isfinite(loss), f"损失为非有限值: {loss}"
    print(f"PASS (loss={loss:.4f})")


def test_neurofm_trainer():
    """验证 NeuroFMTrainer 能完成训练 epoch 和推理评估。"""
    print("TEST: NeuroFMTrainer.train_epoch + evaluate ... ", end="")
    from data.openmind_dataset import OpenMindDataset, make_dataloader
    from models.neurofm.neurofm_multitask import NeuroFMMultiTask
    from training.neurofm_trainer import NeuroFMTrainer
    samples = _dummy_samples()
    ds = OpenMindDataset(samples, MRI_SHAPE)
    loader = make_dataloader(ds, batch_size=1, shuffle=False)
    task_specs = {"age": {"type": "regression", "n_classes": 1}}
    model = NeuroFMMultiTask(task_specs=task_specs, device=DEVICE, random_init=True)
    trainer = NeuroFMTrainer(model, device=DEVICE, lr=1e-3, mixed_precision=False)
    loss = trainer.train_epoch(loader)
    y_true, y_pred = trainer.evaluate(loader)
    assert len(y_true) > 0
    print(f"PASS (loss={loss:.4f}, n_eval={len(y_true)})")


def test_metrics():
    """验证 compute_domain_metrics 能计算出包含有限值 MAE 的指标字典。"""
    print("TEST: domain metrics ... ", end="")
    from evaluation.metrics import compute_domain_metrics
    y_true = np.array([25.0, 30.0, 40.0, 50.0])
    y_pred = np.array([26.0, 31.0, 38.0, 52.0])
    m = compute_domain_metrics(y_true, y_pred)
    assert "MAE" in m and np.isfinite(m["MAE"])
    print(f"PASS (MAE={m['MAE']:.2f})")


def test_fed_fedavg():
    """冒烟测试：FedAvg 完整流程（1 轮，虚拟数据，3 个客户端）。"""
    print("TEST: FedAvg (1 round, dummy) ... ", end="")
    cfg = _dummy_cfg()
    nd = _dummy_datasets(cfg, n_clients=3)
    from federated.server import Server
    from federated.client import Client
    server = Server(cfg, nd["server_private"], nd["server_public"], DEVICE)
    clients = [Client(i, cfg, nd[f"client_{i}"], DEVICE) for i in range(3)]
    from federated.methods.fedavg import run_fedavg
    task_specs = {"age": {"type": "regression", "n_classes": 1}}
    res = run_fedavg(server, clients, cfg, task_specs, "results/test")
    # 验证返回结果中包含客户端均值指标
    assert "clients_mean" in res
    print("PASS")


def test_fed_fedprox():
    """冒烟测试：FedProx 完整流程（1 轮，虚拟数据，3 个客户端）。"""
    print("TEST: FedProx (1 round, dummy) ... ", end="")
    cfg = _dummy_cfg()
    nd = _dummy_datasets(cfg, n_clients=3)
    from federated.server import Server
    from federated.client import Client
    server = Server(cfg, nd["server_private"], nd["server_public"], DEVICE)
    clients = [Client(i, cfg, nd[f"client_{i}"], DEVICE) for i in range(3)]
    from federated.methods.fedprox import run_fedprox
    task_specs = {"age": {"type": "regression", "n_classes": 1}}
    res = run_fedprox(server, clients, cfg, task_specs, "results/test")
    assert "clients_mean" in res
    print("PASS")


def test_fed_m4fed():
    """冒烟测试：M4Fed 三阶段完整流程（1 轮，虚拟数据，2 个客户端）。"""
    print("TEST: M4Fed (1 round, dummy) ... ", end="")
    cfg = _dummy_cfg()
    nd = _dummy_datasets(cfg, n_clients=2)
    from federated.server import Server
    from federated.client import Client
    server = Server(cfg, nd["server_private"], nd["server_public"], DEVICE)
    clients = [Client(i, cfg, nd[f"client_{i}"], DEVICE) for i in range(2)]
    from federated.methods.m4fed import run_m4fed
    task_specs = {"age": {"type": "regression", "n_classes": 1}}
    res = run_m4fed(server, clients, cfg, task_specs, "results/test")
    assert "clients_mean" in res
    print("PASS")


def test_visualisation():
    """验证损失曲线图和指标柱状图能正常返回 Figure 对象（无报错）。"""
    print("TEST: visualisation (loss curves + bar charts) ... ", end="")
    from evaluation.visualization import plot_loss_curves, plot_metric_bars
    import matplotlib
    matplotlib.use("Agg")   # 无显示器环境（CI/服务器）必须使用非交互式后端
    losses = {
        "FedAvg": {"mean": [1.0, 0.8, 0.6], "std": [0.1, 0.1, 0.1]},
        "M4Fed": {"mean": [0.9, 0.7, 0.5], "std": [0.05, 0.05, 0.05]},
    }
    fig = plot_loss_curves(losses)
    assert fig is not None
    metric_means = {"FedAvg": {"MAE": 5.0}, "M4Fed": {"MAE": 3.5}}
    fig2 = plot_metric_bars(metric_means, metric_name="MAE")
    assert fig2 is not None
    print("PASS")


# ============================================================= #
#  测试注册列表（按依赖顺序排列）                                   #
# ============================================================= #
ALL_TESTS = [
    test_preprocessing,      # 数据预处理（无外部依赖）
    test_table_to_text,      # 文本化
    test_dataset_loading,    # 数据集构建
    test_vae_wrapper,        # VAE 模型
    test_conditioning,       # 文本编码器
    test_medmd_forward,      # 扩散模型前向
    test_medmd_synthesise,   # 扩散模型推理
    test_neurofm_forward,    # NeuroFM 前向
    test_neurofm_loss,       # NeuroFM 损失
    test_tabpfn_imputer,     # 表格填充
    test_medmd_trainer,      # 扩散训练器
    test_neurofm_trainer,    # NeuroFM 训练器
    test_metrics,            # 评估指标
    test_fed_fedavg,         # FedAvg 完整流程
    test_fed_fedprox,        # FedProx 完整流程
    test_fed_m4fed,          # M4Fed 完整流程（最重，放最后）
    test_visualisation,      # 可视化
]


if __name__ == "__main__":
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
    print(f"结果：{passed} 通过，{failed} 失败，共 {len(ALL_TESTS)} 个测试。")
    if failed == 0:
        print("所有测试通过！")
    else:
        sys.exit(1)
