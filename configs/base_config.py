"""统一配置数据类，支持 YAML 文件覆盖默认值。"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml
import os

"""
@dataclass避免手写
class Config:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
"""
@dataclass
class DataConfig:
    """数据相关配置。"""
    # HuggingFace 上 OpenMind 数据集的仓库 ID
    hf_dataset_id: str = "AnonRes/OpenMind"
    # 本地缓存目录（留空则禁用本地缓存）
    cache_dir: str = ""
    # 跨站点子数据集数量（cross-site split 使用）
    n_cross_ds: int = 5
    # 每个节点（服务器 + 每个客户端）的私有子数据集数量
    n_private_ds_per_node: int = 20
    # 训练集占数据集的比例（剩余为测试集）
    train_split: float = 0.9
    # MRI 体积目标尺寸 (D, H, W)，单位为体素
    mri_shape: List[int] = field(default_factory=lambda: [96, 96, 96])
    # 数据集划分的随机种子
    seed: int = 42
    # 是否使用 HuggingFace 流式加载（不预下载全部数据）
    streaming: bool = True


@dataclass
class ModelConfig:
    """模型结构与预训练权重相关配置。"""
    # VAE（MONAI AutoencoderKL）的 HuggingFace 仓库 ID
    # MONAI/brats_mri_generative_diffusion：3D 脑 MRI VAE，spatial_dims=3，latent_channels=8
    vae_repo: str = "MONAI/brats_mri_generative_diffusion"
    # U-Net 的空间维度（3 表示 3D）
    unet_spatial_dims: int = 3
    # U-Net 输入通道数（对应 VAE 潜空间通道数，须与 vae_repo 的 latent_channels 一致）
    unet_in_channels: int = 8
    # U-Net 输出通道数（与输入通道数相同，预测噪声）
    unet_out_channels: int = 8
    # 每个分辨率级别的残差块数量
    unet_num_res_blocks: int = 2
    # 各分辨率级别的特征通道数列表（从粗到细）
    unet_num_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 64])
    # 各分辨率级别是否启用注意力机制
    unet_attention_levels: List[bool] = field(default_factory=lambda: [False, False, True, True])
    # 组归一化的分组数
    unet_norm_num_groups: int = 32
    # 交叉注意力维度（文本嵌入维度，与 BioMedCLIP 对齐）
    unet_cross_attention_dim: int = 512
    # BioMedCLIP 文本编码器的 HuggingFace 仓库 ID
    biomedclip_repo: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # NeuroFM 骨干网络的 HuggingFace 仓库 ID
    neurofm_repo: str = "NeuroAI-UofG/NeuroFM"
    # TabPFN 的 HuggingFace 仓库 ID（用于表格缺失值填充）
    tabpfn_repo: str = "Prior-Labs/tabpfn_2_5"
    # 扩散过程的总时间步数 T
    diffusion_timesteps: int = 1000
    # DDIM 采样步数（推理时使用，远小于 T 以加速）
    ddim_steps: int = 50
    # 线性 beta 调度的起始值
    beta_start: float = 1e-4
    # 线性 beta 调度的结束值
    beta_end: float = 2e-2
    # 联邦循环阶段的 img2img 噪声强度（> 0.7，保留更多源图像结构）
    noise_strength_fed: float = 0.75
    # 本地训练阶段噪声强度的最小值（随机均匀采样区间下界）
    noise_strength_local_min: float = 0.35
    # 本地训练阶段噪声强度的最大值
    noise_strength_local_max: float = 0.70
    # VAE 潜空间缩放因子（对潜向量的幅度进行归一化）
    vae_scale_factor: float = 0.18215
    # 是否在首次 build_models 时自动从数据集统计并校准 vae_scale_factor
    # True：忽略 vae_scale_factor，改用数据驱动的 1/std(latents)
    calibrate_vae_scale: bool = True
    # calibrate_scale_factor() 使用的批次数（批次越多统计越稳，但初始化越慢）
    vae_calibration_batches: int = 8


@dataclass
class FederatedConfig:
    """联邦学习协议相关配置。"""
    # 联邦方法名称，可选：m4fed | fedavg | fedprox | fan | gptfl
    method: str = "m4fed"
    # 参与联邦训练的客户端数量
    n_clients: int = 2
    # 热身阶段（服务器预训练 MedMD）的轮数
    warm_rounds: int = 5
    # 联邦循环轮数（客户端↔服务器交换合成数据的轮数）
    federated_rounds: int = 10
    # 本地训练阶段每个节点的训练轮数
    local_rounds: int = 10
    # 联邦循环阶段合成数据量占本地私有数据量的比例
    synth_ratio_fed: float = 0.1
    # 本地训练阶段合成数据量占本地私有数据量的比例
    synth_ratio_local: float = 1.0
    # FedProx 近端正则化系数 μ
    fedprox_mu: float = 0.001
    # QLoRA 低秩矩阵的秩 r
    qlora_r: int = 16
    # QLoRA 缩放系数 α（实际学习率缩放比例为 α/r）
    qlora_alpha: int = 32
    # QLoRA LoRA 层的 Dropout 比例
    qlora_dropout: float = 0.05
    # QLoRA 量化位数（4-bit 量化）
    qlora_bits: int = 4


@dataclass
class TrainingConfig:
    """训练循环相关超参数。"""
    # 每个批次的样本数
    batch_size: int = 1
    # MedMD（扩散模型 U-Net）的学习率
    lr_medmd: float = 1e-4
    # NeuroFM（领域预测模型）的学习率
    lr_neurofm: float = 5e-5
    # AdamW 优化器的权重衰减系数
    weight_decay: float = 1e-5
    # 梯度裁剪阈值（防止梯度爆炸）
    grad_clip: float = 1.0
    # 是否启用混合精度训练（FP16，仅 CUDA 可用时生效）
    mixed_precision: bool = True
    # 是否启用梯度检查点（以计算换内存）
    gradient_checkpointing: bool = True
    # DataLoader 的工作线程数（0 表示在主进程中加载，确保单机可复现性）
    num_workers: int = 0
    # 计算设备：cuda 或 cpu
    device: str = "cuda"
    # 训练随机种子
    seed: int = 42
    # 年龄预测任务的损失权重（相对其他任务的加权倍数）
    age_loss_weight: float = 2.0


@dataclass
class EvalConfig:
    """评估与可视化相关配置。"""
    # 结果保存根目录
    results_dir: str = "results"
    # 是否保存训练后的模型权重
    save_model: bool = True
    # 是否在每轮联邦训练后进行评估
    eval_every_round: bool = True
    # 图表保存格式：svg（矢量图，适合论文）或 pdf
    figure_format: str = "svg"
    # 图表字体
    figure_font: str = "Carlito"
    # 各基线的颜色调色板（顺序：FedAvg, FedProx, FAN, GPT-FL, M4Fed-clients, M4Fed-server）
    palette: List[str] = field(default_factory=lambda: [
        "#8172B3", "#4874CB", "#EE822F", "#75BD42", "#D64A4A", "#2AB7A9"
    ])


@dataclass
class M4FedConfig:
    """顶层配置类，聚合所有子配置。"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # 实验名称（用于结果目录命名）
    experiment_name: str = "m4fed_default"


def load_config(yaml_path: Optional[str] = None) -> M4FedConfig:
    """
    从 YAML 文件加载配置，不存在时使用全部默认值。

    YAML 文件的结构对应 M4FedConfig 的字段层次：
      data:
        n_cross_ds: 10
      training:
        lr_medmd: 0.0005
    """
    cfg = M4FedConfig()  # 先以默认值初始化
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            overrides = yaml.safe_load(f)  # 读取 YAML 覆盖项
        if overrides:
            _apply_overrides(cfg, overrides)  # 将覆盖项写入配置对象
    return cfg


def _apply_overrides(cfg: M4FedConfig, d: dict) -> None:
    """将字典 d 中的顶层键递归写入 cfg 的对应子配置。"""
    for key, val in d.items():
        if hasattr(cfg, key):
            sub = getattr(cfg, key)
            # 如果对应字段是子配置对象且 YAML 值是字典，则递归应用
            if isinstance(sub, (DataConfig, ModelConfig, FederatedConfig,
                                TrainingConfig, EvalConfig)) and isinstance(val, dict):
                _apply_dict(sub, val)
            else:
                # 顶层标量字段（如 experiment_name）直接赋值
                setattr(cfg, key, val)


def _apply_dict(obj, d: dict) -> None:
    """将字典 d 中的键值对逐一写入对象 obj 的同名属性。"""
    for k, v in d.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
