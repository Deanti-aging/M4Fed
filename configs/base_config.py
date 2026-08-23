"""
统一配置数据类，支持 YAML 文件覆盖默认值。

设计思路：
  - 所有超参数以 Python dataclass 的形式定义（含默认值）
  - load_config(yaml_path) 先用默认值初始化，再用 YAML 文件中的字段覆盖
  - 好处：不需要 YAML 文件也能运行（退回全默认配置），YAML 只需写需要覆盖的字段

@dataclass 的作用（对比手写 __init__）：
  自动生成 __init__（含所有字段的默认值参数）、__repr__、__eq__ 等，
  避免手写：
    class Config:
        def __init__(self, lr, epochs):
            self.lr = lr
            self.epochs = epochs
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
from dataclasses import dataclass, field, asdict   # dataclass 装饰器、field（复杂默认值）、asdict（转字典）
from typing import List, Optional     # 类型注解：列表、可选值
import yaml                           # PyYAML：解析 YAML 配置文件
import os                             # 路径判断（os.path.exists）


@dataclass
class DataConfig:
    """数据相关配置（数据集来源、划分方式、MRI 预处理参数等）。"""
    # HuggingFace 上 OpenMind 数据集的仓库 ID（AnonRes/OpenMind）
    hf_dataset_id: str = "AnonRes/OpenMind"
    # 本地缓存目录（留空 "" 则使用 huggingface_hub 默认缓存目录）
    cache_dir: str = ""
    # 跨站点子数据集数量（保留用于 cross-site generalisation 评估，不参与训练）
    n_cross_ds: int = 5
    # 每个节点（服务器 + 每个客户端）分配的私有子数据集数量
    n_private_ds_per_node: int = 20
    # 训练集占数据集的比例（如 0.9 = 90% 训练 + 10% 测试）
    # 仅作用于私有数据集（服务器私有 + 各客户端私有）；公开数据全量作训练，不划分测试集
    train_split: float = 0.9
    # MRI 体积目标尺寸 (D, H, W)，单位为体素
    # field(default_factory=...) 为可变默认值（list）使用工厂函数，避免跨实例共享
    mri_shape: List[int] = field(default_factory=lambda: [96, 96, 96])
    # 3D 随机裁剪保留的边长比例（相对 mri_shape，裁剪后插值回原尺寸）
    # 0.875 表示 96³ → 裁剪 84³ 子体积 → 插值回 96³（等效随机平移 + 轻微放大）
    augment_crop_ratio: float = 0.875
    # 数据集划分的随机种子（影响 cross/private/public 的 ID 分配和 train/test 划分）
    seed: int = 42
    # 是否使用 HuggingFace 流式加载（不预下载全部数据，边训练边下载）
    streaming: bool = True
    # 惰性下载的 NIfTI 文件在本地磁盘上允许占用的总空间上限（GiB）
    # 超过该上限时 _LRUDiskCache 按最久未访问优先淘汰旧文件（LRU 策略）
    # 设为 0 或负数表示不限制（不建议在磁盘紧张的机器上使用）
    max_cache_gb: float = 80.0
    # 本地 openneuro_metadata.csv 的文件路径
    # 优先从此文件读取子数据集 ID（覆盖全量 800 个 ds_id，无需 HuggingFace API 请求）
    # 留空 "" 时退回通过 HuggingFace 元数据接口获取
    local_metadata_csv: str = "openneuro_metadata.csv"
    # 挑选 cross（跨站点评估）数据集时，age/sex/health_status 各字段允许的最大缺失率
    # 超过该阈值的 ds_id 会被跳过（换成下一个候选），避免跨站点偏差分析（性别/健康状态×年龄
    # 误差热图）因样本量过少而不可靠；仅在能读取本地 local_metadata_csv 时生效
    cross_max_missing_ratio: float = 0.5


@dataclass
class ModelConfig:
    """模型结构与预训练权重相关配置（MedMD U-Net、VAE、SwinUNETR、TabPFN 等）。"""
    # VAE（MONAI AutoencoderKL）的 HuggingFace 仓库 ID
    # MONAI/brats_mri_generative_diffusion：3D 脑 MRI 专用 VAE，
    # spatial_dims=3，latent_channels=8，将 96³ 压缩为 12³×8 潜空间
    vae_repo: str = "MONAI/brats_mri_generative_diffusion"
    # U-Net 的空间维度（3 表示处理 3D 体积；对应 DDPM 中的去噪网络 ε_θ）
    unet_spatial_dims: int = 3
    # U-Net 输入通道数（对应 VAE 潜空间通道数，必须与 vae_repo 的 latent_channels 一致）
    unet_in_channels: int = 8
    # U-Net 输出通道数（与输入通道数相同，因为预测的是与噪声同尺寸的噪声残差）
    unet_out_channels: int = 8
    # 每个分辨率级别的残差块数量（块越多容量越大，训练越慢）
    unet_num_res_blocks: int = 2
    # 各分辨率级别的特征通道数列表（从粗分辨率到细分辨率的通道数）
    unet_num_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 64])
    # 各分辨率级别是否启用自注意力（True 处的层做 Cross-Attention 条件注入）
    unet_attention_levels: List[bool] = field(default_factory=lambda: [False, False, True, True])
    # 组归一化的分组数（须整除所有通道数，否则 MONAI 报错）
    unet_norm_num_groups: int = 32
    # 交叉注意力维度（U-Net 内部使用）。MedMD 会用一个冻结投影层将 BioMedCLIP 的文本嵌入
    # （biomedclip_repo 对应的 HF BertModel 隐藏维度，PubMedBERT 为 768）对齐到这一维度，
    # 因此该值可自由配置，无需与 biomedclip_repo 的输出维度一致（如冒烟测试用更小的值提速）。
    unet_cross_attention_dim: int = 512
    # BioMedCLIP 文本编码器的 HuggingFace 仓库 ID（Microsoft 发布的生物医学 CLIP）
    biomedclip_repo: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # SwinUNETR 骨干（swinViT）自监督预训练权重下载 URL
    # 来源：MONAI 官方在脑 MRI（MSD 数据集）上自监督预训练的 checkpoint（Tang et al., CVPR 2022）
    # 留空 "" 则跳过下载、使用随机初始化（仅用于调试或消融对照实验）
    swinunetr_pretrain_url: str = (
        "https://github.com/Project-MONAI/MONAI-extra-test-data/"
        "releases/download/0.8.1/model_swinvit.pt"
    )
    # SwinUNETR 编码器基础特征维度（embed_dim = feature_size × 16，默认 48×16=768）
    # 减小 feature_size 可降低显存占用（如测试时设为 12）
    swinunetr_feature_size: int = 48
    # TabPFN 的 HuggingFace 仓库 ID（Prior-Labs 的元学习表格补全模型）
    tabpfn_repo: str = "Prior-Labs/tabpfn_2_5"
    # 扩散过程的总时间步数 T（训练时从 [1, T] 随机采样 t；T 越大扩散越充分）
    diffusion_timesteps: int = 1000
    # DDIM 采样步数（推理时使用，远小于 T 以加速；50 步通常足够）
    ddim_steps: int = 50
    # 线性 beta 调度的起始值（t=1 时的噪声方差，控制第一步加噪程度）
    beta_start: float = 1e-4
    # 线性 beta 调度的结束值（t=T 时的噪声方差，控制最后一步加噪程度）
    beta_end: float = 2e-2
    # 联邦循环阶段的 img2img 噪声强度（较高值 > 0.7，保留更多源图像的结构信息）
    noise_strength_fed: float = 0.75
    # 本地训练阶段噪声强度均匀采样区间下界（随机采样增加多样性）
    noise_strength_local_min: float = 0.35
    # 本地训练阶段噪声强度均匀采样区间上界
    noise_strength_local_max: float = 0.70
    # VAE 潜空间缩放因子（对潜向量幅度进行归一化，使其方差接近 1）
    # 默认值 0.18215 来自 Stable Diffusion，实际使用时应由 calibrate_scale_factor 覆盖
    vae_scale_factor: float = 0.18215
    # 是否在首次 build_models 时自动校准 vae_scale_factor
    # True：用真实数据统计潜向量标准差，设 scale_factor = 1/std（数据驱动）
    # False：使用 vae_scale_factor 中的固定值（适用于离线校准好缩放因子的场景）
    calibrate_vae_scale: bool = True
    # calibrate_scale_factor() 使用的批次数（批次越多统计越准确，但初始化越慢）
    vae_calibration_batches: int = 8


@dataclass
class FederatedConfig:
    """联邦学习协议相关配置（轮数、合成比例、FedProx 系数、QLoRA 超参数等）。"""
    # 联邦方法名称（选择运行哪个基线）
    # 可选：m4fed | fedavg | fedprox | fan | gptfl
    method: str = "m4fed"
    # 参与联邦训练的客户端数量（M4Fed 用 2 个，FedAvg/FedProx/FAN 用 3 个）
    n_clients: int = 2
    # 热身阶段轮数（服务器用公开数据预训练 MedMD，使 U-Net 具备基本扩散能力）
    warm_rounds: int = 5
    # 联邦循环轮数（客户端↔服务器交换合成 MRI 数据并更新领域模型的轮数）
    federated_rounds: int = 50
    # 本地训练阶段每个节点的训练轮数（Phase 3：使用合成 + 私有数据微调 SwinMultiTask）
    local_rounds: int = 50
    # 联邦循环阶段的合成数据比例（相对于本地私有数据量）
    # 0.1 = 合成 10% 的数据（较保守，避免联邦循环初期合成质量较低时引入噪声）
    synth_ratio_fed: float = 0.1
    # 本地训练阶段的合成数据比例（相对于本地私有数据量）
    # 1.0 = 合成 100% 的数据（等量合成，Phase 3 充分利用扩散模型）
    synth_ratio_local: float = 1.0
    # FedProx 近端正则化系数 μ（0 = 退化为 FedAvg，μ 越大本地更新越保守）
    fedprox_mu: float = 0.001
    # ---- FAN 热身阶段的标签平衡过采样参数 ----
    # 年龄分箱宽度（岁）：脑龄是连续标签，先按该宽度分箱再按箱内计数做平衡过采样
    fan_age_bin_width: float = 10.0
    # 单个样本允许的最大重复倍数：防止极稀有年龄段（如仅 1 个 80 岁样本）被复制上百次，
    # 导致增强也无法提供足够多样性、反而引发严重过拟合
    fan_oversample_cap: int = 5
    # QLoRA 低秩矩阵的秩 r（越小参数越少，通信效率越高，但表达能力越弱）
    qlora_r: int = 16
    # QLoRA 缩放系数 α（实际学习率缩放比例 = α/r；α=2r 是常见选择）
    qlora_alpha: int = 32
    # QLoRA LoRA 层的 Dropout 比例（防止 LoRA 适配器过拟合）
    qlora_dropout: float = 0.05
    # QLoRA 量化位数（4-bit NF4 量化，压缩骨干内存占用约 75%）
    qlora_bits: int = 4


@dataclass
class TrainingConfig:
    """训练循环相关超参数（优化器、批大小、设备、精度等）。"""
    # 每个批次的样本数（MRI 体积较大，通常为 1；增大批大小需要更多显存）
    batch_size: int = 1
    # MedMD 扩散模型 U-Net 的 AdamW 学习率
    lr_medmd: float = 1e-4
    # SwinMultiTask 领域预测模型的 AdamW 学习率（比 MedMD 低一个数量级）
    lr_swin: float = 5e-5
    # AdamW 优化器的权重衰减系数（L2 正则化，防止过拟合）
    weight_decay: float = 1e-5
    # 梯度裁剪阈值（防止梯度爆炸，特别是联邦训练中 Non-IID 数据导致的大梯度）
    grad_clip: float = 1.0
    # 是否启用混合精度训练（FP16 + FP32，仅 CUDA 设备有效，加速约 2-3 倍）
    mixed_precision: bool = True
    # 是否启用梯度检查点（以额外计算时间换取内存，处理大体积 MRI 时必要）
    gradient_checkpointing: bool = True
    # DataLoader 的工作线程数（0 = 在主进程中加载，保证可复现性和惰性下载兼容性）
    num_workers: int = 0
    # 计算设备（"cuda" 使用 GPU，"cpu" 用于无 GPU 的环境或快速测试）
    device: str = "cuda"
    # 训练随机种子（影响模型初始化、DataLoader shuffle、数据增强等）
    seed: int = 42
    # 年龄预测任务的损失权重（相对其他任务的加权倍数，年龄是主任务所以权重更大）
    age_loss_weight: float = 2.0


@dataclass
class EvalConfig:
    """评估与可视化相关配置（结果保存路径、图表格式、调色板等）。"""
    # 结果保存根目录（各实验在此创建子目录）
    results_dir: str = "results"
    # 是否在训练结束后保存模型权重（.pt 文件）
    save_model: bool = True
    # 是否在每轮联邦训练后进行一次领域模型（SwinMultiTask）评估（开启后训练更慢但更易发现过拟合）
    eval_every_round: bool = True
    # 生成模型（MedMD）质量评估参与的最大样本数（PSNR/SSIM/LPIPS/FID/KID）
    # 每个样本需一次 img2img 合成（ddim_steps 步去噪），值越大评估越慢，故设上限控制耗时
    gen_eval_max_samples: int = 16
    # 生成模型评估的 img2img 噪声强度（评估阶段固定值，独立于训练/联邦阶段的噪声区间）
    # 取中等强度：既改变足够多细节以体现生成能力，又保留结构使配对指标有意义
    gen_eval_noise_strength: float = 0.5
    # 图表保存格式：
    # "svg"  ── 矢量图，放大不失真，适合论文投稿（LaTeX/PDF 编译器直接嵌入）
    # "pdf"  ── 矢量图，PDF 阅读器直接打开
    # "png"  ── 光栅图，文件小但放大有锯齿
    figure_format: str = "svg"
    # 图表字体（需预先安装；fallback 到 DejaVu Sans）
    figure_font: str = "Carlito"
    # 各基线方法的 Hex 颜色调色板（顺序与 run_all_baselines 中的方法顺序对应）
    # 顺序：FedAvg(紫), FedProx(蓝), FAN(橙), GPT-FL(绿), M4Fed-clients(红), M4Fed-server(青)
    palette: List[str] = field(default_factory=lambda: [
        "#8172B3", "#4874CB", "#EE822F", "#75BD42", "#D64A4A", "#2AB7A9"
    ])


@dataclass
class M4FedConfig:
    """
    顶层配置类，聚合所有子配置对象。

    使用方式：
      cfg = load_config("configs/m4fed.yaml")
      print(cfg.training.lr_medmd)   # 访问训练配置
      print(cfg.model.ddim_steps)    # 访问模型配置
    """
    data: DataConfig = field(default_factory=DataConfig)              # 数据配置
    model: ModelConfig = field(default_factory=ModelConfig)           # 模型配置
    federated: FederatedConfig = field(default_factory=FederatedConfig)  # 联邦配置
    training: TrainingConfig = field(default_factory=TrainingConfig)  # 训练配置
    eval: EvalConfig = field(default_factory=EvalConfig)              # 评估配置
    # 实验名称（用于结果目录命名，如 "m4fed_default"、"ablation_wo_tabpfn" 等）
    experiment_name: str = "m4fed_default"


def load_config(yaml_path: Optional[str] = None) -> M4FedConfig:
    """
    配置加载工厂函数：用 YAML 文件中的字段覆盖默认配置并返回 M4FedConfig。

    行为：
      - 若不给 yaml_path 或文件不存在：直接返回全默认 M4FedConfig
      - 若给了有效 YAML 文件：用文件中的值覆盖对应字段（只需写需要改的字段）

    YAML 文件结构示例（对应 M4FedConfig 的字段层次）：
      data:
        n_cross_ds: 10        # 覆盖默认 5
      training:
        lr_medmd: 0.0005      # 覆盖默认 1e-4

    参数：
      yaml_path : YAML 配置文件的路径（支持相对路径）；None 时使用全默认值

    返回：
      完整的 M4FedConfig 对象（含所有子配置）
    """
    cfg = M4FedConfig()   # 以全默认值初始化顶层配置（所有子配置也用各自默认值）
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            overrides = yaml.safe_load(f)   # 解析 YAML 文件（safe_load 防止任意代码执行）
        if overrides:
            _apply_overrides(cfg, overrides)   # 将 YAML 覆盖值写入配置对象
    return cfg   # 返回最终配置（无 YAML 文件时为全默认值，有时为覆盖后的值）


def _apply_overrides(cfg: M4FedConfig, d: dict) -> None:
    """
    将顶层 YAML 字典的键值对递归写入 M4FedConfig 及其子配置。

    处理两种情况：
      1. 键对应子配置对象（如 "data" → DataConfig）且值是 dict：
         递归调用 _apply_dict 写入子配置的各字段
      2. 键对应顶层标量字段（如 "experiment_name"）且值是标量：
         直接 setattr 赋值

    参数：
      cfg : 待修改的 M4FedConfig 对象
      d   : YAML 顶层字典（键为 data/model/federated/training/eval/experiment_name 等）
    """
    for key, val in d.items():   # 遍历 YAML 顶层所有键值对
        if hasattr(cfg, key):    # 只处理 M4FedConfig 中存在的字段（忽略未知键）
            sub = getattr(cfg, key)   # 获取对应的子配置或标量值
            # 判断是否为子配置对象 + YAML 值是字典（则递归覆盖子配置字段）
            if isinstance(sub, (DataConfig, ModelConfig, FederatedConfig,
                                TrainingConfig, EvalConfig)) and isinstance(val, dict):
                _apply_dict(sub, val)   # 递归写入子配置的各字段
            else:
                # 顶层标量字段（如 experiment_name="my_exp"）直接赋值
                setattr(cfg, key, val)


def _apply_dict(obj, d: dict) -> None:
    """
    将字典 d 中的键值对逐一写入对象 obj 的同名属性。

    用于将 YAML 子节（如 data: {n_cross_ds: 10}）写入对应的子配置对象（DataConfig）。
    只写入 obj 中已存在的属性（忽略 YAML 中未知字段，防止拼写错误引发 AttributeError）。

    参数：
      obj : 子配置对象（DataConfig / ModelConfig / FederatedConfig / TrainingConfig / EvalConfig）
      d   : YAML 子节字典（键为子配置的字段名，值为覆盖值）
    """
    for k, v in d.items():      # 遍历 YAML 子节的所有键值对
        if hasattr(obj, k):     # 只写入对象中已定义的字段（防止 typo 引发崩溃）
            setattr(obj, k, v)  # 用 YAML 值覆盖默认值（Python 动态属性赋值）
