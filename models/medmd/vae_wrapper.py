"""
冻结的 3D AutoencoderKL（变分自编码器）封装。

目标预训练模型：MONAI/brats_mri_generative_diffusion（HuggingFace）
  - spatial_dims=3，latent_channels=8，num_channels=(64,128,256)
  - 权重文件：models/model_autoencoder.pt（约 84 MB）

加载优先级：
  1. 尝试从 monai.networks.nets 加载 AutoencoderKL（monai>=1.4.0 核心包自带），并从 HuggingFace 下载预训练权重
  2. 若 monai 版本过低/不可用，退回极简的 _MinimalVAE（仅用于单元测试，不具备真实编解码质量）

加载完成后冻结所有参数（VAE 在 M4Fed 中始终保持固定，不参与训练）。
"""
from __future__ import annotations  # 允许注解中使用尚未定义的类名
import torch                        # PyTorch 深度学习框架
import torch.nn as nn               # 神经网络模块（用于构建 MinimalVAE 备用模型）
from typing import Optional         # 类型注解工具


def load_vae(
    repo_id: Optional[str] = None,
    device: str = "cpu",
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    latent_channels: int = 8,
) -> nn.Module:
    """
    加载 3D AutoencoderKL，用于 MRI 体积的编码（图像 → 潜向量）
    和解码（潜向量 → 图像）。

    架构与 MONAI/brats_mri_generative_diffusion 对齐：
      num_channels=(64,128,256)，attention_levels 全 False，
      latent_channels=8，下采样倍率=4（2 个 stride-2 层）。

    参数：
      repo_id        : HuggingFace 仓库 ID（"MONAI/brats_mri_generative_diffusion"）
      device         : 目标设备
      spatial_dims   : 空间维度（3 = 3D 体积）
      in_channels    : 输入通道数（灰度 MRI 为 1）
      out_channels   : 输出通道数（与 in_channels 相同）
      latent_channels: 潜空间通道数（须与 U-Net in/out_channels 一致）

    返回：
      冻结的 VAE 模型（eval 模式，所有参数 requires_grad=False）
    """
    # 尝试加载 monai 核心包中的 AutoencoderKL（monai>=1.4.0 自带，原独立的 monai-generative 已合并进来）
    try:
        from monai.networks.nets import AutoencoderKL
        # 架构参数与 MONAI/brats_mri_generative_diffusion 的 train_autoencoder.json 完全对齐
        # 该 VAE 关闭了所有注意力机制（空间注意力 + 编码/解码器非局部注意力），目的是与 MONAI/brats_mri_generative_diffusion 预训练权重的配置保持一致。
        vae = AutoencoderKL(
            spatial_dims=spatial_dims,                             # 3D 空间（处理体积数据）
            in_channels=in_channels,                               # 输入通道（灰度 MRI = 1）
            out_channels=out_channels,                             # 输出通道（重建与输入一致）
            latent_channels=latent_channels,                       # 潜空间通道数（=8，与 UNet 输入一致）
            num_channels=(64, 128, 256),                           # 3 个分辨率级别的特征通道数
            attention_levels=(False, False, False),                # 无空间注意力（与预训练权重设置一致）
            with_encoder_nonlocal_attn=False,                      # 编码器不使用非局部注意力
            with_decoder_nonlocal_attn=False,                      # 解码器不使用非局部注意力
            num_res_blocks=2,                                      # 每个分辨率级别的残差块数
            norm_num_groups=32,                                    # 组归一化的分组数
            norm_eps=1e-6,                                         # 归一化层的数值稳定系数，其作用是在分母中加上一个极小值，防止方差为零时出现除以零的数值错误。1e-6 是一个常见的标准选择，足够小不影响计算精度，又足够大避免数值不稳定。
        )
        # 尝试从 HuggingFace Hub 下载预训练权重（首次下载后本地缓存）
        if repo_id:
            try:
                from huggingface_hub import hf_hub_download         # HuggingFace 文件下载工具
                weight_path = hf_hub_download(
                    repo_id=repo_id,                                # HuggingFace 仓库 ID
                    filename="models/model_autoencoder.pt",          # 仓库中的权重文件路径
                    cache_dir=None,                                  # 使用默认缓存目录（~/.cache/huggingface）
                )
                state = torch.load(weight_path, map_location="cpu", weights_only=True)  # 加载到 CPU
                vae.load_state_dict(state)                          # 将预训练权重写入 VAE
                print(f"[VAE] 已加载预训练权重：{repo_id}")
            except Exception as e:
                # 下载失败（无网络/文件缺失等），继续使用随机初始化（训练前应手动下载）
                print(f"[VAE] 无法加载预训练权重: {e}。使用随机初始化。")
    except ImportError:
        # monai 版本过低（<1.4.0）或未安装，退回极简 VAE（不具备真实编解码质量）
        print("[VAE] monai.networks.nets.AutoencoderKL 不可用。使用 MinimalVAE 替代。")
        vae = _MinimalVAE(in_channels, latent_channels)  # 创建极简 VAE 实例

    # 冻结所有参数：VAE 在整个训练过程中保持固定（不参与反向传播）
    for p in vae.parameters():
        p.requires_grad_(False)   # 将所有参数的梯度标志设为 False
    vae.eval()                     # 切换为推理模式（关闭 BatchNorm/Dropout 的训练行为）
    return vae.to(device)          # 将模型移动到指定设备（CPU/CUDA）并返回


# --------------------------------------------------------------------------- #
#  极简 VAE（仅用于无 MONAI 环境下的冒烟测试）                                     #
# --------------------------------------------------------------------------- #

class _MinimalVAE(nn.Module):
    """
    轻量级 3D VAE，仅用于在无 MONAI 依赖环境下验证管道流程的正确性。

    不具备真实的生成质量，不应用于正式实验。

    编码器：(B, 1, D, H, W) → (B, lat_ch, D//4, H//4, W//4)
      - 两次步长为 2 的 3D 卷积将空间分辨率降低 4 倍
      - 输出通道数为 lat_ch * 2（均值 + 方差，用 chunk 分离）

    解码器：(B, lat_ch, D//4, H//4, W//4) → (B, 1, D, H, W)
      - 两次步长为 2 的 3D 转置卷积恢复原始分辨率
      - 输出经 Tanh 激活，值域在 [-1, 1]
    """

    def __init__(self, in_ch: int = 1, lat_ch: int = 4):
        super().__init__()   # 调用父类 nn.Module 的初始化方法
        self.lat_ch = lat_ch  # 保存潜空间通道数，供 encode 时使用
        # 编码器：空间分辨率 × 1/4，通道数 × lat_ch * 2（前半为均值，后半为对数方差）
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, stride=2, padding=1),  # 第1层：输入通道→8通道，空间 ×1/2
            nn.SiLU(),                                      # SiLU（Swish）激活函数
            nn.Conv3d(8, lat_ch * 2, 3, stride=2, padding=1),  # 第2层：8→lat_ch*2通道，空间 ×1/2
        )
        # 解码器：空间分辨率 × 4，通道数恢复为 in_ch（转置卷积上采样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(lat_ch, 8, 4, stride=2, padding=1),  # 第1层：上采样 ×2
            nn.SiLU(),                                                # SiLU 激活
            nn.ConvTranspose3d(8, in_ch, 4, stride=2, padding=1),    # 第2层：上采样 ×2，恢复原始通道
            nn.Tanh(),   # Tanh 激活：将输出值域限制在 [-1, 1]，与 MRI 预处理后的值域一致
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：返回均值向量（丢弃方差部分，简化实现）。"""
        # 这两行是 VAE（变分自编码器）编码器的简化推断逻辑：
        # 背景：编码器输出的通道数是 lat_ch * 2，刻意设计成两倍——按照 VAE 的标准做法，前半通道代表 均值 μ，后半通道代表 对数方差 log σ²。
        # 为什么这样做？完整 VAE 推断时应该用重参数化采样 z = μ + ε·σ（ε 是随机噪声），引入随机性。但这里是推断/测试阶段的简化：直接用均值 μ 作为确定性的潜向量，避免随机性，让编码结果稳定可复现。这是一种常见的 VAE 推断惯例。
        h = self.encoder(x)               # 通过编码器得到 2×lat_ch 通道的特征图
        mu, _ = h.chunk(2, dim=1)         # 沿通道维度分成两半：前半为均值（μ），后半为对数方差
        return mu                          # 只返回均值向量（简化：忽略重参数化采样）

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码：潜向量 → 重建图像。"""
        return self.decoder(z)  # 通过解码器将潜向量映射回图像空间

    def forward(self, x: torch.Tensor):
        """
        完整自编码前向传播，返回 (重建图像, None)。
        第二个返回值为 None，与 MONAI AutoencoderKL 的 (output, _) 接口兼容。
        """
        z = self.encode(x)           # 编码：图像 → 潜向量
        return self.decode(z), None  # 解码：潜向量 → 重建图像；None 占位（兼容 MONAI 接口）
