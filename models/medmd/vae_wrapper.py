"""
冻结的 3D AutoencoderKL（变分自编码器）封装。

目标预训练模型：MONAI/brats_mri_generative_diffusion（HuggingFace）
  - spatial_dims=3，latent_channels=8，num_channels=(64,128,256)
  - 权重文件：models/model_autoencoder.pt（约 84 MB）

加载优先级：
  1. 尝试从 MONAI GenerativeModels 加载 AutoencoderKL，并从 HuggingFace 下载预训练权重
  2. 若 MONAI 不可用，退回极简的 _MinimalVAE（仅用于单元测试，不具备真实编解码质量）

加载完成后冻结所有参数（VAE 在 M4Fed 中始终保持固定，不参与训练）。
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


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
    # 尝试加载 MONAI GenerativeModels 中的 AutoencoderKL
    try:
        from generative.networks.nets import AutoencoderKL
        # 架构参数与 MONAI/brats_mri_generative_diffusion 的 train_autoencoder.json 完全对齐
        vae = AutoencoderKL(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            num_channels=(64, 128, 256),              # 3 个分辨率级别
            attention_levels=(False, False, False),    # 无空间注意力（与预训练权重一致）
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            num_res_blocks=2,
            norm_num_groups=32,
            norm_eps=1e-6,
        )
        # 尝试从 HuggingFace Hub 下载预训练权重
        if repo_id:
            try:
                from huggingface_hub import hf_hub_download
                weight_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="models/model_autoencoder.pt",
                    cache_dir=None,
                )
                state = torch.load(weight_path, map_location="cpu", weights_only=True)
                vae.load_state_dict(state)
                print(f"[VAE] 已加载预训练权重：{repo_id}")
            except Exception as e:
                print(f"[VAE] 无法加载预训练权重: {e}。使用随机初始化。")
    except ImportError:
        # MONAI GenerativeModels 未安装，退回极简 VAE
        print("[VAE] MONAI GenerativeModels 未找到。使用 MinimalVAE 替代。")
        vae = _MinimalVAE(in_channels, latent_channels)

    # 冻结所有参数：VAE 在整个训练过程中保持固定
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()
    return vae.to(device)


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
        super().__init__()
        self.lat_ch = lat_ch
        # 编码器：空间分辨率 × 1/4，通道数 × lat_ch * 2
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(8, lat_ch * 2, 3, stride=2, padding=1),
        )
        # 解码器：空间分辨率 × 4，通道数恢复为 in_ch
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(lat_ch, 8, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(8, in_ch, 4, stride=2, padding=1),
            nn.Tanh(),   # 输出值域 [-1, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：返回均值向量（丢弃方差部分，简化实现）。"""
        h = self.encoder(x)
        mu, _ = h.chunk(2, dim=1)   # 前半为均值，后半为对数方差
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码：潜向量 → 重建图像。"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        完整自编码前向传播，返回 (重建图像, None)。
        第二个返回值为 None，与 MONAI AutoencoderKL 的 (output, _) 接口兼容。
        """
        z = self.encode(x)
        return self.decode(z), None
