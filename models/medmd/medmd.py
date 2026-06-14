"""
MedMD：用于医学 MRI 合成的 3D 潜在扩散模型（Latent Diffusion Model）。

整体架构：
  - 冻结的 MONAI AutoencoderKL   : MRI 体积 ↔ 潜向量（编码/解码）
  - 可训练的 MONAI DiffusionModelUNet（3D）: 在潜空间预测去噪噪声，支持文本交叉注意力
  - 冻结的 BioMedCLIP 文本编码器 : 元数据文本 → 交叉注意力条件向量

img2img 推理流程（对应论文图 2）：
  1. z₀ = VAE.encode(x_source)                                  —— 将源图像编码为潜向量
  2. z_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε    (t = noise_strength × T)  —— 前向加噪
  3. z₀_hat = DDIM_denoise(z_t, condition=c)                     —— 条件去噪
  4. x_hat = VAE.decode(z₀_hat)                                  —— 解码为合成图像

其中：
  - noise_strength ∈ (0, 1)，控制与源图像的相似程度（越大越不相似）
  - c = BioMedCLIP(text)，来自目标节点的元数据文本条件
"""
from __future__ import annotations
import copy
import math
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

from .vae_wrapper import load_vae
from .conditioning import BioMedCLIPEncoder


# --------------------------------------------------------------------------- #
#  DDIM 噪声调度器                                                               #
# --------------------------------------------------------------------------- #

class DDIMScheduler:
    """
    DDIM（Denoising Diffusion Implicit Models）噪声调度器。

    管理扩散过程的噪声参数（β 调度）和 DDIM 确定性采样步骤。

    β 调度：线性调度，从 beta_start 到 beta_end，但在平方根空间插值
    （即 betas[t] = (√start + t/(T-1) * (√end - √start))²），
    比直接线性 β 调度更平滑。

    核心公式：
      ᾱ_t = ∏_{s=1}^{t} (1 - β_s)   —— 累积乘积，控制信号与噪声的比例
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        """
        参数：
          T          : 扩散总步数
          beta_start : 线性 β 调度的起始值
          beta_end   : 线性 β 调度的结束值
        """
        self.T = T
        # 在 √β 空间线性插值后取平方，等效于 cosine schedule 的温和变体
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2
        alphas = 1.0 - betas
        # ᾱ_t = α₁ · α₂ · ... · α_t（累积连乘积）
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(self, z0: torch.Tensor, t: torch.Tensor,
                  eps: Optional[torch.Tensor] = None) -> tuple:
        """
        前向加噪（扩散过程）：z_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε

        参数：
          z0 : 干净潜向量，形状 (B, C, D, H, W)
          t  : 时间步，形状 (B,)，每个样本可以不同
          eps: 可选的预设噪声；若为 None 则随机采样

        返回：
          (z_t, eps) - 加噪后的潜向量和实际使用的噪声
        """
        ac = self.alphas_cumprod.to(z0.device)
        # 取出对应时间步的 ᾱ_t，扩展为与 z0 相同维度以便广播
        ab = ac[t].view(-1, *([1] * (z0.ndim - 1)))
        if eps is None:
            eps = torch.randn_like(z0)
        return ab.sqrt() * z0 + (1 - ab).sqrt() * eps, eps

    def get_timesteps_for_strength(self, strength: float, ddim_steps: int) -> torch.Tensor:
        """
        根据 noise_strength 生成 DDIM 采样时间步序列。

        img2img 模式下从 t_start = int(strength × T) 开始去噪（而非从 T-1 开始），
        因为只需要去掉强度为 strength 对应的噪声量。

        返回：
          形状 (ddim_steps,) 的 long 张量，从 t_start 均匀递减到 1
        """
        t_start = int(strength * self.T)
        # 生成 ddim_steps+1 个时间点，去掉最后一个（0 由调用方处理）
        steps = torch.linspace(t_start, 0, ddim_steps + 1, dtype=torch.long)
        return steps[:-1]

    @torch.no_grad()
    def ddim_step(
        self,
        z_t: torch.Tensor,
        t_now: int,
        t_next: int,
        eps_pred: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        执行一步 DDIM 确定性去噪：从 z_{t_now} 推进到 z_{t_next}。

        DDIM 公式（eta=0 时为确定性推理，eta>0 时引入随机性）：
          z₀_pred = (z_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t     —— 预测 z₀
          σ = η · √((1-ᾱ_s)/(1-ᾱ_t) · (1-ᾱ_t/ᾱ_s))     —— 随机性强度
          z_{t-1} = √ᾱ_s · z₀_pred + √(1-ᾱ_s-σ²) · ε_θ + σ · noise

        参数：
          z_t     : 当前含噪潜向量
          t_now   : 当前时间步
          t_next  : 下一时间步（更小，向 0 前进）
          eps_pred: U-Net 预测的噪声
          eta     : 随机性系数（0=完全确定性，1=DDPM 等价）

        返回：
          z_{t_next}
        """
        ac = self.alphas_cumprod.to(z_t.device)
        ab_t = ac[t_now]
        ab_s = ac[t_next] if t_next >= 0 else torch.tensor(1.0, device=z_t.device)

        # 用预测的噪声 ε_θ 反推干净图像 z₀（带 clamp 防止数值不稳定）
        z0_pred = (z_t - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        z0_pred = z0_pred.clamp(-1, 1)

        # 计算随机性强度（eta=0 时 sigma=0，完全确定性）
        sigma = eta * ((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s)).sqrt()
        noise = torch.randn_like(z_t) if eta > 0 else torch.zeros_like(z_t)

        # DDIM 更新公式
        z_next = (ab_s.sqrt() * z0_pred
                  + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps_pred
                  + sigma * noise)
        return z_next


# --------------------------------------------------------------------------- #
#  3D U-Net 构建（优先 MONAI，降级到极简版）                                       #
# --------------------------------------------------------------------------- #

class _SimpleUNet3D(nn.Module):
    """
    极简 3D U-Net，仅用于 MONAI GenerativeModels 不可用时的冒烟测试。

    结构：编码器（单卷积层）→ 中间层 → 解码器（单卷积层）
    时间步嵌入和文本条件通过加法方式注入（通道偏置），非标准但够用于测试。

    注意：此模型不具备真实生成质量。
    """

    def __init__(self, in_ch: int = 8, out_ch: int = 8, cross_attn_dim: int = 512,
                 base_ch: int = 32):
        super().__init__()
        # 时间步嵌入：将标量 t 映射为 base_ch 维向量
        self.time_emb = nn.Sequential(
            nn.Linear(1, base_ch), nn.SiLU(), nn.Linear(base_ch, base_ch)
        )
        # 文本条件投影：将 cross_attn_dim 维文本嵌入映射为 base_ch 维
        self.context_proj = nn.Linear(cross_attn_dim, base_ch)
        # 编码器、中间层、解码器
        self.enc = nn.Sequential(nn.Conv3d(in_ch, base_ch, 3, padding=1), nn.SiLU())
        self.mid = nn.Sequential(nn.Conv3d(base_ch, base_ch, 3, padding=1), nn.SiLU())
        self.dec = nn.Conv3d(base_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        参数：
          x       : 含噪潜向量 (B, in_ch, D, H, W)
          t       : 时间步 (B,)，整数
          context : 文本条件 (B, L, cross_attn_dim)
        """
        # 时间步归一化到 [0, 1] 后嵌入
        t_emb = self.time_emb(t.float().unsqueeze(-1) / 1000.0)   # (B, base_ch)
        # 对序列维度取均值，压缩为单个向量
        ctx = self.context_proj(context.mean(1))                   # (B, base_ch)
        h = self.enc(x)
        # 将时间步和条件向量作为空间偏置叠加到特征图
        bias = (t_emb + ctx).view(-1, h.shape[1], 1, 1, 1)
        h = self.mid(h + bias)
        return self.dec(h)


def _build_monai_unet(cfg) -> nn.Module:
    """
    尝试构建 MONAI DiffusionModelUNet（3D 带文本交叉注意力）。
    失败时退回 _SimpleUNet3D 以保证管道可以运行。
    """
    try:
        from generative.networks.nets import DiffusionModelUNet
        unet = DiffusionModelUNet(
            spatial_dims=cfg.unet_spatial_dims,
            in_channels=cfg.unet_in_channels,
            out_channels=cfg.unet_out_channels,
            num_res_blocks=cfg.unet_num_res_blocks,
            num_channels=cfg.unet_num_channels,
            attention_levels=cfg.unet_attention_levels,
            norm_num_groups=cfg.unet_norm_num_groups,
            with_conditioning=True,                      # 启用交叉注意力
            cross_attention_dim=cfg.unet_cross_attention_dim,
        )
        return unet
    except Exception as e:
        print(f"[UNet] MONAI DiffusionModelUNet 不可用: {e}。使用极简 UNet 替代。")
        return _SimpleUNet3D(
            in_ch=cfg.unet_in_channels,
            out_ch=cfg.unet_out_channels,
            cross_attn_dim=cfg.unet_cross_attention_dim,
        )


# --------------------------------------------------------------------------- #
#  MedMD 主体                                                                   #
# --------------------------------------------------------------------------- #

class MedMD(nn.Module):
    """
    MedMD：M4Fed 的医学图像扩散生成模块。

    角色：
      - 在联邦训练中，各节点使用 MedMD 执行 img2img 合成，
        生成满足目标条件的合成 MRI，用于数据增广和跨节点知识传递
      - U-Net 是唯一可训练的组件；VAE 和文本编码器始终冻结

    训练目标（LDM 损失）：
      L = E_{z₀, t, ε} [ ‖ ε - ε_θ(z_t, t, c) ‖² ]
      即：给定加噪潜向量 z_t 和文本条件 c，U-Net 预测噪声 ε 的 MSE
    """

    def __init__(self, model_cfg, device: str = "cpu"):
        super().__init__()
        self.cfg = model_cfg
        self.device_str = device

        # ---- 冻结组件：不参与训练 ----
        # VAE：将 MRI 体积编码到潜空间（空间降采样 4 倍，latent_channels=8），并从潜向量重建图像
        self.vae = load_vae(
            repo_id=model_cfg.vae_repo,
            device=device,
            latent_channels=model_cfg.unet_in_channels,
        )
        # 潜向量缩放因子（将 VAE 编码的潜向量方差归一化到单位量级）
        self.scale_factor = model_cfg.vae_scale_factor

        # 文本编码器：将元数据文本条件编码为交叉注意力向量
        self.text_encoder = BioMedCLIPEncoder(
            repo_id=model_cfg.biomedclip_repo,
            device=device,
        )

        # ---- 可训练组件 ----
        # U-Net：在潜空间中预测去噪方向（条件去噪网络 ε_θ）
        self.unet = _build_monai_unet(model_cfg)

        # DDIM 调度器：管理噪声时间步和去噪步骤
        self.scheduler = DDIMScheduler(
            T=model_cfg.diffusion_timesteps,
            beta_start=model_cfg.beta_start,
            beta_end=model_cfg.beta_end,
        )

    # ---------------------------------------------------------------------- #
    #  训练前向传播：计算 LDM 去噪损失                                            #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        mri: torch.Tensor,    # (B, 1, D, H, W)，值域 [-1, 1]
        texts: List[str],
    ) -> torch.Tensor:
        """
        计算 LDM 去噪损失：L = E[‖ ε - ε_θ(z_t, t, c) ‖²]

        流程：
          1. VAE 编码：MRI → 潜向量 z₀（冻结，不计梯度）
          2. 随机采样时间步 t ∈ [0, T)
          3. 前向加噪：z₀ + 噪声 → z_t
          4. 文本条件：texts → 交叉注意力向量 c（冻结）
          5. U-Net 预测：(z_t, t, c) → ε_pred
          6. MSE 损失：(ε_pred, ε)

        参数：
          mri   : 输入 MRI 批次，形状 (B, 1, D, H, W)
          texts : 对应的文本条件列表，长度 B

        返回：
          标量损失张量
        """
        B = mri.shape[0]

        # 编码（冻结 VAE，不反向传播）
        with torch.no_grad():
            z0 = self._encode(mri)   # (B, lat_ch, d, h, w)

        # 随机时间步（每个样本独立采样，增加训练多样性）
        t = torch.randint(0, self.scheduler.T, (B,), device=mri.device)

        # 前向加噪
        z_t, eps = self.scheduler.add_noise(z0, t)

        # 文本条件（冻结文本编码器）
        with torch.no_grad():
            context = self.text_encoder(texts)   # (B, L, D_clip)

        # U-Net 预测噪声
        eps_pred = self.unet(z_t, t, context)

        loss = nn.functional.mse_loss(eps_pred, eps)
        return loss

    # ---------------------------------------------------------------------- #
    #  img2img 推理：从源 MRI 生成条件合成 MRI                                    #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def synthesise(
        self,
        source_mri: torch.Tensor,   # (B, 1, D, H, W)，值域 [-1, 1]
        texts: List[str],
        noise_strength: float = 0.75,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        img2img 合成：在源 MRI 的基础上，根据目标文本条件生成新的 MRI。

        控制参数：
          noise_strength=0.75：保留源 MRI 约 25% 的结构信息，改变约 75%
          eta=0.0：完全确定性 DDIM 采样（更稳定）

        流程：
          1. 编码源 MRI 为 z₀
          2. 在 t_start = noise_strength×T 处加噪得到 z_t
          3. 从 t_start 开始 DDIM 去噪 ddim_steps 步，条件为目标文本
          4. 解码 z₀_hat 为合成图像

        返回：
          合成 MRI 张量，形状 (B, 1, D, H, W)，值域 [-1, 1]
        """
        self.eval()
        z0 = self._encode(source_mri)

        # 在 t_start 处加噪（部分扩散）
        t_start = int(noise_strength * self.scheduler.T)
        t_tensor = torch.full((z0.shape[0],), t_start - 1,
                               dtype=torch.long, device=z0.device)
        z_t, _ = self.scheduler.add_noise(z0, t_tensor)

        # 文本条件
        context = self.text_encoder(texts)

        # DDIM 去噪迭代
        timesteps = self.scheduler.get_timesteps_for_strength(noise_strength, ddim_steps)
        z = z_t
        for i, t_now in enumerate(timesteps):
            t_next = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0
            t_tensor = torch.full((z.shape[0],), int(t_now), dtype=torch.long, device=z.device)
            eps_pred = self.unet(z, t_tensor, context)
            z = self.scheduler.ddim_step(z, int(t_now), t_next, eps_pred, eta=eta)

        return self._decode(z)

    @torch.no_grad()
    def synthesise_from_label(
        self,
        age_label,
        batch_size: int = 1,
        mri_shape: Tuple[int, ...] = (96, 96, 96),
        noise_strength: float = 0.75,
        ddim_steps: int = 50,
    ) -> torch.Tensor:
        """
        纯文本条件生成（GPT-FL 风格）：不需要源 MRI，
        直接从随机噪声出发，以年龄文本为条件生成脑年龄 MRI。

        与 synthesise() 的区别：
          - 起点是随机高斯噪声（而非源 MRI 加噪）
          - 从 T-1 开始完整去噪（而非部分去噪）
          - 仅使用年龄标签作为文本条件（text = "age is {age_label}"）

        用途：GPT-FL 服务器根据客户端共享的年龄标签分布，
        生成具有代表性的合成数据分发给客户端。

        返回：
          合成 MRI 张量，形状 (batch_size, 1, D, H, W)
        """
        self.eval()
        # 计算潜向量的空间尺寸（VAE 每个方向降采样 4 倍）
        lat_d = mri_shape[0] // 4
        lat_h = mri_shape[1] // 4
        lat_w = mri_shape[2] // 4
        lat_ch = self.cfg.unet_in_channels
        # 从标准正态分布采样初始潜向量（完全随机，不依赖源图像）
        z = torch.randn(batch_size, lat_ch, lat_d, lat_h, lat_w,
                        device=next(self.unet.parameters()).device)

        texts = [f"age is {age_label}"] * batch_size
        context = self.text_encoder(texts)

        # 从 T-1 开始完整去噪（生成模式，非 img2img 模式）
        timesteps = torch.linspace(self.scheduler.T - 1, 0, ddim_steps, dtype=torch.long)
        for i, t_now in enumerate(timesteps):
            t_next = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0
            t_tensor = torch.full((z.shape[0],), int(t_now), dtype=torch.long, device=z.device)
            eps_pred = self.unet(z, t_tensor, context)
            z = self.scheduler.ddim_step(z, int(t_now), t_next, eps_pred, eta=0.0)

        return self._decode(z)

    # ---------------------------------------------------------------------- #
    #  VAE 编码/解码辅助方法                                                     #
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    #  VAE 缩放因子校准                                                         #
    # ---------------------------------------------------------------------- #

    def calibrate_scale_factor(self, loader, n_batches: int = 8) -> float:
        """
        从 DataLoader 采样若干批次，统计 VAE 潜向量的全局标准差，
        以 1/std 更新 self.scale_factor。

        应在首次 build_models 后、训练开始前调用一次。
        VAE 保持冻结；不产生梯度。

        参数：
          loader   : 返回 {'mri': Tensor(B,1,D,H,W)} 的 DataLoader
          n_batches: 用于统计的批次数（默认 8）

        返回：
          更新后的 scale_factor
        """
        chunks: list = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                mri = batch["mri"].to(self.device_str)
                try:
                    z, _ = self.vae.encode(mri)
                except Exception:
                    z = self.vae.encode(mri)
                chunks.append(z.float().flatten().cpu())

        if not chunks:
            return self.scale_factor

        global_std = torch.cat(chunks).std().item()
        if global_std < 1e-6:
            print("[MedMD] 警告：潜向量 std 过小，跳过校准，保留原 scale_factor。")
            return self.scale_factor

        new_scale = 1.0 / global_std
        old_scale = self.scale_factor
        self.scale_factor = new_scale
        print(
            f"[MedMD] VAE scale_factor 已校准: {old_scale:.5f} → {new_scale:.5f}"
            f"（潜向量 std={global_std:.4f}，共 {len(chunks)} 个批次）"
        )
        return new_scale

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 VAE 编码器将 MRI 映射到潜空间，并乘以缩放因子。

        兼容两种 VAE 接口：
          - MONAI AutoencoderKL：encode() 返回 (z, posterior)
          - _MinimalVAE：encode() 直接返回 z
        """
        try:
            z, _ = self.vae.encode(x)
            return z * self.scale_factor
        except Exception:
            return self.vae.encode(x) * self.scale_factor

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        使用 VAE 解码器将潜向量重建为 MRI 体积，并除以缩放因子。

        兼容两种 VAE 接口：
          - MONAI AutoencoderKL：decode() 直接返回重建图像
          - _MinimalVAE：forward() 返回 (output, None)
        """
        z = z / self.scale_factor
        try:
            return self.vae.decode(z)
        except Exception:
            out, _ = self.vae(z)
            return out

    # ---------------------------------------------------------------------- #
    #  权重访问接口（联邦聚合使用）                                                #
    # ---------------------------------------------------------------------- #

    def unet_state_dict(self) -> dict:
        """
        返回 U-Net 的状态字典（深拷贝）。

        仅 U-Net 参与联邦热身阶段的权重分发，
        VAE 和文本编码器始终冻结，无需传输。
        """
        return copy.deepcopy(self.unet.state_dict())

    def load_unet_state_dict(self, state: dict, strict: bool = False) -> None:
        """从状态字典恢复 U-Net 权重（strict=False 允许部分加载）。"""
        self.unet.load_state_dict(state, strict=strict)
