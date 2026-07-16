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
from __future__ import annotations   # 允许注解中使用尚未定义的类名（前向引用）
import copy                           # 标准库：深拷贝（用于状态字典的安全复制）
import math                           # 标准库：数学函数（sqrt 等，供注释参考）
import torch                          # PyTorch 深度学习框架
import torch.nn as nn                 # 神经网络模块基类（Module、Conv3d 等）
import numpy as np                    # NumPy 数值计算（供校准时的统计计算）
from typing import List, Optional, Tuple  # 类型注解工具

from .vae_wrapper import load_vae           # 本包的 VAE 加载函数（见 vae_wrapper.py）
from .conditioning import BioMedCLIPEncoder  # 本包的 BioMedCLIP 文本编码器（见 conditioning.py）


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
          T          : 扩散总步数（训练时均匀采样，推理时 DDIM 使用子集）
          beta_start : 线性 β 调度的起始值（小→扩散初期噪声少）
          beta_end   : 线性 β 调度的结束值（大→扩散末期噪声多）
        """
        self.T = T   # 保存扩散总步数，供外部访问（如计算 t_start）
        # 在 √β 空间线性插值后取平方：此调度比直接线性 β 更平缓，数值更稳定
        # 效果是让噪声调度在初期更"温和"，避免早期扩散步骤中信号衰减过快，数值上也更稳定（不会出现 β 突变导致的训练不稳定）。这是 improved DDPM (2021) 中提出的调度技巧之一
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2
        # α_t = 1 - β_t：每步保留的信号比例
        alphas = 1.0 - betas
        # ᾱ_t = α₁ · α₂ · ... · α_t（累积连乘积）；t 步后信号强度 = √ᾱ_t
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(self, z0: torch.Tensor, t: torch.Tensor,
                  eps: Optional[torch.Tensor] = None) -> tuple:
        """
        前向加噪（扩散过程正向传播）：z_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε

        这是 DDPM 的重参数化加噪公式，允许在任意时间步 t 直接加噪，
        无需逐步迭代（训练时高效批量采样随机时间步）。

        参数：
          z0 : 干净潜向量，形状 (B, C, D, H, W)
          t  : 时间步索引，形状 (B,)，每个样本可以独立采样不同的 t
          eps: 可选的预设噪声；若为 None 则从标准正态分布随机采样

        返回：
          (z_t, eps) 元组
            - z_t : 加噪后的潜向量，形状同 z0
            - eps : 实际使用的噪声（用于计算训练损失 ‖ε - ε_pred‖²）
        """
        ac = self.alphas_cumprod.to(z0.device)  # 将调度表移到与 z0 相同的设备
        # 从 alphas_cumprod 中取出每个样本对应时间步的 ᾱ_t
        # view(-1, *([1]*(z0.ndim-1))) 将 (B,) 扩展为 (B,1,1,1,1)，以便与 z0 广播
        ab = ac[t].view(-1, *([1] * (z0.ndim - 1)))
        if eps is None:
            eps = torch.randn_like(z0)   # 从标准正态分布采样噪声（形状同 z0）
        # DDPM 加噪公式：信号项 √ᾱ_t · z₀ + 噪声项 √(1-ᾱ_t) · ε
        return ab.sqrt() * z0 + (1 - ab).sqrt() * eps, eps

    def get_timesteps_for_strength(self, strength: float, ddim_steps: int) -> torch.Tensor:
        """
        根据 noise_strength 生成 DDIM 采样时间步序列（img2img 推理专用）。

        img2img 模式下，不需要从 T-1 开始完整去噪，只需从
        t_start = int(strength × T) 开始去噪（对应前向加了多少噪就去多少噪）。

        参数：
          strength   : 噪声强度 ∈ (0, 1)，越大表示加噪越多，越不像源图像
          ddim_steps : DDIM 去噪步数（少步快速，多步高质量）

        返回：
          形状 (ddim_steps,) 的 long 张量，从 t_start 均匀递减到 1
          （最后一步到 0 由 ddim_step 中的 t_next=0 处理）
        """
        t_start = int(strength * self.T)   # 加噪量对应的最高时间步
        # linspace 生成 ddim_steps+1 个等间隔点（包含两端），去掉最后一个（0）
        steps = torch.linspace(t_start, 0, ddim_steps + 1, dtype=torch.long)
        return steps[:-1]   # 返回前 ddim_steps 个时间步（不含 0，由 t_next 处理）

    @torch.no_grad()   # 推理时不需要梯度，节省显存和计算
    def ddim_step(
        self,
        z_t: torch.Tensor,      # 当前含噪潜向量 (B, C, D, H, W)
        t_now: int,              # 当前时间步（整数，对应 alphas_cumprod 的索引）
        t_next: int,             # 下一时间步（整数，t_now > t_next，向 0 靠近）
        eps_pred: torch.Tensor,  # U-Net 预测的噪声 ε_θ(z_t, t, c)
        eta: float = 0.0,        # 随机性系数（0=完全确定性，1=DDPM 等价）
    ) -> torch.Tensor:
        """
        执行一步 DDIM 确定性去噪：从 z_{t_now} 推进到 z_{t_next}。

        DDIM 公式（eta=0 时为确定性推理，eta>0 时引入随机性）：
          z₀_pred = (z_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t        —— 由 z_t 反推干净 z₀
          σ = η · √((1-ᾱ_s)/(1-ᾱ_t) · (1-ᾱ_t/ᾱ_s))        —— 随机性强度
          z_{t-1} = √ᾱ_s · z₀_pred + √(1-ᾱ_s-σ²) · ε_θ + σ · noise

        其中 ᾱ_s = alphas_cumprod[t_next]，ᾱ_t = alphas_cumprod[t_now]。

        参数：
          z_t     : 当前含噪潜向量 (B, C, D, H, W)
          t_now   : 当前时间步（整数）
          t_next  : 下一时间步（整数，t_now > t_next >= 0）
          eps_pred: U-Net 预测的噪声方向
          eta     : 随机性系数（0 = 完全确定性 DDIM）

        返回：
          z_{t_next}：去噪一步后的潜向量
        """
        ac = self.alphas_cumprod.to(z_t.device)  # 调度表移到与 z_t 相同的设备
        ab_t = ac[t_now]   # ᾱ_{t_now}：当前时间步的累积 α
        # 若 t_next < 0（理论上不应发生，防御性处理），ᾱ_s 设为 1（干净图像）
        ab_s = ac[t_next] if t_next >= 0 else torch.tensor(1.0, device=z_t.device)

        # 用预测噪声反推干净潜向量 z₀_pred：z₀ = (z_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
        z0_pred = (z_t - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        z0_pred = z0_pred.clamp(-1, 1)   # clamp 防止数值不稳定（理论上 z₀ ∈ [-1,1]）

        # 计算随机性强度 σ（eta=0 时 sigma=0，使去噪完全确定性）
        sigma = eta * ((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s)).sqrt()
        # eta=0 时不需要随机噪声，直接用零张量避免无效计算
        noise = torch.randn_like(z_t) if eta > 0 else torch.zeros_like(z_t)

        # DDIM 更新公式：√ᾱ_s · z₀_pred + √(1-ᾱ_s-σ²) · ε_θ + σ · noise
        # clamp(min=0) 防止 (1-ᾱ_s-σ²) 因浮点误差变为负数导致 sqrt 出 NaN
        z_next = (ab_s.sqrt() * z0_pred
                  + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps_pred
                  + sigma * noise)
        return z_next   # 返回去噪一步后的潜向量


# --------------------------------------------------------------------------- #
#  3D U-Net 构建（优先 MONAI，降级到极简版）                                       #
# --------------------------------------------------------------------------- #

class _SimpleUNet3D(nn.Module):
    """
    极简 3D U-Net，仅用于 MONAI GenerativeModels 不可用时的冒烟测试。

    结构：编码器（单卷积层）→ 中间层 → 解码器（单卷积层）
    时间步嵌入和文本条件通过加法方式注入（通道偏置），非标准但够用于测试。

    注意：此模型不具备真实生成质量，不应用于正式实验。
    """

    def __init__(self, in_ch: int = 8, out_ch: int = 8, cross_attn_dim: int = 512,
                 base_ch: int = 32):
        """
        参数：
          in_ch         : 输入潜向量的通道数（=latent_channels）
          out_ch        : 输出噪声预测的通道数（通常与 in_ch 相同）
          cross_attn_dim: 文本条件嵌入的维度（BioMedCLIP 默认 512）
          base_ch       : 中间特征通道数（控制网络容量）
        """
        super().__init__()   # 调用 nn.Module 父类初始化
        # 时间步嵌入：标量时间步 t → base_ch 维向量（MLP 两层）
        # 时间信息通过此 MLP 转化为特征，与空间特征叠加注入去噪方向
        self.time_emb = nn.Sequential(
            nn.Linear(1, base_ch),   # 将标量 t 映射为 base_ch 维
            nn.SiLU(),                # SiLU（Swish）激活：平滑非线性
            nn.Linear(base_ch, base_ch)  # 再映射一次（非线性表达能力）
        )
        # 文本条件投影：BioMedCLIP 嵌入 (cross_attn_dim) → 空间偏置 (base_ch)
        # 将文本信息压缩为与空间特征同维的向量，以加法方式注入
        self.context_proj = nn.Linear(cross_attn_dim, base_ch)
        # 编码器：将含噪潜向量压缩为特征图
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, 3, padding=1),  # 3D 卷积，保持空间分辨率
            nn.SiLU()  # SiLU 激活
        )
        # 中间层：在注入条件后进一步提取特征
        self.mid = nn.Sequential(
            nn.Conv3d(base_ch, base_ch, 3, padding=1),  # 3D 卷积，保持空间分辨率
            nn.SiLU()  # SiLU 激活
        )
        # 解码器：将特征图映射回噪声预测（与输入形状相同）
        self.dec = nn.Conv3d(base_ch, out_ch, 3, padding=1)  # 最终输出层，无激活函数

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        前向传播：预测在时间步 t、条件 context 下的去噪噪声方向 ε_θ。

        参数：
          x       : 含噪潜向量 (B, in_ch, D, H, W)
          t       : 时间步 (B,)，整数（0 到 T-1）
          context : 文本条件嵌入 (B, L, cross_attn_dim)，L 为序列长度

        返回：
          预测的噪声张量，形状 (B, out_ch, D, H, W)
        """
        # 时间步归一化到 [0, 1] 再嵌入（使 MLP 输入范围稳定）
        t_emb = self.time_emb(t.float().unsqueeze(-1) / 1000.0)   # (B, base_ch)
        # 对文本序列维度取均值，将 (B, L, cross_attn_dim) 压缩为 (B, base_ch)
        ctx = self.context_proj(context.mean(1))   # (B, base_ch)
        h = self.enc(x)   # 编码器提取空间特征 (B, base_ch, D, H, W)
        # 将时间步嵌入和文本条件叠加，view 为 (B, base_ch, 1, 1, 1) 以广播到空间维度
        bias = (t_emb + ctx).view(-1, h.shape[1], 1, 1, 1)
        h = self.mid(h + bias)   # 将条件信息以加法注入特征图并通过中间层
        return self.dec(h)       # 解码输出噪声预测 (B, out_ch, D, H, W)


def _build_monai_unet(cfg) -> nn.Module:
    """
    尝试构建 MONAI DiffusionModelUNet（3D 带文本交叉注意力），
    这是 MedMD 的核心去噪网络（U-Net 架构，条件去噪）。
    失败时退回 _SimpleUNet3D 以保证管道可以运行（冒烟测试）。

    参数：
      cfg : M4FedConfig 中的模型配置对象

    返回：
      MONAI DiffusionModelUNet 或 _SimpleUNet3D 实例
    """
    try:
        from generative.networks.nets import DiffusionModelUNet   # MONAI 生成模型库
        unet = DiffusionModelUNet(
            spatial_dims=cfg.unet_spatial_dims,          # 3D 空间（处理体积数据）
            in_channels=cfg.unet_in_channels,             # 输入通道数 = VAE latent_channels
            out_channels=cfg.unet_out_channels,           # 输出通道数 = 噪声预测通道数
            num_res_blocks=cfg.unet_num_res_blocks,       # 每个分辨率级别的残差块数
            num_channels=cfg.unet_num_channels,           # 各级别的特征通道数（升序）
            attention_levels=cfg.unet_attention_levels,   # 哪些级别启用自注意力
            norm_num_groups=cfg.unet_norm_num_groups,     # GroupNorm 的组数
            with_conditioning=True,                        # 启用交叉注意力（接受文本条件）
            cross_attention_dim=cfg.unet_cross_attention_dim,  # 文本嵌入维度
        )
        return unet   # 返回构建好的 MONAI U-Net
    except Exception as e:
        # MONAI GenerativeModels 未安装或版本不兼容，退回极简 U-Net
        print(f"[UNet] MONAI DiffusionModelUNet 不可用: {e}。使用极简 UNet 替代。")
        return _SimpleUNet3D(
            in_ch=cfg.unet_in_channels,                  # 匹配 VAE 输出通道数
            out_ch=cfg.unet_out_channels,                 # 匹配噪声预测通道数
            cross_attn_dim=cfg.unet_cross_attention_dim,  # 匹配文本编码维度
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
      即：给定加噪潜向量 z_t 和文本条件 c，U-Net 预测噪声 ε 的均方误差（MSE）
    """

    def __init__(self, model_cfg, device: str = "cpu"):
        """
        参数：
          model_cfg : M4FedConfig 中的模型配置（含 vae_repo、unet_* 等参数）
          device    : 计算设备（"cuda" 或 "cpu"）
        """
        super().__init__()          # 调用 nn.Module 父类初始化
        self.cfg = model_cfg        # 保存模型配置（供内部方法访问超参数）
        self.device_str = device    # 保存设备字符串（字符串形式，供 tensor 创建使用）

        # ---- 冻结组件：不参与训练（requires_grad=False，eval 模式）----
        # VAE：将 MRI 体积编码到潜空间（空间降采样 4 倍）并能从潜向量重建图像
        self.vae = load_vae(
            repo_id=model_cfg.vae_repo,                  # HuggingFace 仓库 ID（MONAI 预训练 VAE）
            device=device,                                # 目标设备
            latent_channels=model_cfg.unet_in_channels,  # 潜空间通道数（与 U-Net 输入对齐）
        )
        # 潜向量缩放因子：将 VAE 潜向量的方差归一化到单位量级（初始值来自配置，训练前用 calibrate 更新）
        self.scale_factor = model_cfg.vae_scale_factor

        # 文本编码器：将元数据文本条件（如 "age is 32"）编码为交叉注意力向量
        self.text_encoder = BioMedCLIPEncoder(
            repo_id=model_cfg.biomedclip_repo,  # BioMedCLIP 模型的 HuggingFace 仓库 ID
            device=device,                       # 目标设备
        )

        # ---- 可训练组件（U-Net 参与联邦训练） ----
        # U-Net：在潜空间中预测去噪方向 ε_θ(z_t, t, c)
        self.unet = _build_monai_unet(model_cfg)   # 尝试 MONAI U-Net，失败则退回极简版

        # DDIM 调度器：管理噪声时间步序列和 DDIM 采样步骤（非 nn.Module，不含可训练参数）
        self.scheduler = DDIMScheduler(
            T=model_cfg.diffusion_timesteps,   # 扩散总步数（训练时随机采样时间步）
            beta_start=model_cfg.beta_start,   # β 调度起始值
            beta_end=model_cfg.beta_end,        # β 调度终止值
        )

    # ---------------------------------------------------------------------- #
    #  训练前向传播：计算 LDM 去噪损失                                            #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        mri: torch.Tensor,    # (B, 1, D, H, W)，值域 [-1, 1]（经 preprocess_mri 处理后）
        texts: List[str],      # 对应的文本条件列表，长度 = B（来自 table_row_to_text）
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
          mri   : 输入 MRI 批次，形状 (B, 1, D, H, W)，值域 [-1, 1]
          texts : 对应的文本条件列表，长度 B

        返回：
          标量损失张量（反向传播的目标）
        """
        B = mri.shape[0]   # 获取 batch size

        # 编码阶段：用冻结 VAE 将 MRI 映射到潜空间，不需要计算梯度
        with torch.no_grad():
            z0 = self._encode(mri)   # (B, lat_ch, d, h, w)，d/h/w = D/H/W // 4

        # 随机采样时间步（每个样本独立采样，增加训练多样性和难度均衡）
        t = torch.randint(0, self.scheduler.T, (B,), device=mri.device)

        # 前向加噪：将 z₀ 加噪到随机时间步 t 处（重参数化技巧，一步直达）
        z_t, eps = self.scheduler.add_noise(z0, t)   # eps 是实际添加的噪声（训练标签）

        # 文本条件：用冻结文本编码器将文本列表转为交叉注意力条件向量
        with torch.no_grad():
            context = self.text_encoder(texts)   # (B, L, D_clip)，L=77（CLIP 序列长度）

        # U-Net 前向传播：预测加到 z₀ 上的噪声 ε（条件：时间步 t + 文本 c）
        eps_pred = self.unet(z_t, t, context)   # (B, lat_ch, d, h, w)

        # 计算 MSE 损失：预测噪声与真实噪声的均方误差（LDM 训练目标）
        loss = nn.functional.mse_loss(eps_pred, eps)
        return loss   # 返回标量损失张量（用于 optimizer.step()）

    # ---------------------------------------------------------------------- #
    #  img2img 推理：从源 MRI 生成条件合成 MRI                                    #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()   # 推理时不需要梯度，节省显存
    def synthesise(
        self,
        source_mri: torch.Tensor,     # (B, 1, D, H, W)，值域 [-1, 1]，源图像
        texts: List[str],              # 目标文本条件（描述合成目标的元数据）
        noise_strength: float = 0.75,  # 加噪强度（越大越不像源图像，越像条件）
        ddim_steps: int = 50,          # DDIM 去噪步数（越多越精细，越慢）
        eta: float = 0.0,              # 随机性系数（0=确定性，>0 引入随机噪声）
    ) -> torch.Tensor:
        """
        img2img 合成：在源 MRI 的基础上，根据目标文本条件生成新的 MRI。

        控制参数：
          noise_strength=0.75：保留源 MRI 约 25% 的结构信息，改变约 75%
          eta=0.0：完全确定性 DDIM 采样（更稳定，可复现）

        流程：
          1. 编码源 MRI 为 z₀（VAE 压缩到潜空间）
          2. 在 t_start = noise_strength×T 处加噪得到 z_t（部分扩散）
          3. 从 t_start 开始 DDIM 去噪 ddim_steps 步，以目标文本为条件
          4. 解码 z₀_hat 为合成图像（VAE 从潜空间重建）

        返回：
          合成 MRI 张量，形状 (B, 1, D, H, W)，值域 [-1, 1]
        """
        self.eval()   # 切换为推理模式（关闭 Dropout 等训练专用行为）
        z0 = self._encode(source_mri)   # 编码源图像为潜向量 (B, lat_ch, d, h, w)

        # 在 t_start 处加噪（部分扩散：只加到强度 strength 对应的噪声量）
        t_start = int(noise_strength * self.scheduler.T)   # 计算目标时间步
        # 构造时间步张量（批内所有样本使用同一 t_start）
        t_tensor = torch.full((z0.shape[0],), t_start - 1,
                              dtype=torch.long, device=z0.device)
        z_t, _ = self.scheduler.add_noise(z0, t_tensor)   # 将 z₀ 加噪到 t_start 处

        # 文本条件：将目标元数据文本编码为交叉注意力向量
        context = self.text_encoder(texts)   # (B, L, D_clip)

        # DDIM 去噪迭代：从 t_start 均匀减小到 0（条件去噪）
        timesteps = self.scheduler.get_timesteps_for_strength(noise_strength, ddim_steps)
        z = z_t   # 初始化为加噪后的潜向量
        for i, t_now in enumerate(timesteps):
            # 确定下一时间步（最后一步的 t_next=0）
            t_next = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0
            # 构造时间步张量（整个 batch 使用同一时间步）
            t_tensor = torch.full((z.shape[0],), int(t_now), dtype=torch.long, device=z.device)
            # U-Net 预测当前时间步的噪声方向
            eps_pred = self.unet(z, t_tensor, context)
            # DDIM 单步去噪：z_t → z_{t_next}
            z = self.scheduler.ddim_step(z, int(t_now), t_next, eps_pred, eta=eta)

        return self._decode(z)   # 解码最终潜向量为合成 MRI 图像

    @torch.no_grad()   # 推理时不需要梯度
    def synthesise_from_label(
        self,
        age_label,                              # 目标年龄（整数或字符串，如 45）
        batch_size: int = 1,                    # 一次生成的图像数量
        mri_shape: Tuple[int, ...] = (96, 96, 96),  # 目标图像的空间分辨率
        noise_strength: float = 0.75,           # 不影响此函数（从完整 T 开始去噪）
        ddim_steps: int = 50,                   # DDIM 去噪步数
    ) -> torch.Tensor:
        """
        纯文本条件生成（GPT-FL 风格）：不需要源 MRI，
        直接从随机噪声出发，以年龄文本为条件生成脑年龄 MRI。

        与 synthesise() 的区别：
          - 起点是随机高斯噪声（而非源 MRI 加噪）
          - 从 T-1 开始完整去噪（而非部分去噪）
          - 仅使用年龄标签作为文本条件（text = "age is {age_label}"）

        用途：GPT-FL 服务器根据客户端共享的年龄标签分布，
        生成具有代表性的合成数据分发给客户端，用于知识传递。

        参数：
          age_label    : 目标年龄（作为文本条件，如 "age is 45"）
          batch_size   : 生成图像数量
          mri_shape    : 目标 MRI 空间分辨率（D, H, W）
          noise_strength: 此函数中未使用（保留接口一致性）
          ddim_steps   : DDIM 去噪步数

        返回：
          合成 MRI 张量，形状 (batch_size, 1, D, H, W)，值域 [-1, 1]
        """
        self.eval()   # 切换为推理模式

        # 计算潜向量的空间尺寸（VAE 每个空间维度降采样 4 倍）
        lat_d = mri_shape[0] // 4   # 深度方向的潜空间尺寸
        lat_h = mri_shape[1] // 4   # 高度方向的潜空间尺寸
        lat_w = mri_shape[2] // 4   # 宽度方向的潜空间尺寸
        lat_ch = self.cfg.unet_in_channels   # 潜空间通道数（= VAE latent_channels）

        # 从标准正态分布采样初始潜向量（纯生成模式，不依赖源图像）
        z = torch.randn(batch_size, lat_ch, lat_d, lat_h, lat_w,
                        device=next(self.unet.parameters()).device)   # 与 U-Net 在同一设备

        # 构造文本条件：批内所有样本使用同一年龄标签
        texts = [f"age is {age_label}"] * batch_size
        context = self.text_encoder(texts)   # (B, L, D_clip)

        # 从 T-1 开始完整去噪（生成模式，而非 img2img 的部分去噪）
        # linspace 从 T-1 到 0，共 ddim_steps 步（等间隔采样时间步）
        timesteps = torch.linspace(self.scheduler.T - 1, 0, ddim_steps, dtype=torch.long)
        for i, t_now in enumerate(timesteps):
            t_next = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0   # 计算下一时间步
            # 构造时间步张量（批内所有样本统一）
            t_tensor = torch.full((z.shape[0],), int(t_now), dtype=torch.long, device=z.device)
            eps_pred = self.unet(z, t_tensor, context)   # U-Net 预测噪声方向
            z = self.scheduler.ddim_step(z, int(t_now), t_next, eps_pred, eta=0.0)   # 确定性去噪

        return self._decode(z)   # 解码最终潜向量为合成 MRI

    # ---------------------------------------------------------------------- #
    #  VAE 缩放因子校准                                                         #
    # ---------------------------------------------------------------------- #

    def calibrate_scale_factor(self, loader, n_batches: int = 8) -> float:
        """
        从 DataLoader 采样若干批次，统计 VAE 潜向量的全局标准差，
        以 1/std 更新 self.scale_factor（使潜向量方差接近 1）。

        应在首次 build_models 后、训练开始前调用一次。
        VAE 保持冻结；不产生梯度。

        参数：
          loader   : 返回 {'mri': Tensor(B,1,D,H,W)} 的 DataLoader
          n_batches: 用于统计的批次数（默认 8，平衡精度与速度）

        返回：
          更新后的 scale_factor（float）
        """
        chunks: list = []   # 收集各批次的潜向量展平值（用于全局统计）
        with torch.no_grad():   # 校准不需要梯度
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break   # 达到指定批次数后停止
                mri = batch["mri"].to(self.device_str)   # 将 MRI 移到计算设备
                try:
                    z, _ = self.vae.encode(mri)   # MONAI AutoencoderKL 返回 (z, posterior)
                except Exception:
                    z = self.vae.encode(mri)        # _MinimalVAE 直接返回 z（兼容两种接口）
                chunks.append(z.float().flatten().cpu())   # 展平为一维，移到 CPU 收集

        if not chunks:
            return self.scale_factor   # 没有有效数据时保持原值（安全处理）

        global_std = torch.cat(chunks).std().item()   # 合并所有批次，计算全局标准差
        if global_std < 1e-6:
            # 标准差过小说明潜向量几乎为零（VAE 未正常工作），跳过校准
            print("[MedMD] 警告：潜向量 std 过小，跳过校准，保留原 scale_factor。")
            return self.scale_factor

        new_scale = 1.0 / global_std   # 缩放因子 = 1/std，使潜向量方差 ≈ 1
        old_scale = self.scale_factor   # 记录旧值用于日志对比
        self.scale_factor = new_scale   # 更新为新的缩放因子
        print(
            f"[MedMD] VAE scale_factor 已校准: {old_scale:.5f} → {new_scale:.5f}"
            f"（潜向量 std={global_std:.4f}，共 {len(chunks)} 个批次）"
        )
        return new_scale   # 返回更新后的 scale_factor

    # ---------------------------------------------------------------------- #
    #  VAE 编码/解码辅助方法                                                     #
    # ---------------------------------------------------------------------- #

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 VAE 编码器将 MRI 映射到潜空间，并乘以缩放因子归一化方差。

        兼容两种 VAE 接口：
          - MONAI AutoencoderKL：encode() 返回 (z, posterior)，取第一个元素
          - _MinimalVAE：encode() 直接返回 z（无 posterior 对象）

        参数：
          x : MRI 批次张量 (B, 1, D, H, W)，值域 [-1, 1]

        返回：
          缩放后的潜向量 (B, lat_ch, d, h, w)
        """
        try:
            z, _ = self.vae.encode(x)      # MONAI VAE：返回 (z, posterior)
            return z * self.scale_factor    # 乘以缩放因子使方差 ≈ 1
        except Exception:
            return self.vae.encode(x) * self.scale_factor   # MinimalVAE：直接返回 z

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        使用 VAE 解码器将潜向量重建为 MRI 体积，并除以缩放因子还原尺度。

        兼容两种 VAE 接口：
          - MONAI AutoencoderKL：decode() 直接返回重建图像
          - _MinimalVAE：forward() 返回 (output, None)（无独立 decode 方法）

        参数：
          z : 缩放后的潜向量 (B, lat_ch, d, h, w)

        返回：
          重建的 MRI 张量 (B, 1, D, H, W)，值域 [-1, 1]
        """
        z = z / self.scale_factor   # 除以缩放因子还原 VAE 原始尺度
        try:
            return self.vae.decode(z)   # MONAI VAE：直接调用 decode()
        except Exception:
            out, _ = self.vae(z)         # MinimalVAE：调用 forward() 取第一个返回值
            return out

    # ---------------------------------------------------------------------- #
    #  权重访问接口（联邦聚合使用）                                                #
    # ---------------------------------------------------------------------- #

    def unet_state_dict(self) -> dict:
        """
        返回 U-Net 的状态字典（深拷贝）。

        仅 U-Net 参与联邦热身阶段的权重分发；
        VAE 和文本编码器始终冻结，其权重不需要在节点间传输。
        """
        return copy.deepcopy(self.unet.state_dict())   # 深拷贝保证外部修改不影响模型

    def load_unet_state_dict(self, state: dict, strict: bool = False) -> None:
        """
        从状态字典恢复 U-Net 权重（strict=False 允许部分加载，兼容不同版本）。

        参数：
          state  : 来自其他节点的 U-Net 状态字典（通常是服务器聚合后的权重）
          strict : 是否严格要求键完全匹配（False 允许缺少或多余的键）
        """
        self.unet.load_state_dict(state, strict=strict)   # 将外部权重加载到本地 U-Net
