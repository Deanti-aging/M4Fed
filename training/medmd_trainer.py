"""
MedMD 训练循环。

损失函数：L_LDM = E[‖ ε - ε_θ(z_t, t, c) ‖²]（潜扩散模型去噪 MSE 损失）

支持两种训练模式：
  - 全参数微调（use_qlora=False）：更新 U-Net 所有参数，适合热身和服务器端训练
  - QLoRA 微调（use_qlora=True） ：仅更新 LoRA 适配器参数，适合客户端资源受限场景
"""
from __future__ import annotations   # 允许注解中使用尚未定义的类名
import os                             # 标准库：路径操作（供 ensure_dir 等使用）
import torch                          # PyTorch 深度学习框架
import torch.optim as optim           # 优化器模块（AdamW）
from torch.amp import GradScaler, autocast  # 混合精度训练工具（FP16/BF16）
from torch.utils.data import DataLoader           # PyTorch 数据加载器
from typing import Dict, List, Optional           # 类型注解工具

from models.medmd.medmd import MedMD              # MedMD 生成模型
from utils.qlora_utils import apply_qlora, get_qlora_params  # QLoRA 工具函数
from utils.io_utils import save_model, ensure_dir  # 文件保存工具


class MedMDTrainer:
    """
    封装 MedMD 的训练、推理（合成数据生成）逻辑。

    关键设计：
      - 只优化 U-Net（VAE 和文本编码器始终冻结，不参与反向传播）
      - 支持 CUDA 混合精度（FP16）减少显存占用
      - synthesise_dataset() 方法用于生成合成样本供联邦通信使用
    """

    def __init__(
        self,
        model: MedMD,            # 待训练的 MedMD 实例
        device: str = "cpu",     # 计算设备（"cuda" 或 "cpu"）
        lr: float = 1e-4,        # AdamW 初始学习率（扩散模型通常使用较小的学习率）
        weight_decay: float = 1e-5,   # 权重衰减系数（L2 正则化）
        grad_clip: float = 1.0,       # 梯度裁剪阈值（防止 FP16 梯度爆炸）
        mixed_precision: bool = True,  # 是否启用 FP16 混合精度（仅 CUDA 有效）
        use_qlora: bool = False,       # 是否对 U-Net 应用 QLoRA（客户端微调时使用）
        qlora_r: int = 16,             # LoRA 低秩矩阵的秩（秩越小参数越少）
        qlora_alpha: int = 32,         # LoRA 缩放系数（推荐 = 2 × r）
        qlora_dropout: float = 0.05,   # LoRA 层 Dropout 比例（防止 LoRA 过拟合）
    ):
        """
        初始化 MedMD 训练器，配置优化器和混合精度。

        参数：
          model           : 待训练的 MedMD 实例
          device          : 计算设备
          lr              : AdamW 学习率
          weight_decay    : 权重衰减系数
          grad_clip       : 梯度裁剪阈值
          mixed_precision : 是否启用 FP16 混合精度
          use_qlora       : 是否对 U-Net 应用 QLoRA
          qlora_r         : LoRA 低秩矩阵的秩
          qlora_alpha     : LoRA 缩放系数
          qlora_dropout   : LoRA Dropout 比例
        """
        self.model = model          # 保存 MedMD 模型引用
        self.device = device        # 保存计算设备字符串
        self.grad_clip = grad_clip  # 保存梯度裁剪阈值（用于 clip_grad_norm_）
        # 混合精度仅在 CUDA 设备上有意义（CPU 不支持 FP16 加速）
        self.mixed_precision = mixed_precision and (device == "cuda")
        # 若启用混合精度，创建 GradScaler（用于防止 FP16 下溢）；否则为 None
        self.scaler = GradScaler("cuda") if self.mixed_precision else None

        if use_qlora:
            # QLoRA 模式：为 U-Net 中的线性层添加 LoRA 适配器，冻结原始权重
            self.model.unet = apply_qlora(
                self.model.unet,         # 目标模块（U-Net）
                r=qlora_r,               # LoRA 秩（控制适配器的参数量）
                lora_alpha=qlora_alpha,  # LoRA 缩放系数（控制适配器的影响强度）
                lora_dropout=qlora_dropout,  # LoRA Dropout（防过拟合）
            )
            # 仅优化 LoRA 适配器参数（大幅减少可训练参数量，降低通信开销）
            params = get_qlora_params(self.model.unet)
        else:
            # 全参数微调：优化 U-Net 的所有参数（适合热身和服务器端）
            params = list(self.model.unet.parameters())

        # 初始化 AdamW 优化器（含 L2 正则化）
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def train_epoch(self, loader: DataLoader) -> float:
        """
        训练一个 epoch，返回该 epoch 的平均 LDM 损失。

        仅 U-Net 进入训练模式（调用 unet.train()）；
        VAE 和文本编码器保持 eval 模式（由 MedMD.__init__ 中的冻结保证）。

        混合精度训练流程（当 self.mixed_precision=True 时）：
          1. autocast()：前向传播自动使用 FP16（减少显存，加速计算）
          2. scaler.scale(loss).backward()：缩放梯度以防 FP16 下溢
          3. scaler.unscale_()：还原梯度到 FP32 量级（用于梯度裁剪）
          4. clip_grad_norm_()：梯度裁剪（防止梯度爆炸）
          5. scaler.step()：更新参数（若梯度有溢出则跳过本步）
          6. scaler.update()：自动调整缩放因子（下次迭代使用）

        参数：
          loader : 训练数据 DataLoader（需含 'mri' 和 'text' 字段）

        返回：
          该 epoch 的平均 LDM 损失（标量 float）
        """
        self.model.unet.train()   # 仅 U-Net 进入训练模式（开启 Dropout 等训练行为）
        total_loss = 0.0          # 累计损失（用于计算平均）
        n = 0                     # 批次计数（防止空 DataLoader 时除零）
        for batch in loader:
            mri = batch["mri"].to(self.device)   # 将 MRI 张量移到计算设备
            texts = batch["text"]                 # 文本条件列表（字符串，不需要移设备）

            self.optimizer.zero_grad()   # 清空上一步的梯度（必须在每步前执行）
            if self.mixed_precision:
                # FP16 混合精度训练
                with autocast("cuda"):   # autocast 自动选择合适的精度（FP16/FP32）
                    loss = self.model(mri, texts)   # 前向传播：计算 LDM 损失
                self.scaler.scale(loss).backward()   # 缩放损失后反向传播
                self.scaler.unscale_(self.optimizer)  # 还原梯度量级（为梯度裁剪准备）
                torch.nn.utils.clip_grad_norm_(
                    self.model.unet.parameters(), self.grad_clip)   # 梯度裁剪（防爆炸）
                self.scaler.step(self.optimizer)   # 参数更新（溢出时自动跳过）
                self.scaler.update()               # 更新缩放因子
            else:
                # 标准 FP32 训练
                loss = self.model(mri, texts)   # 前向传播：计算 LDM 损失
                loss.backward()                 # 反向传播：计算梯度
                torch.nn.utils.clip_grad_norm_(
                    self.model.unet.parameters(), self.grad_clip)   # 梯度裁剪
                self.optimizer.step()           # 参数更新

            total_loss += loss.item()   # 累加本批次损失（.item() 转为 Python 标量）
            n += 1                      # 批次计数 +1

        return total_loss / max(n, 1)   # 返回平均损失（max(n,1) 避免空 DataLoader 时除零）

    def train(
        self,
        loader: DataLoader,
        n_epochs: int,                   # 训练轮数
        save_path: Optional[str] = None, # 训练结束后的模型保存路径（可选）
    ) -> List[float]:
        """
        训练多个 epoch，返回每个 epoch 的平均损失列表。

        参数：
          loader    : 训练数据 DataLoader
          n_epochs  : 训练 epoch 数
          save_path : 若不为 None，训练结束后将整个 MedMD 模型保存到该路径

        返回：
          每个 epoch 的平均损失列表（长度 = n_epochs）
        """
        losses = []   # 收集每个 epoch 的损失
        for epoch in range(n_epochs):
            loss = self.train_epoch(loader)   # 训练 1 epoch
            losses.append(loss)               # 记录该 epoch 的损失
        if save_path:
            save_model(self.model, save_path)  # 保存模型（若指定了路径）
        return losses   # 返回损失列表

    @torch.no_grad()   # 推理阶段不需要梯度（节省显存和计算）
    def synthesise_dataset(
        self,
        source_loader: DataLoader,      # 源数据 DataLoader（来自本节点私有数据）
        n_samples: int,                  # 目标合成样本数（达到数量后停止）
        noise_strength: float = 0.75,   # img2img 噪声强度（越大差异越大）
        ddim_steps: int = 50,           # DDIM 采样步数（越多越精细，越慢）
    ) -> List[dict]:
        """
        从源数据集生成合成样本，用于联邦通信中的数据共享。

        使用 img2img 合成模式：
          对每个源批次中的 MRI，以对应文本条件进行条件生成，
          生成与源 MRI 结构相似但细节不同的合成 MRI（隐私保护效果）。

        参数：
          source_loader  : 源数据的 DataLoader（来自本节点私有数据）
          n_samples      : 要生成的目标样本数（达到数量后停止，避免过多合成）
          noise_strength : img2img 噪声强度（越大与源图像差异越大）
          ddim_steps     : DDIM 去噪步数（推荐 50，质量与速度的平衡点）

        返回：
          合成样本字典列表，每个字典与 OpenMindDataset 样本格式兼容：
          {'mri_array': ndarray, 'table_text': str, 'age': float, 'metadata': dict}
        """
        self.model.eval()   # 切换为推理模式（关闭 Dropout 等训练专用行为）
        synthetic = []       # 收集生成的合成样本
        for batch in source_loader:
            if len(synthetic) >= n_samples:
                break   # 已达到目标数量，停止（避免生成过多消耗时间）
            mri = batch["mri"].to(self.device)   # 源 MRI 张量移到计算设备
            texts = batch["text"]                 # 源文本条件列表（保留，用于合成条件）
            ages = batch["age"]                   # 源年龄标签（保留，作为合成样本的年龄标签）
            metadata = batch["metadata"]           # 源元数据（保留，作为合成样本的元数据）

            # img2img 条件合成：源 MRI + 源文本条件 → 合成 MRI
            syn_mri = self.model.synthesise(
                mri, texts,                     # 源图像和文本条件
                noise_strength=noise_strength,  # 控制合成图像与源图像的相似度
                ddim_steps=ddim_steps,          # DDIM 去噪步数
            )
            # 将批次中的每个样本拆解为独立的字典（逐样本存储）
            for i in range(syn_mri.shape[0]):
                if len(synthetic) >= n_samples:
                    break   # 批次内达到目标数量则提前退出
                synthetic.append({
                    "mri_array": syn_mri[i].cpu().numpy(),    # 合成 MRI 转 NumPy（移到 CPU）
                    "table_text": texts[i],                   # 对应的文本条件（原始文本）
                    # 保留原始年龄标签（NaN 时设为 None，与 OpenMindDataset 约定一致）
                    "age": float(ages[i]) if not torch.isnan(ages[i]) else None,
                    # 元数据：若 metadata 是列表则取第 i 个；否则用空字典（防御性处理）
                    "metadata": metadata[i] if isinstance(metadata, list) else {},
                })
        return synthetic   # 返回合成样本字典列表
