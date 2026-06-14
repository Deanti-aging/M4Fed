"""
MedMD 训练循环。

损失函数：L_LDM = E[‖ ε - ε_θ(z_t, t, c) ‖²]（潜扩散模型去噪 MSE 损失）

支持两种训练模式：
  - 全参数微调（use_qlora=False）：更新 U-Net 所有参数，适合热身和服务器端训练
  - QLoRA 微调（use_qlora=True） ：仅更新 LoRA 适配器参数，适合客户端资源受限场景
"""
from __future__ import annotations
import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from models.medmd.medmd import MedMD
from utils.qlora_utils import apply_qlora, get_qlora_params
from utils.io_utils import save_model, ensure_dir


class MedMDTrainer:
    """
    封装 MedMD 的训练、推理（合成数据生成）逻辑。

    关键设计：
      - 只优化 U-Net（VAE 和文本编码器始终冻结）
      - 支持 CUDA 混合精度（FP16）减少显存使用
      - synthesise_dataset() 方法用于生成合成样本供联邦通信使用
    """

    def __init__(
        self,
        model: MedMD,
        device: str = "cpu",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
        use_qlora: bool = False,
        qlora_r: int = 16,
        qlora_alpha: int = 32,
        qlora_dropout: float = 0.05,
    ):
        """
        参数：
          model           : 待训练的 MedMD 实例
          device          : 计算设备
          lr              : AdamW 学习率
          weight_decay    : 权重衰减系数
          grad_clip       : 梯度裁剪阈值（防止梯度爆炸）
          mixed_precision : 是否启用 FP16 混合精度（仅 CUDA 有效）
          use_qlora       : 是否对 U-Net 应用 QLoRA（客户端微调时使用）
          qlora_r         : LoRA 低秩矩阵的秩
          qlora_alpha     : LoRA 缩放系数
          qlora_dropout   : LoRA 层 Dropout 比例
        """
        self.model = model
        self.device = device
        self.grad_clip = grad_clip
        # 混合精度仅在 CUDA 设备上有意义
        self.mixed_precision = mixed_precision and (device == "cuda")
        self.scaler = GradScaler() if self.mixed_precision else None

        if use_qlora:
            # 应用 QLoRA：为 U-Net 添加 LoRA 适配器层，冻结原始权重
            self.model.unet = apply_qlora(
                self.model.unet, r=qlora_r, lora_alpha=qlora_alpha,
                lora_dropout=qlora_dropout,
            )
            # 只优化 LoRA 参数（大幅减少可训练参数量）
            params = get_qlora_params(self.model.unet)
        else:
            # 全参数微调：优化 U-Net 的所有参数
            params = list(self.model.unet.parameters())

        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def train_epoch(self, loader: DataLoader) -> float:
        """
        训练一个 epoch，返回该 epoch 的平均 LDM 损失。

        仅 U-Net 进入训练模式（VAE 和文本编码器的 eval 模式不受影响）。

        混合精度训练流程：
          1. autocast()：前向传播自动使用 FP16
          2. scaler.scale(loss).backward()：缩放梯度以防 FP16 下溢
          3. scaler.unscale_()：还原梯度到原始量级（用于梯度裁剪）
          4. clip_grad_norm_()：梯度裁剪
          5. scaler.step()：更新参数
          6. scaler.update()：调整缩放因子
        """
        self.model.unet.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            mri = batch["mri"].to(self.device)
            texts = batch["text"]

            self.optimizer.zero_grad()
            if self.mixed_precision:
                with autocast():
                    loss = self.model(mri, texts)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.unet.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.model(mri, texts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.unet.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)   # 避免空 DataLoader 时的除零

    def train(
        self,
        loader: DataLoader,
        n_epochs: int,
        save_path: Optional[str] = None,
    ) -> List[float]:
        """
        训练多个 epoch，返回每个 epoch 的平均损失列表。

        参数：
          save_path : 若不为 None，训练结束后将模型保存到该路径
        """
        losses = []
        for epoch in range(n_epochs):
            loss = self.train_epoch(loader)
            losses.append(loss)
        if save_path:
            save_model(self.model, save_path)
        return losses

    @torch.no_grad()
    def synthesise_dataset(
        self,
        source_loader: DataLoader,
        n_samples: int,
        noise_strength: float = 0.75,
        ddim_steps: int = 50,
    ) -> List[dict]:
        """
        从源数据集生成合成样本，用于联邦通信中的数据共享。

        使用 img2img 模式：
          对每个源批次中的 MRI，以对应文本条件进行条件生成，
          生成与源 MRI 结构相似但细节不同的合成 MRI。

        参数：
          source_loader  : 源数据的 DataLoader（来自本节点私有数据）
          n_samples      : 要生成的目标样本数（达到数量后停止）
          noise_strength : img2img 噪声强度（越大差异越大）
          ddim_steps     : DDIM 采样步数

        返回：
          合成样本字典列表，每个字典与 OpenMindDataset 样本格式兼容：
          {'mri_array': ndarray, 'table_text': str, 'age': float, 'metadata': dict}
        """
        self.model.eval()
        synthetic = []
        for batch in source_loader:
            if len(synthetic) >= n_samples:
                break
            mri = batch["mri"].to(self.device)
            texts = batch["text"]
            ages = batch["age"]
            metadata = batch["metadata"]

            # 条件 img2img 合成
            syn_mri = self.model.synthesise(
                mri, texts,
                noise_strength=noise_strength,
                ddim_steps=ddim_steps,
            )
            # 将批次中的每个样本拆解为独立的字典
            for i in range(syn_mri.shape[0]):
                if len(synthetic) >= n_samples:
                    break
                synthetic.append({
                    "mri_array": syn_mri[i].cpu().numpy(),
                    "table_text": texts[i],
                    # 保留原始年龄标签（NaN 时设为 None，与 OpenMindDataset 约定一致）
                    "age": float(ages[i]) if not torch.isnan(ages[i]) else None,
                    "metadata": metadata[i] if isinstance(metadata, list) else {},
                })
        return synthetic
