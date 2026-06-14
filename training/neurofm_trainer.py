"""
NeuroFM 多任务训练循环。

损失函数：L_MT = Σ_k λ_k · L_k
  - L_k（回归）   = MSE（年龄等连续变量）
  - L_k（分类）   = CrossEntropy（性别等类别变量）
  - λ_k（年龄）   = age_loss_weight（默认 2.0）
  - λ_k（其他）   = 1.0

支持：
  - 全参数微调（use_qlora=False）
  - QLoRA 微调（use_qlora=True）：仅更新骨干网络的 LoRA 参数
  - FedProx 近端正则化（proximal_mu > 0）：L_total = L_MT + (μ/2)‖w - w_global‖²
"""
from __future__ import annotations
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.neurofm.neurofm_multitask import NeuroFMMultiTask
from utils.qlora_utils import apply_qlora, get_qlora_params
from utils.io_utils import save_model


class NeuroFMTrainer:
    """
    封装 NeuroFMMultiTask 的训练和评估逻辑。

    关键设计：
      - 优化骨干网络 + 所有任务头（或仅 LoRA 参数）
      - 支持 FedProx 近端项（通过 proximal_mu 和 global_state 注入）
      - evaluate() 方法专门用于年龄预测评估，返回 numpy 数组以便计算指标
    """

    def __init__(
        self,
        model: NeuroFMMultiTask,
        device: str = "cpu",
        lr: float = 5e-5,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
        use_qlora: bool = False,
        qlora_r: int = 16,
        qlora_alpha: int = 32,
        qlora_dropout: float = 0.05,
        proximal_mu: float = 0.0,      # FedProx 近端系数，0 表示不使用
        global_state: Optional[dict] = None,  # FedProx 需要的全局模型权重
    ):
        """
        参数：
          proximal_mu  : FedProx 近端正则化系数 μ（>0 时启用）
          global_state : FedProx 中的全局模型状态字典（用于计算近端项）
                         与 proximal_mu 配合使用，proximal_mu=0 时忽略此参数
        """
        self.model = model
        self.device = device
        self.grad_clip = grad_clip
        self.proximal_mu = proximal_mu
        self.global_state = global_state
        self.mixed_precision = mixed_precision and (device == "cuda")
        self.scaler = GradScaler() if self.mixed_precision else None

        if use_qlora:
            # 对骨干网络应用 QLoRA（任务头始终全量训练，参数量小）
            self.model.backbone = apply_qlora(
                self.model.backbone,
                r=qlora_r, lora_alpha=qlora_alpha, lora_dropout=qlora_dropout,
            )
            backbone_params = get_qlora_params(self.model.backbone)
        else:
            backbone_params = list(self.model.backbone.parameters())

        # 所有任务头始终全量优化
        head_params = list(self.model.heads.parameters())
        self.optimizer = optim.AdamW(
            backbone_params + head_params, lr=lr, weight_decay=weight_decay
        )

    def _build_targets(self, batch: dict) -> Dict[str, torch.Tensor]:
        """
        从批次数据中构建各任务的目标张量。

        来源优先级：
          1. batch["metadata"]（pd.Series 或 dict 列表）中的对应字段
          2. 若 metadata 不含该字段但 batch 有 "age" 键，则直接使用

        无法解析的值（字符串等）和缺失值统一设为 NaN，
        在 compute_loss 中通过 mask 跳过。
        """
        targets = {}
        for field in self.model.task_specs:
            if "metadata" in batch:
                # 从每个样本的 metadata 中提取目标字段值
                vals = []
                for meta in batch["metadata"]:
                    if hasattr(meta, "get"):
                        v = meta.get(field, float("nan"))
                    elif hasattr(meta, "__getitem__"):
                        try:
                            v = meta[field]
                        except (KeyError, IndexError):
                            v = float("nan")
                    else:
                        v = float("nan")
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        vals.append(float("nan"))
                targets[field] = torch.tensor(vals, dtype=torch.float32, device=self.device)
            elif field == "age" and "age" in batch:
                # metadata 不可用时，直接从 batch["age"] 获取年龄标签
                targets["age"] = batch["age"].to(self.device)
        return targets

    def train_epoch(self, loader: DataLoader) -> float:
        """
        训练一个 epoch，返回该 epoch 的平均多任务损失（含近端正则项）。

        FedProx 近端项：(μ/2) ‖w - w_global‖²
          防止本地模型与全局模型偏离过远，提升联邦训练稳定性（尤其在 Non-IID 数据下）。
        """
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            mri = batch["mri"].to(self.device)
            targets = self._build_targets(batch)
            # 确保年龄目标始终可用（部分批次可能从 metadata 中提取失败）
            if "age" in batch and "age" not in targets:
                targets["age"] = batch["age"].to(self.device)

            self.optimizer.zero_grad()
            if self.mixed_precision:
                with autocast():
                    preds = self.model(mri)
                    loss = self.model.compute_loss(preds, targets)
                    loss = loss + self._proximal_term()   # FedProx 近端项
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(mri)
                loss = self.model.compute_loss(preds, targets)
                loss = loss + self._proximal_term()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    def _proximal_term(self) -> torch.Tensor:
        """
        计算 FedProx 近端正则化项：(μ/2) Σ_i ‖w_i - w_global_i‖²

        当 proximal_mu == 0 或 global_state 为 None 时，返回零张量（无效果）。
        对每个参数逐元素计算当前权重与全局权重的平方差，然后求和。
        """
        if self.proximal_mu == 0 or self.global_state is None:
            return torch.tensor(0.0)
        term = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.global_state:
                g = self.global_state[name].to(self.device)
                term = term + ((param - g) ** 2).sum()
        return (self.proximal_mu / 2.0) * term

    def train(
        self,
        loader: DataLoader,
        n_epochs: int,
        save_path: Optional[str] = None,
    ) -> List[float]:
        """
        训练多个 epoch，返回每个 epoch 的平均损失列表。

        参数：
          save_path : 训练结束后保存模型权重的路径（可选）
        """
        losses = []
        for epoch in range(n_epochs):
            loss = self.train_epoch(loader)
            losses.append(loss)
        if save_path:
            save_model(self.model, save_path)
        return losses

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        在给定数据集上评估年龄预测性能。

        处理细节：
          - 自动跳过 age 标签为 NaN 的样本
          - 返回 numpy 数组（便于与 compute_domain_metrics 等函数配合）

        返回：
          (y_true, y_pred) - 真实年龄和预测年龄的 numpy 数组（过滤了 NaN）
        """
        self.model.eval()
        y_true, y_pred = [], []
        for batch in loader:
            mri = batch["mri"].to(self.device)
            preds = self.model(mri)
            # 取年龄预测头的输出，压缩最后一维（(B, 1) → (B,)）
            age_pred = preds["age"].squeeze(-1).cpu().numpy()
            age_true = batch["age"].numpy()
            # 过滤 NaN 标签（无年龄信息的样本）
            valid = ~np.isnan(age_true)
            y_true.extend(age_true[valid].tolist())
            y_pred.extend(age_pred[valid].tolist())
        return np.array(y_true), np.array(y_pred)
