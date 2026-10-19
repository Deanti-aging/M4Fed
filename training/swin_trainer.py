"""
SwinUNETR 多任务训练循环（领域模型：脑年龄预测 + 辅助分类任务）。

损失函数：L_MT = Σ_k λ_k · L_k
  - L_k（回归）   = MSE（年龄等连续变量）
  - L_k（分类）   = CrossEntropy（性别等类别变量）
  - λ_k（年龄）   = age_loss_weight（默认 2.0，加大年龄预测的比重）
  - λ_k（其他）   = 1.0

支持：
  - 全参数微调（use_qlora=False）
  - QLoRA 微调（use_qlora=True）：仅更新骨干网络的 LoRA 参数
  - FedProx 近端正则化（proximal_mu > 0）：L_total = L_MT + (μ/2)‖w - w_global‖²
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import torch                           # PyTorch 深度学习框架
import torch.optim as optim            # 优化器（AdamW）
from torch.amp import GradScaler, autocast  # 混合精度训练工具
from torch.utils.data import DataLoader           # PyTorch 数据加载器
from typing import Dict, List, Optional, Tuple    # 类型注解工具

import numpy as np   # NumPy：NaN 过滤、数组操作（evaluate 返回 numpy 数组）

from models.swin.swin_multitask import SwinMultiTask   # SwinUNETR 多任务模型
from utils.qlora_utils import apply_qlora, get_qlora_params      # QLoRA 工具函数
from utils.io_utils import save_model                            # 模型保存工具


class SwinTrainer:
    """
    封装 SwinMultiTask 的训练和评估逻辑。

    关键设计：
      - 优化骨干网络 + 所有任务头（或仅 LoRA 参数 + 任务头）
      - 支持 FedProx 近端项（通过 proximal_mu 和 global_state 注入）
      - evaluate() 方法专门用于年龄预测评估，返回 numpy 数组以便计算指标
    """

    def __init__(
        self,
        model: SwinMultiTask,            # 待训练的 SwinUNETR 多任务模型
        device: str = "cpu",             # 计算设备（"cuda" 或 "cpu"）
        lr: float = 5e-5,                # AdamW 初始学习率（SwinUNETR 微调通常用小学习率）
        weight_decay: float = 1e-5,      # 权重衰减系数（L2 正则化）
        grad_clip: float = 1.0,          # 梯度裁剪阈值（防止梯度爆炸）
        mixed_precision: bool = True,    # 是否启用 FP16 混合精度（仅 CUDA 有效）
        use_qlora: bool = False,         # 是否对骨干网络应用 QLoRA
        qlora_r: int = 16,               # LoRA 低秩矩阵的秩
        qlora_alpha: int = 32,           # LoRA 缩放系数
        qlora_dropout: float = 0.05,     # LoRA Dropout 比例
        proximal_mu: float = 0.0,        # FedProx 近端系数（0 = 不使用，退化为 FedAvg）
        global_state: Optional[dict] = None,  # FedProx 需要的全局模型权重（近端项锚点）
        age_norm: Optional[Tuple[float, float]] = None,  # 年龄标签 z-score 的 (mean, std)；None = 不标准化
    ):
        """
        初始化训练器，配置优化器、QLoRA 和 FedProx 近端项。

        参数：
          proximal_mu  : FedProx 近端正则化系数 μ（>0 时启用 FedProx）
          global_state : FedProx 中的全局模型状态字典（与 proximal_mu 配合使用，0 时忽略）
          age_norm     : (mean, std)。给定时训练目标改为 (age - mean) / std，evaluate 时把
                         模型输出反变换回原始年龄尺度 pred * std + mean；None 时行为与原来完全一致
        """
        self.model = model                   # 保存 SwinMultiTask 模型引用
        self.device = device                 # 保存计算设备字符串
        self.grad_clip = grad_clip           # 保存梯度裁剪阈值
        self.proximal_mu = proximal_mu       # 保存近端系数（0 = 禁用近端项）
        self.global_state = global_state     # 保存全局权重（用于计算近端项中的 w_global）
        self.age_norm = age_norm             # 年龄 z-score 参数（None = 原始年龄尺度）
        # 混合精度仅在 CUDA 设备上有效（CPU 不支持 FP16 加速）
        self.mixed_precision = mixed_precision and (device == "cuda")
        # 创建 GradScaler（FP16 梯度缩放，防止下溢）；仅在混合精度模式下使用
        self.scaler = GradScaler("cuda") if self.mixed_precision else None

        if use_qlora:
            # QLoRA 模式：仅对骨干网络的线性层添加 LoRA 适配器（任务头始终全量训练）
            self.model.backbone = apply_qlora(
                self.model.backbone,             # 目标模块（SwinUNETR 骨干）
                r=qlora_r,                       # LoRA 秩（秩越小参数越少，节省通信）
                lora_alpha=qlora_alpha,          # LoRA 缩放系数
                lora_dropout=qlora_dropout,      # LoRA Dropout
            )
            # 骨干网络：只优化 LoRA 适配器参数（冻结原始权重）
            backbone_params = get_qlora_params(self.model.backbone)
        else:
            # 全参数微调：优化骨干网络的所有参数
            backbone_params = list(self.model.backbone.parameters())

        # 任务头始终全量优化（参数量小，不用 QLoRA）
        head_params = list(self.model.heads.parameters())
        # 合并骨干参数和任务头参数，统一用 AdamW 优化
        self.optimizer = optim.AdamW(
            backbone_params + head_params,   # 合并两个参数组
            lr=lr,                           # 统一学习率
            weight_decay=weight_decay        # 权重衰减（L2 正则化）
        )

    def _build_targets(self, batch: dict) -> Dict[str, torch.Tensor]:
        """
        从批次数据中构建各任务的目标张量（标签）。

        批次字典里同时保存了 age 和 metadata 两个键（还有 mri、text）
        来源优先级：
          1. batch["metadata"]（dict 或 pd.Series 列表）中对应字段名的值
          2. 若 metadata 不含该字段但 batch 含 "age" 键（向后兼容），直接使用

        无法解析的值（字符串、None 等）和缺失值统一设为 NaN，
        在 compute_loss 中通过 NaN mask 跳过这些样本（不参与损失计算）。

        参数：
          batch : DataLoader 返回的批次字典

        返回：
          {task_name: tensor(B,)} 格式的目标字典（float32）
        """
        targets = {}
        for field in self.model.task_specs:   # 遍历所有预测任务字段
            if "metadata" in batch:
                # 从每个样本的 metadata 中提取目标字段值
                vals = []
                for meta in batch["metadata"]:
                    if hasattr(meta, "get"):
                        v = meta.get(field, float("nan"))   # dict-like：尝试 get 方法
                    elif hasattr(meta, "__getitem__"):
                        try:
                            v = meta[field]   # 支持下标访问（如 pd.Series）
                        except (KeyError, IndexError):
                            v = float("nan")   # 字段不存在时设为 NaN
                    else:
                        v = float("nan")   # 不支持访问时设为 NaN
                    try:
                        vals.append(float(v))   # 转 float（字符串等无效值会在 except 中捕获）
                    except (ValueError, TypeError):
                        vals.append(float("nan"))   # 无法转换时设为 NaN（如 "unknown"）
                # 将 float 列表转为 float32 张量并移到计算设备
                targets[field] = torch.tensor(vals, dtype=torch.float32, device=self.device)
            # 逐样本回退：metadata 缺失该字段（如合成数据的 metadata={}）时，
            # 用 batch 中已预堆叠好的同名字段（_collate_fn 产出的 age/health_status/race）补齐，
            # 否则该字段全为 NaN，compute_loss 会把这个任务整批遮蔽，loss 脱离计算图导致 backward() 报错
            if field in batch:
                batch_vals = batch[field].to(self.device)
                if field in targets:
                    nan_mask = torch.isnan(targets[field])
                    if nan_mask.any():
                        targets[field] = torch.where(nan_mask, batch_vals, targets[field])
                else:
                    targets[field] = batch_vals
        return targets   # 返回各任务的目标张量字典

    def train_epoch(self, loader: DataLoader) -> float:
        """
        训练一个 epoch，返回该 epoch 的平均多任务损失（含近端正则项）。

        FedProx 近端正则化项（当 proximal_mu > 0 时）：
          (μ/2) ‖w - w_global‖²
          惩罚本地模型与全局模型偏离过远，在 Non-IID 数据下提升稳定性。

        参数：
          loader : 训练数据 DataLoader

        返回：
          该 epoch 的平均多任务总损失（标量 float，含近端项）
        """
        self.model.train()    # 切换为训练模式（开启 BatchNorm、Dropout 等）
        total_loss = 0.0      # 累计总损失
        n = 0                 # 批次计数
        for batch in loader:
            mri = batch["mri"].to(self.device)   # 将 MRI 张量移到计算设备
            targets = self._build_targets(batch)  # 从批次数据构建目标标签字典
            # 确保年龄目标始终可用（部分批次 metadata 中提取失败时的兜底）
            if "age" in batch and "age" not in targets:
                targets["age"] = batch["age"].to(self.device)
            if self.age_norm is not None and "age" in targets:
                mean, std = self.age_norm
                targets["age"] = (targets["age"] - mean) / std   # NaN 保持 NaN，仍被 mask 跳过

            self.optimizer.zero_grad()   # 清空梯度（每步前必须执行）
            if self.mixed_precision:
                # FP16 混合精度训练
                with autocast("cuda"):   # 前向传播使用 FP16
                    preds = self.model(mri)                       # 骨干特征提取 + 任务头预测
                    loss = self.model.compute_loss(preds, targets)  # 多任务加权损失
                    loss = loss + self._proximal_term()           # 加上 FedProx 近端项
                self.scaler.scale(loss).backward()                # 缩放损失后反向传播
                self.scaler.unscale_(self.optimizer)              # 还原梯度量级（为裁剪准备）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # 梯度裁剪
                self.scaler.step(self.optimizer)   # 参数更新（溢出时自动跳过）
                self.scaler.update()               # 更新缩放因子
            else:
                # 标准 FP32 训练
                preds = self.model(mri)                        # 前向传播
                loss = self.model.compute_loss(preds, targets)  # 多任务损失
                loss = loss + self._proximal_term()            # FedProx 近端项
                loss.backward()                                 # 反向传播
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # 梯度裁剪
                self.optimizer.step()                           # 参数更新

            total_loss += loss.item()   # 累加本批次损失
            n += 1                      # 批次计数 +1

        return total_loss / max(n, 1)   # 返回平均损失（max(n,1) 防止空 DataLoader 时除零）

    def _proximal_term(self) -> torch.Tensor:
        """
        计算 FedProx 近端正则化项：(μ/2) Σ_i ‖w_i - w_global_i‖²

        设计：
          - 当 proximal_mu == 0 或 global_state 为 None 时，返回零张量（不影响梯度）
          - 对每个参数逐元素计算当前权重与全局权重的平方差，然后求和再乘以 μ/2

        返回：
          FedProx 近端正则化项（标量张量）；proximal_mu=0 时返回 torch.tensor(0.0)
        """
        if self.proximal_mu == 0 or self.global_state is None:
            return torch.tensor(0.0)   # 不使用 FedProx 时返回零（不影响损失）
        term = torch.tensor(0.0, device=self.device)   # 初始化近端项（在计算设备上）
        for name, param in self.model.named_parameters():
            if name in self.global_state:
                g = self.global_state[name].to(self.device)   # 全局权重移到计算设备
                term = term + ((param - g) ** 2).sum()         # 累加各参数的平方差之和
        return (self.proximal_mu / 2.0) * term   # 乘以 μ/2，得到最终近端项

    def train(
        self,
        loader: DataLoader,
        n_epochs: int,                    # 训练轮数
        save_path: Optional[str] = None,  # 训练结束后的模型保存路径（可选）
    ) -> List[float]:
        """
        训练多个 epoch，返回每个 epoch 的平均损失列表。

        参数：
          loader    : 训练数据 DataLoader
          n_epochs  : 本地训练 epoch 数
          save_path : 若不为 None，训练结束后将模型保存到该路径

        返回：
          每个 epoch 的平均损失列表（长度 = n_epochs）
        """
        losses = []   # 收集每个 epoch 的平均损失
        for epoch in range(n_epochs):
            loss = self.train_epoch(loader)   # 训练 1 epoch
            losses.append(loss)               # 记录该 epoch 的损失
        if save_path:
            save_model(self.model, save_path)  # 保存模型（若指定了路径）
        return losses   # 返回损失列表

    @torch.no_grad()   # 评估阶段不需要梯度（节省显存）
    def evaluate(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        在给定数据集上评估年龄预测性能（推理模式，不更新参数）。

        处理细节：
          - 只评估年龄（brain age prediction）任务，忽略其他任务头
          - 自动过滤年龄标签为 NaN 的样本（无标签样本不参与评估）
          - 返回 numpy 数组（便于与 scipy/sklearn 的统计指标函数配合）

        参数：
          loader : 测试数据 DataLoader

        返回：
          (y_true, y_pred) 元组
            - y_true : 真实年龄的 numpy 数组（过滤了 NaN）
            - y_pred : 对应的预测年龄 numpy 数组
        """
        self.model.eval()   # 切换为推理模式（关闭 Dropout、BatchNorm 使用 running stats）
        y_true, y_pred = [], []   # 收集真值和预测值
        for batch in loader:
            mri = batch["mri"].to(self.device)   # MRI 移到计算设备
            preds = self.model(mri)               # 前向传播，返回各任务预测字典
            # 取年龄预测头的输出，squeeze(-1) 将 (B,1) 压缩为 (B,)
            age_pred = preds["age"].squeeze(-1).float().cpu().numpy()
            if self.age_norm is not None:
                age_pred = age_pred * self.age_norm[1] + self.age_norm[0]   # z-score → 原始年龄
            age_true = batch["age"].numpy()   # 真实年龄（可能含 NaN）
            # 过滤 NaN 标签：np.isnan 返回布尔数组，~ 取反得到有效样本掩膜
            valid = ~np.isnan(age_true)
            y_true.extend(age_true[valid].tolist())   # 只保留有效真值
            y_pred.extend(age_pred[valid].tolist())   # 只保留对应预测值
        return np.array(y_true), np.array(y_pred)   # 转为 numpy 数组返回
