"""
NeuroFM 多任务预测模型：骨干网络 + 多个任务头。

主任务：脑年龄预测（回归，MSE 损失，权重加倍）。
附加任务：其他元数据字段（性别分类等）。

加载优先级：
  1. 从 HuggingFace 加载 NeuroFM 预训练骨干网络（transformer 架构）
  2. 若无法加载，退回轻量级 3D CNN 骨干（_Backbone3DCNN）

设计原则：
  - 骨干网络提取 MRI 的全局嵌入向量
  - 每个预测任务有独立的 MLP 任务头（避免任务间干扰）
  - 多任务损失：L = Σ_k λ_k · L_k（年龄任务权重加倍）
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
#  退回骨干网络：轻量级 3D CNN                                                    #
# --------------------------------------------------------------------------- #

class _Backbone3DCNN(nn.Module):
    """
    当 NeuroFM 不可用时使用的轻量级 3D CNN 骨干网络。

    网络结构：
      3 个 3D 卷积块（Conv3d + BN + GELU）+ 全局平均池化 + 全连接层
      特征通道数：1 → 16 → 32 → 64 → embed_dim
      每次卷积步长为 2，空间分辨率逐步降为 1/8

    输出：形状 (B, embed_dim) 的全局嵌入向量
    """

    def __init__(self, in_ch: int = 1, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # 第 1 块：空间分辨率 × 1/2
            nn.Conv3d(in_ch, 16, 3, stride=2, padding=1), nn.BatchNorm3d(16), nn.GELU(),
            # 第 2 块：空间分辨率 × 1/4
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.BatchNorm3d(32), nn.GELU(),
            # 第 3 块：空间分辨率 × 1/8
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.GELU(),
            # 全局平均池化：将任意空间尺寸压缩为 (B, 64, 1, 1, 1)
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            # 全连接映射到 embed_dim
            nn.Linear(64, embed_dim),
            nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, 1, D, H, W)，输出 (B, embed_dim)。"""
        return self.net(x)


def _load_neurofm_backbone(repo_id: str, device: str) -> Tuple[nn.Module, int]:
    """
    从 HuggingFace 加载 NeuroFM 预训练骨干网络。

    NeuroFM 是在大规模神经影像数据上预训练的 transformer 模型，
    对脑年龄预测等神经影像任务有强大的迁移学习能力。

    失败时退回 _Backbone3DCNN 并打印警告。

    返回：
      (backbone_module, embed_dim) - 骨干网络模块和嵌入维度
    """
    try:
        from transformers import AutoModel, AutoConfig
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        backbone = AutoModel.from_pretrained(repo_id, trust_remote_code=True).to(device)
        embed_dim = getattr(config, "hidden_size", 768)
        return backbone, embed_dim
    except Exception as e:
        print(f"[NeuroFM] 无法从 HuggingFace 加载: {e}。使用退回 CNN 骨干。")
        backbone = _Backbone3DCNN(embed_dim=256)
        return backbone.to(device), 256


# --------------------------------------------------------------------------- #
#  多任务预测头                                                                   #
# --------------------------------------------------------------------------- #

class _TaskHead(nn.Module):
    """
    单任务预测头：两层 MLP，隐藏层维度为输入维度的一半。

    回归任务（如年龄）：输出维度 = 1，损失为 MSE
    分类任务（如性别）：输出维度 = n_classes，损失为 CrossEntropy
    """

    def __init__(self, in_dim: int, out_dim: int, task_type: str = "regression"):
        super().__init__()
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# --------------------------------------------------------------------------- #
#  NeuroFMMultiTask 主体                                                         #
# --------------------------------------------------------------------------- #

class NeuroFMMultiTask(nn.Module):
    """
    NeuroFM 骨干网络 + 每字段独立任务头的多任务预测模型。

    使用方式：
      task_specs = {
        'age': {'type': 'regression', 'n_classes': 1},
        'sex': {'type': 'classification', 'n_classes': 2},
      }
      model = NeuroFMMultiTask(task_specs=task_specs)
      preds = model(mri_tensor)   # {'age': tensor(...), 'sex': tensor(...)}
      loss = model.compute_loss(preds, targets)

    联邦学习中的角色：
      - 该模型是 M4Fed 的"领域模型"，负责从 MRI 预测年龄等临床指标
      - QLoRA 微调时仅更新骨干网络中的 LoRA 参数，大幅减少通信开销
    """

    def __init__(
        self,
        repo_id: str = "NeuroAI-UofG/NeuroFM",
        task_specs: Optional[Dict[str, dict]] = None,
        device: str = "cpu",
        random_init: bool = False,
        age_loss_weight: float = 2.0,
    ):
        """
        参数：
          repo_id          : NeuroFM 的 HuggingFace 仓库 ID
          task_specs       : 任务规格字典（见类说明）
          device           : 计算设备
          random_init      : 若为 True，跳过 NeuroFM 加载，使用随机初始化的 CNN 骨干
                             （消融实验 C 使用，验证预训练骨干的作用）
          age_loss_weight  : 年龄预测任务的损失权重倍数
        """
        super().__init__()
        self.device_str = device
        self.age_loss_weight = age_loss_weight

        # 选择骨干网络（预训练 NeuroFM 或随机初始化 CNN）
        if random_init:
            self.backbone = _Backbone3DCNN(embed_dim=256)
            self.embed_dim = 256
        else:
            self.backbone, self.embed_dim = _load_neurofm_backbone(repo_id, device)

        # 默认只预测年龄（单任务）
        if task_specs is None:
            task_specs = {"age": {"type": "regression", "n_classes": 1}}
        self.task_specs = task_specs

        # 为每个任务创建独立的 MLP 预测头（存入 ModuleDict 以被 PyTorch 追踪）
        self.heads = nn.ModuleDict()
        for field, spec in task_specs.items():
            # 回归输出 1 个值；分类输出 n_classes 个 logit
            out_dim = 1 if spec["type"] == "regression" else spec.get("n_classes", 2)
            self.heads[field] = _TaskHead(self.embed_dim, out_dim, spec["type"])

    # ---------------------------------------------------------------------- #
    def forward(self, mri: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向推理：MRI 体积 → 各任务的预测张量。

        参数：
          mri : (B, 1, D, H, W)，值域 [-1, 1]

        返回：
          {'age': (B, 1), 'sex': (B, n_classes), ...}
        """
        embed = self._get_embed(mri)   # (B, embed_dim)
        return {field: head(embed) for field, head in self.heads.items()}

    def _get_embed(self, mri: torch.Tensor) -> torch.Tensor:
        """
        从骨干网络提取全局嵌入向量，兼容三种不同接口：
          1. CNN 骨干：直接返回 (B, embed_dim) 张量
          2. Transformers 模型（带 CLS token）：取 last_hidden_state[:, 0, :]
          3. Transformers 模型（带 pooler）：取 pooler_output
          4. 最终退回：3D 自适应平均池化（兜底方案）
        """
        try:
            out = self.backbone(mri)
            if isinstance(out, torch.Tensor) and out.ndim == 2:
                return out   # CNN 直接输出 (B, embed_dim)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state[:, 0, :]   # CLS token
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            return out.last_hidden_state.mean(1)         # 序列均值
        except Exception as e:
            print(f"[NeuroFM] 骨干网络前向传播出错: {e}。使用自适应平均池化兜底。")
            h = nn.functional.adaptive_avg_pool3d(mri, 1).flatten(1)
            # 如果特征维度不够，通过重复拼接补全到 embed_dim
            return (h[..., :self.embed_dim] if h.shape[-1] >= self.embed_dim else
                    torch.cat([h] * (self.embed_dim // h.shape[-1] + 1), dim=-1)[..., :self.embed_dim])

    # ---------------------------------------------------------------------- #
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算多任务加权损失。

        损失公式：L_MT = Σ_k λ_k · L_k
          - L_k（回归）  = MSE(pred_k, target_k)
          - L_k（分类）  = CrossEntropy(pred_k, target_k)
          - λ_k（年龄）  = age_loss_weight（默认 2.0，强调年龄预测）
          - λ_k（其他）  = 1.0

        处理细节：
          - 自动忽略 targets 中没有对应字段的任务
          - NaN 标签自动遮蔽（mask），不参与损失计算
            （缺失标签在数据集中很常见，NaN 是约定的缺失值标记）
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for field, pred in predictions.items():
            if field not in targets:
                continue
            tgt = targets[field]
            # 遮蔽 NaN 标签（缺失年龄等字段的样本）
            mask = ~torch.isnan(tgt)
            if mask.sum() == 0:
                continue  # 该批次此任务全为 NaN，跳过
            spec = self.task_specs.get(field, {"type": "regression"})
            if spec["type"] == "regression":
                lam = self.age_loss_weight if field == "age" else 1.0
                # squeeze(-1)：将 (B, 1) 压缩为 (B,) 以匹配目标形状
                loss_k = nn.functional.mse_loss(pred[mask].squeeze(-1), tgt[mask])
            else:
                lam = 1.0
                loss_k = nn.functional.cross_entropy(pred[mask], tgt[mask].long())
            total_loss = total_loss + lam * loss_k
        return total_loss

    # ---------------------------------------------------------------------- #
    #  权重访问接口（联邦聚合和模型保存使用）                                        #
    # ---------------------------------------------------------------------- #

    def state_dict_backbone(self) -> dict:
        """返回骨干网络的状态字典（深拷贝）。"""
        return copy.deepcopy(self.backbone.state_dict())

    def state_dict_heads(self) -> dict:
        """返回所有任务头的状态字典（深拷贝）。"""
        return copy.deepcopy(self.heads.state_dict())

    def full_state_dict(self) -> dict:
        """返回完整模型（骨干 + 所有头）的状态字典（深拷贝）。"""
        return copy.deepcopy(self.state_dict())

    def load_full_state_dict(self, state: dict, strict: bool = False) -> None:
        """从状态字典恢复完整模型权重（strict=False 允许键不完全匹配）。"""
        self.load_state_dict(state, strict=strict)

    @property
    def age_head(self) -> nn.Module:
        """快捷属性：直接访问年龄预测头（论文中的主任务头）。"""
        return self.heads["age"]
