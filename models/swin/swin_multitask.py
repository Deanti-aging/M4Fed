"""
SwinUNETR 多任务预测模型：骨干网络 + 多个任务头。

主任务：脑年龄预测（回归，MSE 损失，权重加倍）。
附加任务：其他元数据字段（性别分类、健康状态分类、种族分类等）。

骨干网络：MONAI SwinUNETR encoder（swinViT），Tang et al., CVPR 2022，
在脑 MRI（MSD 数据集）上用 masked image modeling 自监督预训练。
替代原 TensorFlow 版 NeuroFM（NeuroAI-UofG/NeuroFM，与 PyTorch 不兼容）。

加载优先级：
  1. 下载 MONAI 官方 SwinViT SSL 预训练权重（model_swinvit.pt）
  2. 若无法加载，退回轻量级 3D CNN 骨干（_Backbone3DCNN）

设计原则：
  - 骨干网络对 bottleneck 特征做全局平均池化，输出 (B, embed_dim) 嵌入向量
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
    SwinUNETR 不可用时使用的轻量级 3D CNN 骨干网络。

    网络结构：
      3 个 3D 卷积块（Conv3d + BN + GELU）+ 全局平均池化 + 全连接层
      特征通道数：1 → 16 → 32 → 64 → embed_dim
      每次卷积步长为 2，空间分辨率逐步降为 1/8

    输出：形状 (B, embed_dim) 的全局嵌入向量
    """

    def __init__(self, in_ch: int = 1, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
            nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, 1, D, H, W)，输出 (B, embed_dim) 的全局嵌入向量。"""
        return self.net(x)


# --------------------------------------------------------------------------- #
#  SwinUNETR 编码器封装                                                          #
# --------------------------------------------------------------------------- #

class _SwinUNETREncoder(nn.Module):
    """
    MONAI SwinTransformer（swinViT）编码器封装，提取全局嵌入向量。

    SwinUNETR 的 swinViT 前向输出 5 个分辨率递减的特征图列表（patch embed 到
    4 个 Swin 阶段）。取最深层（bottleneck）做全局平均池化后展平，得到
    (B, feature_size × 16) 的全局表示。

    embed_dim = feature_size × 16
      feature_size=48 → embed_dim=768（与 BERT-base 一致）
    """

    def __init__(self, swin_vit: nn.Module, feature_size: int):
        super().__init__()
        self.swin_vit = swin_vit
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flat = nn.Flatten()
        # 4 个 patch-merge 阶段各使 channels 翻倍：feature_size × 2^4 = feature_size × 16
        self.embed_dim = feature_size * 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, 1, D, H, W)，输出 (B, embed_dim)。"""
        hidden_states = self.swin_vit(x)      # list[5]，最后一个是 bottleneck
        bottleneck = hidden_states[-1]         # (B, feature_size×16, d, h, w)
        return self.flat(self.pool(bottleneck))  # (B, embed_dim)


# --------------------------------------------------------------------------- #
#  SwinUNETR 骨干加载函数                                                        #
# --------------------------------------------------------------------------- #

def _load_swinunetr_backbone(
    img_size: tuple,
    feature_size: int,
    pretrain_url: str,
    device: str,
) -> Tuple[nn.Module, int]:
    """
    构建 SwinUNETR encoder 并可选加载 MONAI 官方自监督预训练权重。

    预训练权重（model_swinvit.pt）：MONAI 在脑 MRI（MSD 数据集）上用
    masked image modeling 自监督预训练的 SwinViT checkpoint。
    checkpoint 格式：{"state_dict": {...}}，键名含 "module." 前缀（DataParallel 保存）。

    pretrain_url 为空串时跳过下载，使用随机初始化权重（仅调试用）。
    失败时退回 _Backbone3DCNN 并打印警告。

    返回：
      (backbone_module, embed_dim)，embed_dim = feature_size × 16
    """
    try:
        from monai.networks.nets import SwinUNETR

        full_model = SwinUNETR(
            # monai>=1.5 移除了 img_size 参数：SwinUNETR 改为在 forward 时动态推断输入尺寸
            in_channels=1,
            out_channels=2,            # 解码器不使用，取最小值
            feature_size=feature_size,
            use_checkpoint=True,       # 梯度检查点：以重计算换显存
        )

        if pretrain_url:
            state = torch.hub.load_state_dict_from_url(
                pretrain_url, map_location="cpu", check_hash=False,
            )
            raw = state.get("state_dict", state)
            # checkpoint 由 DataParallel 保存，去除 "module." 前缀后再载入
            cleaned = {k.replace("module.", ""): v for k, v in raw.items()}
            full_model.swinViT.load_state_dict(cleaned, strict=False)
            print(f"[SwinUNETR] 预训练权重加载成功（feature_size={feature_size}）。")

        encoder = _SwinUNETREncoder(full_model.swinViT, feature_size)
        return encoder.to(device), encoder.embed_dim

    except Exception as e:
        print(f"[SwinUNETR] 加载失败: {e}。退回 CNN 骨干。")
        backbone = _Backbone3DCNN(embed_dim=256)
        return backbone.to(device), 256


def _create_swinunetr_random(
    img_size: tuple,
    feature_size: int,
    device: str,
) -> Tuple[nn.Module, int]:
    """
    构建随机初始化的 SwinUNETR encoder（消融实验 C 用）。

    与正式模式使用完全相同的 SwinUNETR 架构，但权重随机初始化（不加载预训练权重），
    从而单独评估预训练对性能的贡献（控制变量，排除架构差异的干扰）。

    失败时退回 _Backbone3DCNN 并打印警告。

    返回：
      (backbone_module, embed_dim)
    """
    try:
        from monai.networks.nets import SwinUNETR

        full_model = SwinUNETR(
            # monai>=1.5 移除了 img_size 参数：SwinUNETR 改为在 forward 时动态推断输入尺寸
            in_channels=1,
            out_channels=2,
            feature_size=feature_size,
            use_checkpoint=True,
        )
        encoder = _SwinUNETREncoder(full_model.swinViT, feature_size)
        print(f"[SwinUNETR] 随机初始化骨干（消融实验，feature_size={feature_size}）。")
        return encoder.to(device), encoder.embed_dim

    except Exception as e:
        print(f"[SwinUNETR] 随机初始化失败: {e}。退回 CNN 骨干。")
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
        """输入 (B, in_dim) 的嵌入，输出 (B, out_dim) 的预测结果。"""
        return self.head(x)


# --------------------------------------------------------------------------- #
#  SwinMultiTask 主体                                                            #
# --------------------------------------------------------------------------- #

class SwinMultiTask(nn.Module):
    """
    SwinUNETR encoder 骨干 + 每字段独立任务头的多任务预测模型。

    骨干：MONAI SwinUNETR encoder（swinViT），CVPR 2022，在脑 MRI 上自监督预训练。
    embed_dim = feature_size × 16（feature_size=48 时为 768）。

    使用方式：
      task_specs = {
        'age':           {'type': 'regression',     'n_classes': 1},
        'sex':           {'type': 'classification', 'n_classes': 2},
        'health_status': {'type': 'classification', 'n_classes': 2},
        'race':          {'type': 'classification', 'n_classes': 5},
      }
      model = SwinMultiTask(task_specs=task_specs)
      preds = model(mri_tensor)   # {'age': tensor(...), 'sex': tensor(...)}
      loss = model.compute_loss(preds, targets)

    联邦学习中的角色：
      - M4Fed 的"领域模型"，负责从 MRI 预测年龄等临床指标
      - QLoRA 微调时仅更新骨干中的 LoRA 参数，减少本地计算和通信开销
    """

    def __init__(
        self,
        pretrain_url: str = "",
        img_size: tuple = (96, 96, 96),
        feature_size: int = 48,
        task_specs: Optional[Dict[str, dict]] = None,
        device: str = "cpu",
        random_init: bool = False,
        age_loss_weight: float = 2.0,
    ):
        """
        参数：
          pretrain_url     : MONAI SwinViT SSL 预训练权重的下载 URL
                             （空串则跳过下载，仅用于调试或消融对照）
          img_size         : 输入 MRI 空间尺寸 (D, H, W)，须为 32 的整数倍
          feature_size     : SwinUNETR 基础特征维度，embed_dim = feature_size × 16
          task_specs       : 任务规格字典（见类说明）
          device           : 计算设备（"cuda" 或 "cpu"）
          random_init      : True = 随机初始化骨干（消融实验 C，不加载预训练权重）
          age_loss_weight  : 年龄预测任务的损失权重倍数（默认 2.0）
        """
        super().__init__()
        self.device_str = device
        self.age_loss_weight = age_loss_weight

        if random_init:
            # 消融实验 C：相同架构 + 随机权重，单独验证预训练的增益
            self.backbone, self.embed_dim = _create_swinunetr_random(
                img_size, feature_size, device
            )
        else:
            # 正式模式：加载 MONAI 官方预训练权重（失败时自动退回 CNN 骨干）
            self.backbone, self.embed_dim = _load_swinunetr_backbone(
                img_size, feature_size, pretrain_url, device
            )

        if task_specs is None:
            task_specs = {"age": {"type": "regression", "n_classes": 1}}
        self.task_specs = task_specs

        # 为每个任务创建独立的 MLP 预测头（ModuleDict 自动追踪参数）
        self.heads = nn.ModuleDict()
        for field, spec in task_specs.items():
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
        embed = self.backbone(mri)   # (B, embed_dim)
        return {field: head(embed) for field, head in self.heads.items()}

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
            mask = ~torch.isnan(tgt)
            if mask.sum() == 0:
                continue
            spec = self.task_specs.get(field, {"type": "regression"})
            if spec["type"] == "regression":
                lam = self.age_loss_weight if field == "age" else 1.0
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
        """返回骨干网络的状态字典（深拷贝），用于传输骨干权重。"""
        return copy.deepcopy(self.backbone.state_dict())

    def state_dict_heads(self) -> dict:
        """返回所有任务头的状态字典（深拷贝），用于传输任务头权重。"""
        return copy.deepcopy(self.heads.state_dict())

    def full_state_dict(self) -> dict:
        """返回完整模型（骨干 + 所有任务头）的状态字典（深拷贝）。"""
        return copy.deepcopy(self.state_dict())

    def load_full_state_dict(self, state: dict, strict: bool = False) -> None:
        """从状态字典恢复完整模型权重（strict=False 允许键不完全匹配）。"""
        self.load_state_dict(state, strict=strict)

    @property
    def age_head(self) -> nn.Module:
        """快捷属性：直接访问年龄预测头（论文中的主任务头）。"""
        return self.heads["age"]


