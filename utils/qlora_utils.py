"""
QLoRA 工具：为模型的线性层添加 4-bit LoRA 适配器（使用 PEFT 库）。

背景：
  QLoRA（Quantized Low-Rank Adaptation）将大型预训练模型量化为 4-bit，
  同时只微调少量低秩适配器参数，大幅降低微调所需的内存和通信开销。

  在 M4Fed 中，客户端使用 QLoRA 微调 NeuroFM 骨干和 MedMD U-Net，
  减少本地训练的计算负担。

退回策略：
  - bitsandbytes 不可用：退回标准 LoRA（无 4-bit 量化）
  - PEFT 不可用：直接全量微调（所有参数都可训练）
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Optional


def apply_qlora(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bits: int = 4,
) -> nn.Module:
    """
    为模型添加 QLoRA 适配器层：基础权重冻结，只训练低秩 LoRA 矩阵。

    LoRA 原理：
      原始权重 W₀ 保持不变（冻结），添加可训练的低秩更新 ΔW = BA：
        W = W₀ + ΔW = W₀ + BA
      其中 B ∈ R^{d×r}，A ∈ R^{r×k}，r ≪ min(d, k)。
      实际学习率缩放为 lora_alpha / r。

    参数：
      model          : 要应用 LoRA 的模型（通常是 U-Net 或 NeuroFM 骨干）
      r              : LoRA 低秩矩阵的秩（越小参数量越少，表达能力越弱）
      lora_alpha     : 缩放系数（影响有效学习率）
      lora_dropout   : LoRA 层的 Dropout 比例（防止过拟合）
      target_modules : 要替换的线性层名称列表；None 时自动检测
      bits           : 量化位数（4 = QLoRA；若 bitsandbytes 不可用则忽略）

    返回：
      添加 LoRA 适配器后的模型（基础权重已冻结）
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        try:
            from peft import prepare_model_for_kbit_training
            import bitsandbytes  # 检查 bitsandbytes 是否可用
            model = prepare_model_for_kbit_training(model)   # 量化预处理
            use_4bit = (bits == 4)
        except ImportError:
            use_4bit = False  # 无 bitsandbytes，退回标准 LoRA

        # 自动检测目标模块（若未指定）
        if target_modules is None:
            target_modules = _find_linear_names(model)

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",   # 不训练 bias 项（通常不影响性能）
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()   # 打印可训练参数比例（供调试参考）
        return model
    except Exception as e:
        print(f"[QLoRA] PEFT 库不可用: {e}。将全量微调所有参数。")
        return model  # 退回：直接返回原始模型（全量微调）


def get_qlora_params(model: nn.Module) -> List[nn.Parameter]:
    """
    提取模型中 LoRA 适配器的可训练参数（用于传给优化器）。

    筛选规则：参数名称中包含 "lora_" 且 requires_grad=True。
    若 PEFT 未正确应用（无 LoRA 参数），退回所有参数。

    参数：
      model : 已应用 LoRA 的模型

    返回：
      可训练 LoRA 参数列表
    """
    try:
        return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    except Exception:
        return list(model.parameters())


def _find_linear_names(model: nn.Module, max_names: int = 10) -> List[str]:
    """
    自动检测模型中线性层的名称（作为 LoRA 的替换目标）。

    策略：遍历所有模块，收集 nn.Linear 层的最末级名称（如 "to_q"、"fc1" 等），
    最多收集 max_names 个不同名称。

    退回值：若未找到任何线性层，返回 transformer 交叉注意力的常见层名
    ["to_q", "to_k", "to_v", "to_out"]。

    参数：
      max_names : 最多收集的不同层名数量（避免目标过多导致微调代价过高）
    """
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            part = name.split(".")[-1]   # 取最末级名称（忽略父路径）
            if part not in names:
                names.append(part)
        if len(names) >= max_names:
            break
    return names or ["to_q", "to_k", "to_v", "to_out"]
