"""
冻结的 BioMedCLIP 文本编码器封装。

将元数据文本（如 "age is 32, sex is female"）编码为交叉注意力条件张量，
供 U-Net 在去噪时使用。

BioMedCLIP 是 open_clip 模型，必须通过 open_clip 库加载。
若 open_clip 不可用或权重下载失败，退回随机嵌入（仅用于冒烟测试）。
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List


class BioMedCLIPEncoder(nn.Module):
    """
    使用 BioMedCLIP 对文本进行编码，输出逐 token 隐藏状态供交叉注意力使用。

    本质上还是从 HuggingFace Hub 下载，
    只是下载和解析的工具是 open_clip 而不是 transformers。
    两者的区别只是客户端库不同，
    数据来源都是 huggingface.co。

    输出形状：(B, context_length, hidden_dim)
    所有参数初始化后立即冻结。
    """

    def __init__(
        self,
        repo_id: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: str = "cpu",
        max_length: int = 77,
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.hidden_dim = 512  # BioMedCLIP text transformer width

        self._loaded = False
        self._load(repo_id)

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def _load(self, repo_id: str) -> None:
        """用 open_clip 加载 BioMedCLIP；失败时 forward 返回随机嵌入。"""
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                f"hf-hub:{repo_id}"
            )
            self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{repo_id}")
            # 只保留文本编码器分支，丢弃视觉分支以节省显存
            self.text_encoder = model.text.to(self.device)
            self.hidden_dim = self.text_encoder.transformer.width
            self._loaded = True
        except Exception as e:
            print(f"[BioMedCLIP] 加载失败: {e}。将使用随机嵌入替代。")
            self._loaded = False

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        参数:
          texts: 长度为 B 的文本列表

        返回:
          (B, context_length, hidden_dim) float32 张量
        """
        if not self._loaded:
            B = len(texts)
            return torch.randn(B, self.max_length, self.hidden_dim, device=self.device)

        # open_clip tokenizer 直接返回 LongTensor (B, context_length)
        tokens = self.tokenizer(texts).to(self.device)

        # 逐步调用 TextTransformer 内部，取 ln_final 之后、text_projection 之前的
        # 逐 token 表示，形状 (B, context_length, hidden_dim)
        enc = self.text_encoder

        # 第①步：词元嵌入
        # token_embedding 是一个 Embedding(vocab_size, dim) 查找表，
        # 将每个 token id 映射为 dim 维的稠密向量。
        # 输出形状：(B, L, dim)，其中 L = context_length（默认 77）
        x = enc.token_embedding(tokens)            # (B, L, dim)

        # 第②步：叠加位置编码
        # positional_embedding 形状为 (max_seq_len, dim)，是可学习参数。
        # 用切片 [:L] 保证对任意短于 max_seq_len 的序列都能对齐。
        # 位置编码与词元嵌入直接相加（Transformer 标准做法），
        # 为模型注入每个 token 在序列中的绝对位置信息。
        x = x + enc.positional_embedding[:tokens.shape[1]]

        # 第③步：转置为 LND 布局供 Transformer 处理
        # open_clip 的 Transformer 期望输入为 (L, B, dim)，即"序列长度在最前"。
        # permute(1, 0, 2) 将 (B, L, dim) → (L, B, dim)。
        x = x.permute(1, 0, 2)                    # NLD -> LND（open_clip 约定）

        # 第④步：Transformer 编码
        # enc.transformer 是一个多层因果自注意力堆栈（Causal Self-Attention）。
        # attn_mask 是上三角因果掩码（若 enc 上存在则传入，否则传 None）；
        # 掩码确保每个位置只能看到自身及之前的 token，符合自回归语言模型惯例。
        # 输出形状不变，仍为 (L, B, dim)。
        x = enc.transformer(x, attn_mask=getattr(enc, "attn_mask", None))

        # 第⑤步：转置回 NLD 布局
        # 将 (L, B, dim) → (B, L, dim)，恢复标准批次优先格式，
        # 以便后续层按批次索引处理。
        x = x.permute(1, 0, 2)                    # LND -> NLD

        # 第⑥步：最终层归一化
        # ln_final 是 LayerNorm(dim)，对最后一个维度做归一化。
        # 此处**故意不做** text_projection（open_clip 默认会再乘一个投影矩阵
        # 将 dim → embed_dim），因为我们需要的是逐 token 的隐藏状态，
        # 而非压缩后的全局句向量，以保留更丰富的上下文信息供交叉注意力使用。
        # 输出形状：(B, L, hidden_dim)
        x = enc.ln_final(x)                        # (B, L, hidden_dim)

        return x.float()
