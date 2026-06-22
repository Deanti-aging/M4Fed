"""
冻结的 BioMedCLIP 文本编码器封装。

将元数据文本（如 "age is 32, sex is female"）编码为交叉注意力条件张量，
供 U-Net 在去噪时使用。

若 transformers 库或模型权重不可用，退回随机嵌入生成器
（不影响管道流程，但合成质量无意义——仅用于冒烟测试）。
"""
from __future__ import annotations   # 允许注解中使用尚未定义的类名
import torch                         # PyTorch 深度学习框架
import torch.nn as nn                # 神经网络模块基类
from typing import List, Optional    # 类型注解工具


class BioMedCLIPEncoder(nn.Module):
    """
    使用 BioMedCLIP（基于 PubMedBERT 的生物医学 CLIP 模型）对文本进行编码。

    BioMedCLIP 在大规模生物医学图像-文本对上预训练，
    对医学术语（年龄、诊断、BMI 等）具有良好的语义理解能力。

    输出形状：(B, seq_len, hidden_dim)
      - B       : batch size
      - seq_len : 最大 token 长度（77，与 CLIP 对齐）
      - hidden_dim: 隐藏层维度（BioMedCLIP 为 512）

    所有参数在初始化后立即冻结（不参与训练）。
    """

    def __init__(
        self,
        repo_id: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: str = "cpu",
        max_length: int = 77,
    ):
        """
        参数：
          repo_id   : HuggingFace 仓库 ID
          device    : 计算设备
          max_length: tokenizer 最大序列长度（短句 pad，长句 truncate）
        """
        super().__init__()           # 调用 nn.Module 父类初始化
        self.device = device         # 保存计算设备字符串（用于生成随机嵌入时指定设备）
        self.max_length = max_length  # 最大 token 序列长度（短句补零，长句截断）
        self.hidden_dim = 512         # BioMedCLIP 文本编码的默认维度（后续可能被动态更新）

        self._loaded = False    # 标记模型是否已成功加载（False = 使用随机嵌入作为退回方案）
        self._load(repo_id)     # 尝试从 HuggingFace 加载 tokenizer 和文本模型

        # 冻结所有参数：文本编码器在训练过程中保持固定（仅作为特征提取器）
        for p in self.parameters():
            p.requires_grad_(False)  # 关闭所有参数的梯度计算
        self.eval()   # 切换为推理模式（关闭 Dropout 和 BatchNorm 的训练行为）

    def _load(self, repo_id: str) -> None:
        """
        从 HuggingFace 加载 tokenizer 和文本模型。
        失败时设置 _loaded=False，forward 将返回随机嵌入。
        """
        try:
            from transformers import AutoTokenizer, AutoModel  # HuggingFace Transformers 库
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id)   # 加载文本分词器
            self.text_model = AutoModel.from_pretrained(repo_id).to(self.device)  # 加载模型并移到设备
            self._loaded = True   # 标记加载成功
            # 动态读取实际的隐藏层维度（不同版本/模型的 hidden_size 可能不同）
            cfg = self.text_model.config   # 获取模型配置
            if hasattr(cfg, "text_config"):
                # CLIP 类模型：配置分为 text_config 和 vision_config 两部分
                self.hidden_dim = cfg.text_config.hidden_size
            elif hasattr(cfg, "hidden_size"):
                # BERT 类模型：直接读取 hidden_size
                self.hidden_dim = cfg.hidden_size
        except Exception as e:
            # 加载失败（网络问题/依赖未安装等）：退回随机嵌入
            print(f"[BioMedCLIP] 模型加载失败: {e}。将使用随机嵌入作为替代。")
            self._loaded = False   # 标记加载失败，forward 时返回随机嵌入

    @torch.no_grad()  # 推理时不计算梯度，节省显存
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        将文本列表编码为条件嵌入张量。

        参数：
          texts : 文本字符串列表，长度为 batch size

        返回：
          形状 (B, seq_len, hidden_dim) 的 float32 张量
            - 模型正常加载时：返回 BioMedCLIP 的 last_hidden_state
            - 模型未加载时：返回随机正态噪声（维度相同，仅用于测试流程）
        """
        if not self._loaded:
            # 模型不可用时返回形状正确的随机嵌入（保证管道不报错，但语义无意义）
            B = len(texts)   # batch size = 文本列表长度
            return torch.randn(B, self.max_length, self.hidden_dim, device=self.device)

        # Tokenize：将文本转为 token ID 序列；pad 到 max_length，超长截断
        inputs = self.tokenizer(
            texts,                        # 输入文本列表
            return_tensors="pt",          # 返回 PyTorch 张量（而非 numpy/list）
            padding="max_length",         # 补零到 max_length（对齐批次内的所有序列）
            truncation=True,              # 超长序列截断（保留前 max_length 个 token）
            max_length=self.max_length,   # 最大序列长度（与 CLIP 标准对齐）
        ).to(self.device)   # 将 token 张量移到模型所在设备

        # 提取 last_hidden_state（包含每个 token 的上下文化表示）
        try:
            out = self.text_model.text_model(**inputs)  # 调用文本分支（CLIP 模型结构）
            return out.last_hidden_state   # (B, seq_len, hidden_dim)
        except Exception:
            # 部分模型接口不同（如整体 CLIP 模型而非仅文本分支）
            out = self.text_model(**inputs)  # 尝试调用整体模型接口
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state  # 直接返回 last_hidden_state
            # 最后的兜底：将 pooler_output (B, hidden_dim) 扩展为序列维度 (B, max_length, hidden_dim)
            return out.pooler_output.unsqueeze(1).expand(-1, self.max_length, -1)
