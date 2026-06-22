# models/medmd 子包的公开接口：将 MedMD 相关类直接暴露到 models.medmd 命名空间
from .medmd import MedMD                   # 3D 潜扩散模型（包含 DDIMScheduler、_SimpleUNet3D/_build_monai_unet）
from .vae_wrapper import load_vae          # VAE 加载函数（尝试 MONAI 实现，退回 _MinimalVAE）
from .conditioning import BioMedCLIPEncoder  # 文本编码器（BioMedCLIP → 文本嵌入，用于 U-Net 交叉注意力）

# 声明 `from models.medmd import *` 时可以导入哪些名称
__all__ = ["MedMD", "load_vae", "BioMedCLIPEncoder"]
