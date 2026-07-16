# models 包的公开接口：将三个核心模型类直接暴露到 models 命名空间
# 使外部代码可以 `from models import MedMD, SwinMultiTask, TabPFNImputer`

from .medmd.medmd import MedMD                       # 3D LDM 合成模型（VAE + DDIMScheduler + U-Net）
from .swin.swin_multitask import SwinMultiTask       # SwinUNETR 多任务领域预测模型
from .tabpfn_wrapper import TabPFNImputer            # 表格缺失值填充器（TabPFN 或 median/mode 退回）

__all__ = ["MedMD", "SwinMultiTask", "TabPFNImputer"]
