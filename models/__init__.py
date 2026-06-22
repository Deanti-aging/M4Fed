# models 包的公开接口：将三个核心模型类直接暴露到 models 命名空间
# 使外部代码可以 `from models import MedMD, NeuroFMMultiTask, TabPFNImputer`

from .medmd.medmd import MedMD                   # 3D LDM 合成模型（VAE + DDIMScheduler + U-Net）
from .neurofm.neurofm_multitask import NeuroFMMultiTask  # 多任务领域预测模型（骨干 + 任务头）
from .tabpfn_wrapper import TabPFNImputer         # 表格缺失值填充器（TabPFN 或 median/mode 退回）

# 声明 `from models import *` 时可以导入哪些名称
__all__ = ["MedMD", "NeuroFMMultiTask", "TabPFNImputer"]
