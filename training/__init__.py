# training 子包的公开接口：暴露 MedMD 和 NeuroFM 两个训练器类
from .medmd_trainer import MedMDTrainer       # 扩散模型训练器（FP16 + GradScaler + img2img 合成）
from .neurofm_trainer import NeuroFMTrainer   # 多任务预测模型训练器（支持 QLoRA + FedProx 近端项）

# 声明 `from training import *` 时可以导入哪些名称
__all__ = ["MedMDTrainer", "NeuroFMTrainer"]
