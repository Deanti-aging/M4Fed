# training 子包的公开接口：MedMD 生成模型训练器 + SwinMultiTask 领域模型训练器
from .medmd_trainer import MedMDTrainer   # 扩散模型训练器（FP16 + GradScaler + img2img 合成）
from .swin_trainer import SwinTrainer     # SwinUNETR 多任务训练器（支持 QLoRA + FedProx 近端项）

__all__ = ["MedMDTrainer", "SwinTrainer"]
