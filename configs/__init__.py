# configs 包的公开接口：将常用类和 load_config 直接暴露到 configs 命名空间
from .base_config import (
    DataConfig, ModelConfig, FederatedConfig, TrainingConfig,
    EvalConfig, M4FedConfig, load_config
)

# 声明了当外部代码执行 from configs import * 时可以导入哪些名称
__all__ = [
    "DataConfig", "ModelConfig", "FederatedConfig", "TrainingConfig",
    "EvalConfig", "M4FedConfig", "load_config",
]
