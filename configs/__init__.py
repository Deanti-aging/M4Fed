# configs 包的公开接口：将常用配置类和工厂函数直接暴露到 configs 命名空间
# 外部代码只需 `from configs import load_config` 而不必写完整路径 base_config.load_config
from .base_config import (
    DataConfig,         # 数据相关配置（hf_dataset_id、mri_shape、训练比例等）
    ModelConfig,        # 模型结构配置（VAE/U-Net/NeuroFM/TabPFN 的超参数）
    FederatedConfig,    # 联邦协议配置（轮数、合成比例、QLoRA 参数、FedProx μ 等）
    TrainingConfig,     # 训练超参数（学习率、批大小、设备、混合精度等）
    EvalConfig,         # 评估与可视化配置（图表格式、调色板、结果保存路径等）
    M4FedConfig,        # 顶层配置类（聚合所有子配置）
    load_config,        # 工厂函数：从 YAML 文件加载配置（不存在时退回全默认值）
)

# 声明当外部代码执行 `from configs import *` 时可以导入哪些名称
__all__ = [
    "DataConfig", "ModelConfig", "FederatedConfig", "TrainingConfig",
    "EvalConfig", "M4FedConfig", "load_config",
]
