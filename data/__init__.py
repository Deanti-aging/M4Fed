# data 包的公开接口：将各模块中常用的函数和类暴露到 data 命名空间
# 使外部代码可以 `from data import OpenMindDataset` 而不必写完整路径

from .openmind_dataset import (
    OpenMindDataset,             # PyTorch Dataset：封装 (MRI体积, 文本, 年龄, 元数据) 样本
    build_node_datasets,         # 高层构建函数：为每个联邦节点返回 train/test 数据集字典
    list_openmind_ds_ids,        # 从 HuggingFace 仓库文件结构枚举子数据集 ID 列表（有 max_ids 上限）
    list_ds_ids_from_metadata,   # 从本地 CSV 提取有效子数据集 ID（全量，过滤模态+age）
    make_openmind_sample_loader, # 构建真实数据加载闭包（ds_id → List[sample_dict]）
    compute_ds_quality_stats,    # 按 ds_id 统计 age/sex/health_status 缺失率（供 cross 质量筛选）
)
from .splits import assign_ds_splits          # 将子数据集 ID 按角色分配（cross/private/public）
from .preprocessing import preprocess_mri, preprocess_table  # MRI 和表格预处理函数
from .table_to_text import table_row_to_text  # 元数据行 → 自然语言描述字符串

# 声明 `from data import *` 时可以导入哪些名称
__all__ = [
    "OpenMindDataset", "build_node_datasets", "list_openmind_ds_ids", "list_ds_ids_from_metadata", "make_openmind_sample_loader",
    "compute_ds_quality_stats",
    "assign_ds_splits",
    "preprocess_mri", "preprocess_table",
    "table_row_to_text",
]
