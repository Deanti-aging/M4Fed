# federated 包的公开接口：暴露服务器和客户端参与者类
from .server import Server   # 服务器节点（含公开数据、MedMD 预训练能力、聚合逻辑）
from .client import Client   # 客户端节点（含私有数据、QLoRA 微调能力）

# 声明 `from federated import *` 时可以导入哪些名称
__all__ = ["Server", "Client"]
