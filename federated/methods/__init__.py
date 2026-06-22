# federated/methods 子包的公开接口：暴露 5 个联邦学习方法的入口函数
from .m4fed import run_m4fed     # M4Fed（本文提出）：三阶段协议（热身→联邦循环→本地训练）
from .fedavg import run_fedavg   # FedAvg 基线：加权平均聚合，不含合成数据增强
from .fedprox import run_fedprox # FedProx 基线：FedAvg + 近端正则化项（μ/2)‖w-w_global‖²
from .fan import run_fan         # FAN 基线：数据增强 + FedAvg（MONAI 3D 随机翻转）
from .gptfl import run_gptfl    # GPT-FL 基线：服务器生成文本条件 MRI，分发给客户端

# 声明 `from federated.methods import *` 时可以导入哪些名称
__all__ = ["run_m4fed", "run_fedavg", "run_fedprox", "run_fan", "run_gptfl"]
