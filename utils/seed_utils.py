"""全局随机种子固定工具：确保实验可复现性。"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    固定所有随机数生成器的种子，保证实验结果可复现。

    覆盖范围：
      - Python 内置 random 模块
      - NumPy 随机数生成器
      - PyTorch CPU 随机数生成器
      - PyTorch 所有 GPU 随机数生成器（多卡场景）
      - Python hash 随机化（通过环境变量）
      - cuDNN 算法选择（确定性模式，可能略微降低性能）

    注意：
      - cuDNN 确定性模式（deterministic=True）会禁用某些 CUDA 优化算法，
        可能导致训练速度略慢，但保证相同输入得到相同输出。
      - benchmark=False 禁止 cuDNN 自动为不同输入尺寸选择最优算法
        （该功能会引入随机性）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)        # 覆盖所有 GPU 设备
    os.environ["PYTHONHASHSEED"] = str(seed)   # 禁止 Python 哈希随机化
    # 使 cuDNN 运算确定性（禁用非确定性的快速算法）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
