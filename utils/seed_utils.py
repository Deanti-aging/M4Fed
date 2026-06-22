"""全局随机种子固定工具：确保实验可复现性。"""
import os           # 操作系统接口，用于设置环境变量
import random       # Python 内置随机数模块
import numpy as np  # NumPy 科学计算库（含自己的随机数引擎）
import torch        # PyTorch 深度学习框架（含 CPU/GPU 随机数引擎）


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
    random.seed(seed)                          # 固定 Python 内置 random 模块种子
    np.random.seed(seed)                       # 固定 NumPy 随机数生成器种子
    torch.manual_seed(seed)                    # 固定 PyTorch CPU 随机数生成器种子
    torch.cuda.manual_seed_all(seed)           # 固定所有 GPU 的随机数种子（支持多卡训练）
    os.environ["PYTHONHASHSEED"] = str(seed)   # 禁止 Python 哈希随机化（影响 dict/set 迭代顺序）
    # 使 cuDNN 运算确定性：禁用非确定性的快速算法，同一输入每次输出相同
    torch.backends.cudnn.deterministic = True
    # 禁止 cuDNN 为不同输入尺寸自动选优化算法（该自动选择会引入随机性）
    torch.backends.cudnn.benchmark = False
