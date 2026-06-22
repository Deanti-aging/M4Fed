"""服务器和客户端的抽象基类，定义联邦参与者的公共接口。"""
from __future__ import annotations  # 允许注解中使用尚未定义的类名
import copy                          # 标准库：深拷贝（用于返回权重状态字典副本）
import torch                         # PyTorch 深度学习框架
import torch.nn as nn                # 神经网络模块
from typing import Dict, List, Optional   # 类型注解工具
from torch.utils.data import DataLoader   # PyTorch 数据加载器


class BaseParticipant:
    """
    M4Fed 联邦学习中的参与者基类（服务器和客户端均继承自此类）。

    封装了：
      - 配置和设备信息
      - 模型引用（MedMD 生成模型 + NeuroFM 领域预测模型）
      - DataLoader 引用
      - 模型权重的获取/设置接口（供联邦聚合时使用）

    子类（Server / Client）在此基础上：
      - 初始化自己的 DataLoader（私有/公开/混合等）
      - 实现 build_models() 方法（从配置构建模型实例）
      - 添加节点特有的方法（如服务器的公开数据访问）
    """

    # 节点标识符，子类覆盖为具体名称（如 "server"、"client_0"）
    node_id: str = "base"

    def __init__(self, cfg, device: str = "cpu"):
        """
        参数：
          cfg    : M4FedConfig 配置对象
          device : 计算设备（"cuda" 或 "cpu"）
        """
        self.cfg = cfg                             # 保存全局配置对象（供子类访问超参数）
        self.device = device                        # 保存计算设备字符串
        self.medmd = None                           # MedMD 生成模型实例（由 build_models 初始化）
        self.neurofm = None                         # NeuroFMMultiTask 领域预测模型实例
        self.train_loader: Optional[DataLoader] = None  # 训练数据加载器（子类初始化）
        self.test_loader: Optional[DataLoader] = None   # 测试数据加载器（子类初始化）

    # ---------------------------------------------------------------------- #
    #  NeuroFM 权重接口（用于联邦聚合时传递领域模型权重）                         #
    # ---------------------------------------------------------------------- #

    def get_neurofm_state(self) -> dict:
        """
        获取 NeuroFM 的完整状态字典（深拷贝）。
        深拷贝确保返回的字典与模型当前权重独立，后续修改不会相互影响。
        """
        return copy.deepcopy(self.neurofm.state_dict())  # 深拷贝保证外部无法意外修改模型权重

    def set_neurofm_state(self, state: dict, strict: bool = False) -> None:
        """
        将状态字典加载到 NeuroFM（strict=False 允许部分加载）。
        联邦聚合后，服务器将聚合权重分发给各参与方时调用此方法。
        """
        self.neurofm.load_state_dict(state, strict=strict)  # 将 state 中的权重写入 NeuroFM

    def get_medmd_unet_state(self) -> dict:
        """
        获取 MedMD 中 U-Net 的状态字典（深拷贝）。
        仅传输 U-Net 权重（VAE 和文本编码器为冻结参数，无需传输）。
        """
        return copy.deepcopy(self.medmd.unet.state_dict())  # 只拷贝 U-Net 部分

    def set_medmd_unet_state(self, state: dict, strict: bool = False) -> None:
        """
        将状态字典加载到 MedMD 的 U-Net 部分。
        热身阶段服务器将预训练 U-Net 权重分发给客户端时调用。
        """
        self.medmd.unet.load_state_dict(state, strict=strict)  # 将权重写入 U-Net

    # ---------------------------------------------------------------------- #
    #  数据量统计接口                                                            #
    # ---------------------------------------------------------------------- #

    def n_train_samples(self) -> int:
        """
        返回训练数据集的样本总数。
        FedAvg 聚合时用于计算各客户端的权重（样本数越多权重越大）。
        """
        if self.train_loader is None:
            return 0   # 尚未初始化 DataLoader，返回 0（安全处理）
        return len(self.train_loader.dataset)  # 取 DataLoader 关联的 Dataset 长度

    def n_test_samples(self) -> int:
        """返回测试数据集的样本总数（用于评估报告）。"""
        if self.test_loader is None:
            return 0   # 尚未初始化 DataLoader，返回 0
        return len(self.test_loader.dataset)   # 取 DataLoader 关联的 Dataset 长度
