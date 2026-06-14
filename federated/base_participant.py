"""服务器和客户端的抽象基类，定义联邦参与者的公共接口。"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from torch.utils.data import DataLoader


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
        self.cfg = cfg
        self.device = device
        self.medmd = None        # MedMD 生成模型实例（由 build_models 初始化）
        self.neurofm = None      # NeuroFMMultiTask 领域预测模型实例
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    # ---------------------------------------------------------------------- #
    #  NeuroFM 权重接口（用于联邦聚合时传递领域模型权重）                         #
    # ---------------------------------------------------------------------- #

    def get_neurofm_state(self) -> dict:
        """
        获取 NeuroFM 的完整状态字典（深拷贝）。
        深拷贝确保返回的字典与模型当前权重独立，后续修改不会相互影响。
        """
        return copy.deepcopy(self.neurofm.state_dict())

    def set_neurofm_state(self, state: dict, strict: bool = False) -> None:
        """
        将状态字典加载到 NeuroFM（strict=False 允许部分加载）。
        联邦聚合后，服务器将聚合权重分发给各参与方时调用此方法。
        """
        self.neurofm.load_state_dict(state, strict=strict)

    def get_medmd_unet_state(self) -> dict:
        """
        获取 MedMD 中 U-Net 的状态字典（深拷贝）。
        仅传输 U-Net 权重（VAE 和文本编码器为冻结参数，无需传输）。
        """
        return copy.deepcopy(self.medmd.unet.state_dict())

    def set_medmd_unet_state(self, state: dict, strict: bool = False) -> None:
        """
        将状态字典加载到 MedMD 的 U-Net 部分。
        热身阶段服务器将预训练 U-Net 权重分发给客户端时调用。
        """
        self.medmd.unet.load_state_dict(state, strict=strict)

    # ---------------------------------------------------------------------- #
    #  数据量统计接口                                                            #
    # ---------------------------------------------------------------------- #

    def n_train_samples(self) -> int:
        """
        返回训练数据集的样本总数。
        FedAvg 聚合时用于计算各客户端的权重（样本数越多权重越大）。
        """
        if self.train_loader is None:
            return 0
        return len(self.train_loader.dataset)

    def n_test_samples(self) -> int:
        """返回测试数据集的样本总数。"""
        if self.test_loader is None:
            return 0
        return len(self.test_loader.dataset)
