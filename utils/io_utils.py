"""CSV 保存、模型检查点读写、目录管理等 I/O 工具函数。"""
from __future__ import annotations  # 允许注解中使用尚未定义的类名
import os                           # 操作系统接口：路径与目录操作
import copy                         # 标准库：对象深拷贝
import torch                        # PyTorch：模型权重保存/加载
import pandas as pd                 # 数据分析库：DataFrame 与 CSV 操作
from typing import Any, Dict, Optional  # 类型注解工具


def ensure_dir(path: str) -> str:
    """
    确保目录存在，若不存在则递归创建。

    参数：
      path : 目录路径

    返回：
      原始路径（方便链式调用：out = ensure_dir(os.path.join(root, subdir))）
    """
    os.makedirs(path, exist_ok=True)  # 递归创建目录；exist_ok=True 表示目录已存在时不报错
    return path                        # 返回原路径，方便链式调用


def save_csv(data: Any, path: str) -> None:
    """
    将数据保存为 CSV 文件（自动创建父目录）。

    支持两种输入类型：
      - pd.DataFrame：直接保存
      - 其他（dict、list 等）：先转为 DataFrame 再保存

    参数：
      data : 要保存的数据
      path : CSV 文件路径
    """
    ensure_dir(os.path.dirname(path) or ".")  # 确保父目录存在（path 若无目录部分则用当前目录-工作目录-启动python脚本的目录）
    if isinstance(data, pd.DataFrame):
        data.to_csv(path, index=False)         # DataFrame 直接保存，不写行索引
    else:
        pd.DataFrame(data).to_csv(path, index=False)  # 先转 DataFrame 再保存


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    保存模型的 state_dict 到 .pt 文件（自动创建父目录）。

    只保存权重（state_dict），不保存模型结构，
    加载时需要先实例化相同结构的模型再调用 load_state_dict。

    参数：
      model : PyTorch 模型
      path  : 保存路径（推荐以 .pt 结尾）
    """
    ensure_dir(os.path.dirname(path) or ".")      # 确保父目录存在
    torch.save(model.state_dict(), path)           # 将权重字典序列化到文件


def load_model(model: torch.nn.Module, path: str, strict: bool = False) -> torch.nn.Module:
    """
    从文件加载模型权重（in-place 修改，同时返回模型方便链式调用）。

    参数：
      model  : 目标模型实例（需与保存时的结构兼容）
      path   : 权重文件路径（.pt 文件）
      strict : False（默认）允许部分加载（键不完全匹配时不报错），
               适用于加载预训练权重（结构可能有微小差异）

    返回：
      加载权重后的模型
    """
    state = torch.load(path, map_location="cpu")   # 先加载到 CPU，避免 GPU OOM；后由模型 .to(device) 移动
    model.load_state_dict(state, strict=strict)     # 将权重写入模型（strict=False 允许部分匹配）
    return model                                     # 返回模型本身，方便链式调用


def copy_model_state(model: torch.nn.Module) -> dict:
    """
    创建模型 state_dict 的深拷贝。

    用于联邦聚合中保存本地训练前的全局状态，
    避免后续操作修改已记录的状态。

    返回：
      state_dict 的深拷贝字典
    """
    return copy.deepcopy(model.state_dict())  # 深拷贝：新字典与原权重完全独立，修改互不影响


def restore_model_state(model: torch.nn.Module, state: dict) -> None:
    """
    将之前保存的 state_dict 恢复到模型（strict=True，完全覆盖）。

    与 copy_model_state 配合使用，实现"保存和恢复"。

    参数：
      model : 目标模型（in-place 修改）
      state : 之前通过 copy_model_state 保存的状态字典
    """
    model.load_state_dict(state)  # 严格加载（不允许键缺失），完全用 state 覆盖当前权重
