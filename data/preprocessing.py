"""MRI 体积与表格数据的预处理工具函数。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import torch


# --------------------------------------------------------------------------- #
#  MRI 预处理                                                                   #
# --------------------------------------------------------------------------- #

def preprocess_mri(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int] = (96, 96, 96),
) -> torch.Tensor:
    """
    对 3D MRI 体积进行裁剪、归一化和插值。

    处理流程：
      1. 去除多余的通道维度（如果输入是 4D 则取第 0 个通道）
      2. 百分位裁剪：将体素值限制在 [0.5th, 99.5th] 百分位范围内，
         消除极端噪声和伪影的影响
      3. 线性归一化到 [0, 1]
      4. 转为 PyTorch 张量，插值到目标尺寸（三线性插值）
      5. 线性变换到 [-1, 1]（扩散模型的标准输入范围）

    参数：
      volume       : numpy 数组，形状为 (D, H, W) 或 (1, D, H, W)
      target_shape : 输出体积的目标尺寸 (D, H, W)

    返回：
      float32 张量，形状 (1, D, H, W)，值域 [-1, 1]
    """
    import torch.nn.functional as F

    # 如果输入是 4D（带通道维度），去掉第 0 维
    if volume.ndim == 4:
        volume = volume[0]

    # 百分位裁剪：抑制体素极值（如头颅外的噪声区域）
    lo, hi = np.percentile(volume, 0.5), np.percentile(volume, 99.5)
    volume = np.clip(volume, lo, hi)

    # 归一化到 [0, 1]；若 lo == hi（均匀体积），分母置 1 避免除零
    denom = hi - lo if hi != lo else 1.0
    volume = (volume - lo) / denom

    # 转为张量，形状扩展为 (1, 1, D, H, W) 以满足 F.interpolate 输入要求
    t = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)

    # 三线性插值到目标尺寸
    t = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)

    # 从 [0, 1] 线性变换到 [-1, 1]
    t = t * 2.0 - 1.0

    return t.squeeze(0)  # 去掉 batch 维度，输出形状：(1, D, H, W)


# --------------------------------------------------------------------------- #
#  表格预处理                                                                    #
# --------------------------------------------------------------------------- #

# 需要转换为 float 的连续型字段（其他字段按类别型处理）
_CONTINUOUS_FIELDS = ["age", "BMI", "weight", "height"]

# 在 get_task_fields 中忽略的标识符字段（不作为预测目标）
_IGNORE_FIELDS = ["participant_id", "filename"]


def preprocess_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    基本表格清洗：
      1. 统一缺失值标记（n/a、NA、空字符串等统一替换为 NaN）
      2. 将连续型字段（年龄、BMI 等）强制转换为 float 类型
         （无法转换的值设为 NaN）

    参数：
      df : 原始元数据 DataFrame

    返回：
      清洗后的 DataFrame（深拷贝，不修改原始数据）
    """
    df = df.copy()
    # 统一缺失值标记
    df.replace(["n/a", "N/A", "NA", "nan", "", " "], np.nan, inplace=True)
    # 连续型字段转 float
    for col in _CONTINUOUS_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_task_fields(df: pd.DataFrame) -> Tuple[list, list, list]:
    """
    从 DataFrame 中提取可用于预测任务的字段列表。

    过滤规则：
      - 排除标识符字段（participant_id、filename）
      - 排除全为空值的字段
      - 剩余字段按连续/类别型分类

    返回：
      (continuous_fields, categorical_fields, all_task_fields)
      - continuous_fields  : 连续型字段名列表（如 age、BMI）
      - categorical_fields : 类别型字段名列表（如 sex、diagnosis）
      - all_task_fields    : 全部任务字段（= 连续 + 类别）
    """
    # 排除忽略字段，保留至少有一个非空值的列
    all_fields = [c for c in df.columns if c not in _IGNORE_FIELDS and df[c].notna().any()]
    cont = [c for c in all_fields if c in _CONTINUOUS_FIELDS]
    cat = [c for c in all_fields if c not in _CONTINUOUS_FIELDS]
    return cont, cat, all_fields
