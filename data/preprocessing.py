"""MRI 体积与表格数据的预处理工具函数。"""
from __future__ import annotations   # 允许注解中使用尚未定义的类名
import numpy as np                   # NumPy：数组运算（百分位裁剪、归一化）
import pandas as pd                  # Pandas：DataFrame 操作（表格预处理）
from typing import Tuple, Optional   # 类型注解工具
import torch                         # PyTorch：张量操作与插值


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
    import torch.nn.functional as F   # PyTorch 函数式 API（用于 F.interpolate 三线性插值）

    # 如果输入是 4D（带通道维度，如 (1, D, H, W)），去掉第 0 维得到 (D, H, W)
    if volume.ndim == 4:
        volume = volume[0]

    # 百分位裁剪：计算 0.5th 和 99.5th 百分位，将极端噪声截断到该范围内
    lo, hi = np.percentile(volume, 0.5), np.percentile(volume, 99.5)
    volume = np.clip(volume, lo, hi)   # 将所有体素值裁剪到 [lo, hi] 范围

    # 归一化到 [0, 1]：若 lo == hi（均匀体积），分母置 1 避免除零
    denom = hi - lo if hi != lo else 1.0  # 计算值域跨度，均匀体积时设为 1.0
    volume = (volume - lo) / denom         # 线性归一化：(x - min) / (max - min)

    # 转为张量，形状扩展为 (1, 1, D, H, W) 以满足 F.interpolate 的输入格式要求（N, C, D, H, W）
    t = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)

    # 三线性插值（trilinear）到目标尺寸；align_corners=False 是 PyTorch 推荐设置
    t = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)

    # 从 [0, 1] 线性变换到 [-1, 1]（扩散模型通常要求输入在此范围内）
    t = t * 2.0 - 1.0

    return t.squeeze(0)  # 去掉 batch 维度（N 维），输出形状：(1, D, H, W)


# --------------------------------------------------------------------------- #
#  表格预处理                                                                    #
# --------------------------------------------------------------------------- #

# 需要转换为 float 的连续型字段（其他字段按类别型处理）
# 对应 OpenMind 的 openneuro_metadata.csv 中的真实列名（均为小写）
_CONTINUOUS_FIELDS = ["age", "weight", "bmi"]

# 在 get_task_fields 中忽略的标识符/扫描技术字段（不作为预测目标或文本条件）
# 对应 openneuro_metadata.csv 中除人口学/健康字段外的所有列
_IGNORE_FIELDS = [
    "unique_id",                    # 唯一标识符，不作为特征
    "modality",                     # 扫描模态（T1w/T2w 等），不作为预测目标
    "image_quality_score",          # 图像质量分数，扫描技术字段
    "manufacturer",                 # 设备制造商（如 Siemens、GE），不作为条件
    "model_name",                   # 设备型号，技术字段
    "phase_encoding_direction",     # 相位编码方向，MRI 扫描参数
    "magnetic_field_strength",      # 磁场强度（1.5T/3T 等），扫描参数
    "repetition_time",              # 重复时间（TR），MRI 扫描参数
    "echo_time",                    # 回波时间（TE），MRI 扫描参数
    "is_brain_extract",             # 是否已做脑提取，处理状态标志
    "derived_from",                 # 派生来源，文件关系字段
    "image_path",                   # NIfTI 文件路径，文件系统字段
    "anat_mask_path",               # 解剖结构掩膜路径，文件路径
    "anon_mask_path",               # 匿名化掩膜路径，文件路径
]


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
    df = df.copy()   # 深拷贝，避免修改传入的原始 DataFrame（保持纯函数性）
    # 统一缺失值标记：将各种缺失值写法统一替换为 NaN（pandas 标准空值）
    df.replace(["n/a", "N/A", "NA", "nan", "", " "], np.nan, inplace=True)
    # 连续型字段转 float：errors="coerce" 表示遇到无法解析的字符串时设为 NaN（而非报错）
    for col in _CONTINUOUS_FIELDS:
        if col in df.columns:                                          # 只处理 DataFrame 中存在的列
            df[col] = pd.to_numeric(df[col], errors="coerce")         # 强制转数值，失败时设 NaN
    return df  # 返回清洗后的 DataFrame（深拷贝，不影响原始数据）


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
    # 排除忽略字段，并过滤掉全为空值的列（这些列没有预测价值）
    all_fields = [c for c in df.columns if c not in _IGNORE_FIELDS and df[c].notna().any()]
    cont = [c for c in all_fields if c in _CONTINUOUS_FIELDS]      # 属于连续型字段集合的列
    cat = [c for c in all_fields if c not in _CONTINUOUS_FIELDS]   # 不在连续型字段集合中的列（视为类别型）
    return cont, cat, all_fields  # 返回三元组：连续字段、类别字段、全部任务字段
