"""将元数据表格的一行转换为自然语言文本字符串，用作扩散模型的文本条件。"""
from __future__ import annotations   # 允许注解中使用尚未定义的类名
import pandas as pd                  # Pandas：数据帧操作
from typing import Optional          # 类型注解工具

from .preprocessing import _IGNORE_FIELDS  # 导入需要跳过的技术/标识符字段集合

# 各字段的自然语言模板（字段名 -> 格式字符串）
# {v} 占位符在运行时被字段值替换
# 字段名对应 openneuro_metadata.csv 的真实列名（均为小写，如 bmi 而非 BMI）
_FIELD_TEMPLATES = {
    "age":           "age is {v}",           # 年龄字段模板
    "sex":           "sex is {v}",           # 性别字段模板
    "diagnosis":     "diagnosis is {v}",     # 诊断信息模板
    "health_status": "health status is {v}", # 健康状态模板
    "bmi":           "BMI is {v}",           # BMI 字段模板（注意显示为大写 BMI）
    "weight":        "weight is {v} kg",     # 体重字段模板（带单位）
    "height":        "height is {v} cm",     # 身高字段模板（带单位）
    "task":          "task is {v}",          # 任务类型模板（如 rest、language 等）
    "session":       "session is {v}",       # 扫描会话模板
}

# 未在上述模板中定义的字段使用的通用格式（{k} 为字段名，{v} 为字段值）
_DEFAULT_TEMPLATE = "{k} is {v}"


def table_row_to_text(row: pd.Series, exclude_fields: Optional[list] = None) -> str:
    """
    将元数据表格的一行转换为逗号分隔的自然语言描述字符串。

    示例输出："age is 32, sex is female, diagnosis is healthy"

    处理逻辑：
      1. 跳过 exclude_fields 和固定忽略字段（标识符/扫描技术字段，见 preprocessing._IGNORE_FIELDS）
      2. 跳过 NaN 值（缺失字段不出现在文本中）
      3. 对已知字段使用专用模板，其余字段使用通用模板
      4. 若所有字段均为空，返回默认描述 "brain MRI scan"

    参数：
      row            : pd.Series，元数据表格的一行
      exclude_fields : 额外需要排除的字段列表（如任务特定的非条件字段）

    返回：
      str，自然语言条件文本
    """
    exclude = set(exclude_fields or [])   # 将额外排除字段转为集合，方便 O(1) 查找
    exclude.update(_IGNORE_FIELDS)         # 加入固定忽略字段（标识符/扫描技术字段始终排除）

    parts = []  # 存储每个字段生成的文本片段（最后用逗号拼接）
    for field, val in row.items():
        if field in exclude:       # 跳过标识符/扫描技术/额外排除字段
            continue
        if pd.isna(val):           # 跳过缺失值（NaN）：缺失字段不出现在条件文本中
            continue
        if field in _FIELD_TEMPLATES:
            # 使用专用模板（更自然的表达，如 "BMI is 24.5" 而非 "bmi is 24.5"）
            parts.append(_FIELD_TEMPLATES[field].format(v=val))
        else:
            # 使用通用模板（兜底格式，自动处理未在模板中定义的字段）
            parts.append(_DEFAULT_TEMPLATE.format(k=field, v=val))

    if not parts:
        return "brain MRI scan"   # 所有字段均缺失时，返回通用描述作为默认条件
    return ", ".join(parts)        # 将所有字段的文本片段用逗号空格拼接为最终字符串


def age_only_text(age_val) -> str:
    """
    生成仅包含年龄信息的文本条件（GPT-FL 方法使用）。

    GPT-FL 中服务器仅接收客户端的年龄标签（不共享其他元数据），
    因此其条件文本只包含年龄信息。
    """
    return f"age is {age_val}"  # 格式化字符串，如 "age is 35"
