"""将元数据表格的一行转换为自然语言文本字符串，用作扩散模型的文本条件。"""
from __future__ import annotations
import pandas as pd
from typing import Optional


# 各字段的自然语言模板（字段名 -> 格式字符串）
# {v} 占位符在运行时被字段值替换
_FIELD_TEMPLATES = {
    "age":           "age is {v}",
    "sex":           "sex is {v}",
    "diagnosis":     "diagnosis is {v}",
    "health_status": "health status is {v}",
    "BMI":           "BMI is {v}",
    "weight":        "weight is {v} kg",
    "height":        "height is {v} cm",
    "task":          "task is {v}",
    "session":       "session is {v}",
}

# 未在上述模板中定义的字段使用的通用格式（{k} 为字段名，{v} 为字段值）
_DEFAULT_TEMPLATE = "{k} is {v}"


def table_row_to_text(row: pd.Series, exclude_fields: Optional[list] = None) -> str:
    """
    将元数据表格的一行转换为逗号分隔的自然语言描述字符串。

    示例输出："age is 32, sex is female, diagnosis is healthy"

    处理逻辑：
      1. 跳过 exclude_fields 和固定忽略字段（participant_id、filename）
      2. 跳过 NaN 值（缺失字段不出现在文本中）
      3. 对已知字段使用专用模板，其余字段使用通用模板
      4. 若所有字段均为空，返回默认描述 "brain MRI scan"

    参数：
      row            : pd.Series，元数据表格的一行
      exclude_fields : 额外需要排除的字段列表（如任务特定的非条件字段）

    返回：
      str，自然语言条件文本
    """
    exclude = set(exclude_fields or [])
    exclude.update(["participant_id", "filename"])  # 标识符字段始终排除

    parts = []
    for field, val in row.items():
        if field in exclude:
            continue
        if pd.isna(val):                           # 跳过缺失值
            continue
        if field in _FIELD_TEMPLATES:
            parts.append(_FIELD_TEMPLATES[field].format(v=val))
        else:
            parts.append(_DEFAULT_TEMPLATE.format(k=field, v=val))

    if not parts:
        return "brain MRI scan"                    # 全空时的默认文本条件
    return ", ".join(parts)


def age_only_text(age_val) -> str:
    """
    生成仅包含年龄信息的文本条件（GPT-FL 方法使用）。

    GPT-FL 中服务器仅接收客户端的年龄标签（不共享其他元数据），
    因此其条件文本只包含年龄信息。
    """
    return f"age is {age_val}"
