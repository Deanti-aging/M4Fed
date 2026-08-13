"""数据质量统计工具：记录训练/评估过程中被判定为不可用而排除的样本。

进程内维护一个全局列表，记录每条不可用样本的标识（HuggingFace 仓库内路径）
和排除原因，导出为 CSV，供实验报告统计真实数据的可用率。
与 utils/timing_utils.py 的全局注册表 + CSV 导出风格保持一致。
"""
from __future__ import annotations   # 允许注解中使用尚未定义的类名
import csv                           # 标准库：CSV 文件读写
import os                            # 标准库：文件路径与目录操作
import threading                     # 标准库：线程锁，保证并发写入时数据一致
from typing import List, Tuple       # 类型注解工具


class _UnusableRegistry:
    """进程内全局不可用样本记录表：[(identifier, reason), ...]。"""

    def __init__(self):
        self._records: List[Tuple[str, str]] = []   # 记录列表（允许同一 identifier 重复出现）
        self._lock = threading.Lock()                # 线程锁，保证多线程/多次调用下数据一致

    def add(self, identifier: str, reason: str) -> None:
        """追加一条不可用样本记录。"""
        with self._lock:
            self._records.append((identifier, reason))

    def snapshot(self) -> List[Tuple[str, str]]:
        """返回当前记录表的副本（避免外部代码修改内部数据）。"""
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        """清空全局记录表（用于单元测试或多次独立运行之间互不干扰）。"""
        with self._lock:
            self._records.clear()


# 全局单例记录表（模块级变量，整个进程共享一个实例）
_REGISTRY = _UnusableRegistry()


def record_unusable_sample(identifier: str, reason: str) -> None:
    """记录一条被判定为不可用而排除的样本（如 NIfTI 文件重新下载后仍损坏）。"""
    _REGISTRY.add(identifier, reason)


def save_unusable_csv(path: str) -> None:
    """将全局不可用样本记录表导出为 CSV（列：identifier, reason）。"""
    records = _REGISTRY.snapshot()
    ensure_dir = os.path.dirname(path) or "."
    os.makedirs(ensure_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "reason"])
        writer.writerows(records)


def reset_unusable() -> None:
    """清空全局不可用样本记录表（用于单元测试或多次独立运行之间互不干扰）。"""
    _REGISTRY.reset()
