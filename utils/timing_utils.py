"""运行耗时统计工具：记录各阶段（网络下载、各 baseline 的训练/测试等）累计耗时，导出为 CSV。

进程内维护一个全局计时表，键为形如 "network_download/metadata_csv"、
"m4fed/train"、"m4fed/eval" 的标签，值为累计秒数与调用次数。
同一标签可在程序运行期间被多次记录（例如消融实验里 run_m4fed 被调用 3 次），
耗时会自动累加，不会相互覆盖。
"""
from __future__ import annotations   # 允许注解中使用尚未定义的类名（PEP 563 延迟求值）
import csv                           # 标准库：CSV 文件读写
import os                            # 标准库：文件路径与目录操作
import threading                     # 标准库：线程锁，保证多线程下计时器的数据一致性
import time                          # 标准库：高精度计时函数 perf_counter
from collections import defaultdict  # 带默认值的字典，简化初始化
from contextlib import contextmanager  # 上下文管理器装饰器，用于 with 语句
from typing import Dict              # 类型注解：字典类型


class _TimingRegistry:
    """进程内全局计时表：{标签: {"seconds": 累计秒数, "calls": 调用次数}}。"""

    def __init__(self):
        # defaultdict：访问不存在的键时自动初始化为 {"seconds": 0.0, "calls": 0}
        self._data: Dict[str, Dict[str, float]] = defaultdict(lambda: {"seconds": 0.0, "calls": 0})
        # 线程锁：确保多线程并发写入时数据不会损坏（DataLoader 多进程时可能并发记录）
        self._lock = threading.Lock()

    def add(self, label: str, seconds: float) -> None:
        """将 seconds 秒累加到 label 条目，并将调用次数加 1。"""
        with self._lock:              # 加锁，保证原子性（防止多线程竞争）
            entry = self._data[label]   # 取出（或自动创建）该标签对应的条目
            entry["seconds"] += seconds  # 累加本次耗时
            entry["calls"] += 1          # 调用次数加 1

    def snapshot(self):
        """返回当前计时表的深拷贝快照（避免外部代码修改内部数据）。"""
        with self._lock:                                        # 加锁，保证读取时数据一致
            return {k: dict(v) for k, v in self._data.items()}  # 深拷贝每个条目字典

    def reset(self) -> None:
        """清空全局计时表（用于单元测试或多次独立运行之间互不干扰）。"""
        with self._lock:      # 加锁，防止清空过程中有其他线程写入
            self._data.clear()  # 清除所有标签条目


# 全局单例计时表（模块级变量，整个进程共享一个实例）
_REGISTRY = _TimingRegistry()


def record_timing(label: str, seconds: float) -> None:
    """将一段耗时累加记录到全局计时表的 label 条目下。"""
    _REGISTRY.add(label, seconds)  # 委托给全局计时表的 add 方法


@contextmanager
def time_block(label: str):
    """计时上下文管理器：代码块执行耗时累加到全局表的 label 条目（适合可自然缩进的代码段）。

    用法示例：
        with time_block("network_download/metadata_csv"):
            csv_path = hf_hub_download(...)
    """
    start = time.perf_counter()  # 记录代码块开始时刻（高精度单调时钟，不受系统时间调整影响）
    try:
        yield               # 将控制权交给 with 块内的代码
    finally:
        # 无论 with 块是否抛出异常，都将耗时记录到计时表
        record_timing(label, time.perf_counter() - start)


def save_timing_csv(path: str) -> None:
    """将全局计时表按总耗时降序导出为 CSV（列：stage, total_seconds, calls, percentage）。"""
    data = _REGISTRY.snapshot()   # 获取当前计时表快照（不影响全局状态）
    # 计算所有标签的总耗时（用于计算百分比）；若总耗时为 0 则设为 1.0 避免除零
    total = sum(v["seconds"] for v in data.values()) or 1.0
    # 按总耗时降序排序，最耗时的阶段排在最前面（方便快速定位瓶颈）
    rows = sorted(data.items(), key=lambda kv: kv[1]["seconds"], reverse=True)
    ensure_dir = os.path.dirname(path) or "."  # 获取父目录路径，空时用当前目录
    os.makedirs(ensure_dir, exist_ok=True)     # 递归创建父目录（exist_ok=True 防止已存在时报错）
    with open(path, "w", newline="", encoding="utf-8") as f:   # 以 utf-8 写模式打开 CSV 文件
        writer = csv.writer(f)   # 创建 CSV 写入器
        # 写入表头行
        writer.writerow(["stage", "total_seconds", "calls", "percentage"])
        for label, v in rows:
            # 逐行写入：标签、保留 4 位小数的总秒数、调用次数、占总耗时的百分比（保留 2 位小数）
            writer.writerow([label, round(v["seconds"], 4), v["calls"],
                              round(100.0 * v["seconds"] / total, 2)])


def reset_timing() -> None:
    """清空全局计时表（用于单元测试或多次独立运行之间互不干扰）。"""
    _REGISTRY.reset()  # 委托给全局计时表的 reset 方法
