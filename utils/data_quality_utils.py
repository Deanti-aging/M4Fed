"""数据质量统计工具：记录训练/评估过程中被判定为不可用而排除的样本。

进程内维护一个全局列表，记录每条不可用样本的标识（HuggingFace 仓库内路径）
和排除原因，导出为 CSV，供实验报告统计真实数据的可用率。
与 utils/timing_utils.py 的全局注册表 + CSV 导出风格保持一致。

跨运行持久化（data/known_unusable_samples.csv）：
  新确认的不可用样本会被追加写入项目 data/ 目录下的 known_unusable_samples.csv
  （随 git 一起提交，团队/多次运行共享同一份"已知损坏"清单），而不只是留在
  当次运行的内存记录里。data.openmind_dataset.make_openmind_sample_loader 在
  构建样本时会读取这份清单，把已确认损坏的样本直接标记为 _unusable，
  跳过一次注定失败的下载尝试。results/.../unusable_samples.csv 由本模块的
  save_unusable_csv 导出，内容即该持久化清单（含本次运行新发现的部分）。
"""
from __future__ import annotations   # 允许注解中使用尚未定义的类名
import csv                           # 标准库：CSV 文件读写
import os                            # 标准库：文件路径与目录操作
import threading                     # 标准库：线程锁，保证并发写入时数据一致
from typing import Dict, List, Tuple  # 类型注解工具

# 持久化的"已知不可用样本"清单路径：<项目根目录>/data/known_unusable_samples.csv
# 与 data/openmind_dataset.py 中 _DEFAULT_MRI_LOCAL_DIR 使用同样的相对定位方式
# （utils/ 与 data/ 是项目根目录下的同级目录）
DEFAULT_KNOWN_UNUSABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "known_unusable_samples.csv"
)


def _read_known_unusable_file(path: str) -> Dict[str, str]:
    """读取持久化清单文件，返回 {identifier: reason} 字典（文件不存在时返回空字典）。"""
    if not os.path.exists(path):
        return {}
    records: Dict[str, str] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            identifier = row.get("identifier")
            if identifier:
                records[identifier] = row.get("reason", "")
    return records


def load_known_unusable(path: str = DEFAULT_KNOWN_UNUSABLE_PATH) -> Dict[str, str]:
    """
    加载项目内持久化记录的已知不可用样本清单（默认 data/known_unusable_samples.csv）。

    供 make_openmind_sample_loader 在构建样本字典时查询：命中的样本直接标记
    _unusable=True，OpenMindDataset._load_mri 会在第一步就短路跳过，
    不再重复触发一次已确认会失败的下载。

    参数：
      path : 清单文件路径（默认项目内的持久化文件）

    返回：
      {identifier: reason} 字典（identifier 即样本的 hf_image_path）
    """
    return _read_known_unusable_file(path)


class _UnusableRegistry:
    """
    进程内不可用样本记录表：[(identifier, reason), ...]。

    启动时从持久化文件（DEFAULT_KNOWN_UNUSABLE_PATH）加载已有记录作为基线，
    之后每次 add() 一条新记录，除了追加到内存列表，若 identifier 此前未出现过，
    还会立即追加写入持久化文件——这样即使当次运行中途崩溃，已确认损坏的样本
    也不会丢失，下次运行能立刻跳过。
    """

    def __init__(self, persist_path: str = DEFAULT_KNOWN_UNUSABLE_PATH):
        self._persist_path = persist_path
        self._known: Dict[str, str] = _read_known_unusable_file(persist_path)  # 已持久化的基线
        self._records: List[Tuple[str, str]] = list(self._known.items())       # 当前快照（允许运行中重复 add 同一 identifier）
        self._lock = threading.Lock()                # 线程锁，保证多线程/多次调用下数据一致

    def add(self, identifier: str, reason: str) -> None:
        """追加一条不可用样本记录；identifier 首次出现时同步落盘到持久化清单。"""
        with self._lock:
            self._records.append((identifier, reason))
            if identifier not in self._known:
                self._known[identifier] = reason
                self._append_to_disk(identifier, reason)

    def _append_to_disk(self, identifier: str, reason: str) -> None:
        """将新发现的不可用样本追加写入持久化 CSV 文件（跨运行生效，可提交进 git）。"""
        is_new_file = not os.path.exists(self._persist_path)
        os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
        with open(self._persist_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(["identifier", "reason"])
            writer.writerow([identifier, reason])

    def snapshot(self) -> List[Tuple[str, str]]:
        """返回当前记录表的副本（避免外部代码修改内部数据）。"""
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        """
        将进程内记录表重置为持久化文件中的基线内容（用于单元测试或多次独立运行之间互不干扰）。

        注意：不会删除持久化文件本身——那是跨运行共享的"已知损坏"清单，
        reset 只清掉本次运行期间在内存里新增的临时记录。
        """
        with self._lock:
            self._known = _read_known_unusable_file(self._persist_path)
            self._records = list(self._known.items())


# 全局单例记录表（模块级变量，整个进程共享一个实例）
_REGISTRY = _UnusableRegistry()


def record_unusable_sample(identifier: str, reason: str) -> None:
    """记录一条被判定为不可用而排除的样本（如 NIfTI 文件重新下载后仍损坏）。

    同时会把首次出现的 identifier 追加写入 data/known_unusable_samples.csv，
    供后续运行的 make_openmind_sample_loader 直接跳过下载。
    """
    _REGISTRY.add(identifier, reason)


def save_unusable_csv(path: str) -> None:
    """将不可用样本记录表导出为 CSV（列：identifier, reason）。

    内容即持久化清单（data/known_unusable_samples.csv）叠加本次运行新发现的记录，
    因为 _REGISTRY 启动时已加载持久化基线，新记录又会同步写回该文件。
    """
    records = _REGISTRY.snapshot()
    ensure_dir = os.path.dirname(path) or "."
    os.makedirs(ensure_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "reason"])
        writer.writerows(records)


def reset_unusable() -> None:
    """重置不可用样本记录表（用于单元测试或多次独立运行之间互不干扰，详见 _UnusableRegistry.reset）。"""
    _REGISTRY.reset()
