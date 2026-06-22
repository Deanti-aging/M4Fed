"""
OpenMind 数据集加载器。

OpenMind（AnonRes/OpenMind）在 HuggingFace 上是按 dsXXXXXX 文件夹组织的原始 NIfTI
文件仓库（外加一份 openneuro_metadata.csv 元数据表），不是打包好的 train split，
因此不通过 datasets.load_dataset 读取，而是：
  - 用 huggingface_hub 列出仓库文件 / 下载 openneuro_metadata.csv 来枚举子数据集 ID
    和样本元数据（list_openmind_ds_ids / make_openmind_sample_loader）
  - 每个样本的 NIfTI 文件在 __getitem__ 中惰性下载（首次访问后命中本地缓存）

职责：
  - 按跨站点 / 服务器私有 / 客户端私有 / 公开等角色分配子数据集 ID
  - 从磁盘、内存或 HuggingFace 仓库加载 MRI 体积与表格文本
  - 返回各节点的 PyTorch Dataset 对象（供 DataLoader 使用）
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import os                              # 路径操作（makedirs、join、exists 等）
import io                              # 字节流（未来扩展用，当前版本未直接使用）
import re                              # 正则表达式（提取 dsXXXXXX 文件夹名）
import random                          # 随机数生成（数据增强的翻转概率）
import time                            # 时间戳记录（LRU 缓存的访问时间）
import numpy as np                     # NumPy（MRI 数组操作、随机噪声生成）
import pandas as pd                    # Pandas（元数据表格处理）
import torch                           # PyTorch（张量操作）
from torch.utils.data import Dataset, DataLoader   # PyTorch 数据集和加载器基类
from typing import Dict, List, Optional, Tuple     # 类型注解工具

# nibabel 用于读取 NIfTI 格式（.nii / .nii.gz）的 MRI 文件；
# 不可用时退回随机噪声体积（仅在单元测试时使用，正式实验必须安装 nibabel）
try:
    import nibabel as nib    # 神经影像文件格式库
    HAS_NIBABEL = True       # 标记 nibabel 已安装
except ImportError:
    HAS_NIBABEL = False      # nibabel 未安装，__getitem__ 将返回随机噪声

# 同包内依赖（相对导入，要求 data/ 目录为 Python 包）
from .preprocessing import preprocess_mri, preprocess_table   # MRI 和表格预处理
from .table_to_text import table_row_to_text                   # 表格行 → 自然语言描述
from .splits import assign_ds_splits, train_test_split_indices  # 数据集划分函数
from utils.timing_utils import time_block, record_timing        # 耗时计时工具


# =========================================================================== #
#  惰性下载文件的 LRU 磁盘缓存（避免训练过程中把磁盘写满）                           #
# =========================================================================== #

# 惰性下载的 NIfTI 文件默认落盘目录（未配置 cache_dir 时使用）
# 路径示例：<项目根目录>/data/mri_cache
_DEFAULT_MRI_LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mri_cache")


class _LRUDiskCache:
    """
    按总字节数限额的简易 LRU 磁盘缓存（按最久未访问优先淘汰策略管理文件）。

    OpenMind 全量约 964GiB/114k 文件，标准 huggingface_hub 缓存（cache_dir）只会
    不断追加、从不淘汰，在磁盘容量有限的机器上会被写满。这里改用
    hf_hub_download(local_dir=...) 把文件下载为 root 目录下的普通文件（而非
    blob+符号链接），这样可以安全地直接删除单个文件来腾出空间：删除后下次访问
    该样本时 hf_hub_download 会发现文件不存在，自动重新下载。

    仅在单进程（num_workers=0，本项目固定如此）内维护内存态的大小/访问时间表，
    初始化时扫描一次 root 目录，之后增量更新，避免每次访问都重新遍历磁盘。
    """

    # 类级别的单例字典：所有实例共享，避免对同一目录重复创建缓存对象
    _instances: Dict[str, "_LRUDiskCache"] = {}

    def __init__(self, root: str, max_bytes: float):
        """
        初始化 LRU 磁盘缓存，扫描 root 目录建立文件大小和访问时间索引。

        参数：
          root      : 缓存根目录（NIfTI 文件存储位置）
          max_bytes : 最大总占用字节数（超出时淘汰最久未访问的文件）
        """
        self.root = root                   # 缓存根目录路径
        self.max_bytes = max_bytes         # 最大磁盘占用字节数（0 或负数表示不限制）
        os.makedirs(root, exist_ok=True)  # 确保缓存目录存在
        self._sizes: Dict[str, int] = {}   # {文件路径: 文件大小（字节）}
        self._atime: Dict[str, float] = {}  # {文件路径: 最近访问时间戳（Unix 时间）}
        # 扫描 root 目录（递归），建立文件大小和修改时间的初始索引
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = os.path.join(dirpath, fn)   # 绝对路径
                try:
                    st = os.stat(p)              # 获取文件元信息（大小、修改时间）
                except OSError:
                    continue                      # 文件可能在扫描期间被删除，跳过
                self._sizes[p] = st.st_size     # 记录文件大小
                self._atime[p] = st.st_mtime    # 用文件修改时间初始化访问时间

    # 是 Python 的一个装饰器，它将方法绑定到类本身而不是类的实例
    @classmethod
    def get(cls, root: str, max_bytes: float) -> "_LRUDiskCache":
        """
        返回（必要时创建）root 目录对应的单例缓存对象。

        单例模式：同一 root 目录对应同一个 _LRUDiskCache 对象（避免内存中
        维护多个独立的大小/访问时间索引导致淘汰判断不一致）。
        """
        inst = cls._instances.get(root)   # 查找已存在的实例
        if inst is None:
            inst = cls(root, max_bytes)    # 首次访问时创建新实例
            cls._instances[root] = inst    # 注册到单例字典
        return inst   # 返回单例

    def note_access(self, path: str) -> None:
        """
        记录 path 刚被访问（新下载或缓存命中），并在超出容量限额时淘汰最久未访问的文件。

        淘汰策略：
          1. 更新被访问文件的大小和时间戳
          2. 检查当前总占用是否超过 max_bytes
          3. 若超出，按访问时间升序排列所有文件，从最旧的开始删除
          4. 直到总占用降至 max_bytes 以下为止
          5. 刚访问的文件（path）绝不被淘汰

        参数：
          path : 刚被访问/下载的文件的绝对路径
        """
        try:
            size = os.path.getsize(path)   # 获取文件实际大小（字节）
        except OSError:
            return   # 文件不存在（竞争条件），直接返回
        self._sizes[path] = size             # 更新文件大小记录
        self._atime[path] = time.time()      # 更新访问时间为当前时间（最新 = 不被淘汰）
        if self.max_bytes <= 0:
            return   # 不限制容量时直接返回（不执行淘汰逻辑）
        total = sum(self._sizes.values())   # 计算当前总占用字节数
        if total <= self.max_bytes:
            return   # 未超出限额，无需淘汰
        # 按访问时间升序排序（最旧的文件排在最前面，优先被淘汰）
        for old_path, _ in sorted(self._atime.items(), key=lambda kv: kv[1]):
            if total <= self.max_bytes:
                break   # 已腾出足够空间，停止淘汰
            if old_path == path:
                continue   # 不淘汰刚访问过的文件（保护当前请求）
            old_size = self._sizes.pop(old_path, 0)   # 从大小索引中移除
            self._atime.pop(old_path, None)            # 从访问时间索引中移除
            try:
                os.remove(old_path)   # 删除磁盘上的旧文件（腾出空间）
                total -= old_size     # 更新总占用计数
            except OSError:
                pass   # 文件可能已被其他进程删除，忽略错误


# =========================================================================== #
#  辅助函数：子数据集质量评分                                                      #
# =========================================================================== #

def _has_mostly_non_null(df: pd.DataFrame, field: str, threshold: float = 0.5) -> bool:
    """
    判断 DataFrame 中某字段的非空值比例是否超过阈值。

    用途：筛选年龄、性别等关键字段覆盖率足够高的子数据集（避免大量 NaN 影响训练）。

    参数：
      df        : 子数据集对应的 metadata DataFrame
      field     : 待检查的字段名（如 "age"）
      threshold : 非空值比例阈值（默认 0.5，即至少 50% 的样本有该字段值）

    返回：
      True 表示覆盖率足够高（可用于训练），False 表示该字段缺失值过多
    """
    if field not in df.columns:
        return False                     # 字段不存在时直接返回 False
    non_null = df[field].notna().mean()  # 计算非空值比例（notna() 返回布尔序列）
    return non_null >= threshold         # 与阈值比较


def _score_ds(df: pd.DataFrame, required_fields: List[str]) -> float:
    """
    计算子数据集的质量分数：所有必要字段的非空值比例的均值。

    分数范围：[0.0, 1.0]，越高说明标注越完整（1.0 = 所有字段均无缺失值）。
    用途：在分配数据集角色时优先选择标注完整的子数据集。

    参数：
      df              : 子数据集对应的 metadata DataFrame
      required_fields : 必须有足够覆盖率的字段列表（如 ["age", "sex"]）

    返回：
      质量分数 float（空 DataFrame 或 None 时返回 0.0）
    """
    if df is None or df.empty:
        return 0.0   # 空数据集质量分数为 0
    # 对每个必要字段计算非空值比例，字段不存在时记为 0
    return np.mean([df[f].notna().mean() if f in df.columns else 0.0
                    for f in required_fields])


# =========================================================================== #
#  核心数据集类                                                                   #
# =========================================================================== #

class OpenMindDataset(Dataset):
    """
    PyTorch Dataset，封装一组 (MRI 体积, 表格文本, 年龄标签, 元数据) 元组。

    支持两种数据来源模式：
      - 内存模式   : 样本字典中直接存储 numpy 数组（mri_array），无需磁盘 I/O
      - 惰性加载模式: 样本字典中存储文件路径（mri_path）或 HuggingFace 路径，
                    按需读取 NIfTI 文件（首次访问时触发下载/磁盘读取）

    每个样本字典的关键字段：
      'mri_path'   or 'mri_array' (np.ndarray) —— MRI 数据来源（二选一）
      'table_text' (str)                        —— 表格转文本的自然语言描述字符串
      'age'        (float or None)              —— 年龄标签（NaN 时不参与损失计算）
      'metadata'   (pd.Series or dict)          —— 原始元数据行（用于偏差分析）
    """

    def __init__(
        self,
        samples: List[dict],                      # 样本字典列表（见类说明）
        mri_shape: Tuple[int, int, int] = (96, 96, 96),  # 预处理后 MRI 目标尺寸
        augment: bool = False,                    # 是否在 __getitem__ 时进行 3D 数据增强
        seed: int = 42,                           # 数据增强的随机种子
    ):
        """
        参数：
          samples   : 样本字典列表（每个元素是一个包含 mri_path/mri_array/table_text/age/metadata 的字典）
          mri_shape : 预处理后 MRI 体积的目标尺寸 (D, H, W)（深度/高度/宽度）
          augment   : 是否在 __getitem__ 时进行 3D 随机翻转数据增强（训练时开启）
          seed      : 随机种子（用于数据增强的确定性初始化）
        """
        self.samples = samples              # 保存样本列表（可在消融实验中直接修改 .samples）
        self.mri_shape = tuple(mri_shape)  # 目标 MRI 分辨率（转为元组确保不可变）
        self.augment = augment              # 数据增强开关
        self.rng = random.Random(seed)     # 独立随机状态（用于增强的随机翻转决策）

    def __len__(self) -> int:
        """返回数据集样本总数（PyTorch DataLoader 用于确定 epoch 长度）。"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        返回第 idx 个样本，包含：
          'mri'      : (1, D, H, W) float32 张量，值域 [-1, 1]（经预处理归一化）
          'text'     : str，表格描述文本（用于 MedMD 文本条件的 BioMedCLIP 编码）
          'age'      : 0 维 float32 张量（可能为 nan，NaN 样本在损失计算时被遮蔽）
          'metadata' : pd.Series 或 dict，原始元数据（用于偏差分析时的分组）

        参数：
          idx : 样本索引（0 到 len-1）
        """
        s = self.samples[idx]   # 取出第 idx 个样本字典

        # 步骤 1：加载 MRI 体积
        # 优先顺序：内存数组（mri_array）> 本地磁盘文件（mri_path）>
        #           HuggingFace 惰性下载（hf_image_path）> 随机噪声（兜底）
        vol = self._load_mri(s)                           # 加载原始体积（numpy ndarray）

        # 步骤 2：MRI 预处理（百分位裁剪 → [-1,1] 归一化 → 插值到目标尺寸）
        mri_tensor = preprocess_mri(vol, self.mri_shape)  # 输出形状：(1, D, H, W)

        # 步骤 3：可选的 3D 数据增强（随机翻转，减少过拟合）
        if self.augment:
            mri_tensor = self._augment_3d(mri_tensor)   # MONAI 随机翻转

        # 步骤 4：构造年龄标签张量
        age = s.get("age", None)   # 从样本字典获取年龄值（可能为 None/float/int）
        # 缺失时设为 NaN（在 compute_loss 中通过 NaN mask 遮蔽该样本的损失贡献）
        age_tensor = (torch.tensor(float(age), dtype=torch.float32)
                      if age is not None else torch.tensor(float("nan")))

        return {
            "mri": mri_tensor,                             # (1, D, H, W) float32 张量
            "text": s.get("table_text", "brain MRI scan"),  # 文本条件（无时用默认值）
            "age": age_tensor,                             # 0 维 float32 张量（可能为 nan）
            "metadata": s.get("metadata", {}),            # 元数据（pd.Series 或空 dict）
        }

    # ---------------------------------------------------------------------- #
    def _load_mri(self, s: dict) -> np.ndarray:
        """
        按优先级加载 MRI 数据，返回原始体积数组（numpy ndarray, float32）。

        优先级顺序：
          1. 内存中的 numpy 数组（mri_array）——最快，无 I/O
          2. 本地磁盘上的 NIfTI 文件（mri_path）——需要 nibabel
          3. HuggingFace 仓库中的远程 NIfTI 文件（hf_repo_id + hf_image_path）——
             惰性下载，下载后写回 s["mri_path"] 缓存为本地路径
          4. 随机噪声体积——兜底 fallback（仅单元测试中使用）

        参数：
          s : 样本字典

        返回：
          numpy float32 数组（形状不固定，后续 preprocess_mri 会插值到目标尺寸）
        """
        # 优先级 1：直接使用内存中的 numpy 数组
        if "mri_array" in s and s["mri_array"] is not None:
            return s["mri_array"]   # 返回内存数组（虚拟模式/合成数据使用此路径）

        # 优先级 2 & 3：尝试从本地磁盘或 HuggingFace 获取文件路径
        mri_path = s.get("mri_path")           # 本地磁盘路径（可能为 None）
        if not mri_path and s.get("hf_image_path"):
            mri_path = self._resolve_hf_path(s)   # 惰性下载 HuggingFace 文件

        if mri_path and os.path.exists(mri_path):
            if not HAS_NIBABEL:
                raise ImportError("加载 NIfTI 文件需要安装 nibabel 库")
            img = nib.load(mri_path)            # 使用 nibabel 加载 NIfTI 文件
            return img.get_fdata(dtype=np.float32)  # 返回 float32 体积数组

        # 优先级 4（兜底）：返回随机噪声体积
        return np.random.randn(*self.mri_shape).astype(np.float32)

    @staticmethod
    def _resolve_hf_path(s: dict) -> str:
        """
        从 HuggingFace 仓库惰性下载 s["hf_image_path"]（首次访问时下载，
        之后命中 LRU 磁盘缓存），并将解析出的本地路径写回 s["mri_path"]，
        使同一样本的后续 __getitem__ 调用无需重新下载。

        设计细节：
          - 使用 local_dir 参数下载为普通文件（而非默认的 blob+符号链接缓存），
            便于 _LRUDiskCache 在超出容量时直接 os.remove() 删除旧文件腾出空间
          - 在惰性下载前检查目标路径是否已存在（缓存命中时跳过网络请求）
          - 下载后调用 _LRUDiskCache.note_access() 更新访问时间并按需淘汰旧文件

        参数：
          s : 样本字典（含 hf_repo_id、hf_image_path、hf_local_dir、hf_max_cache_bytes）

        返回：
          本地文件的绝对路径（可直接传给 nibabel 读取）
        """
        from huggingface_hub import hf_hub_download   # HuggingFace Hub 下载函数
        local_dir = s.get("hf_local_dir") or _DEFAULT_MRI_LOCAL_DIR  # 本地存储目录
        # 将 HuggingFace 仓库内的相对路径映射为本地绝对路径
        target_path = os.path.join(local_dir, *s["hf_image_path"].split("/"))

        if not os.path.exists(target_path):
            # 文件不在缓存中，触发网络下载（计时，记录网络下载耗时）
            with time_block("network_download/mri_files"):
                hf_hub_download(
                    repo_id=s["hf_repo_id"],           # HuggingFace 仓库 ID
                    filename=s["hf_image_path"],        # 仓库内相对路径（如 ds000001/sub-01/.../T1w.nii.gz）
                    repo_type="dataset",                # 仓库类型（数据集，非模型）
                    local_dir=local_dir,               # 下载到普通文件（而非 HF blob+symlink 缓存）
                )

        # 将本地路径写回样本字典（下次 _load_mri 直接走优先级 2，跳过 HuggingFace 路径解析）
        s["mri_path"] = target_path

        # 通知 LRU 缓存此文件被访问（按需淘汰旧文件以控制磁盘占用）
        max_cache_bytes = s.get("hf_max_cache_bytes")   # 最大磁盘占用（字节）
        if max_cache_bytes:
            _LRUDiskCache.get(local_dir, max_cache_bytes).note_access(target_path)

        return target_path   # 返回本地文件绝对路径

    def _augment_3d(self, t: torch.Tensor) -> torch.Tensor:
        """
        使用 MONAI 实现 3D 随机翻转增强（沿 3 个空间轴各 50% 概率翻转）。

        MONAI 增强原理：RandFlipd 在 Dict 模式下对指定 key（"img"）进行随机翻转，
        spatial_axis 指定翻转轴（0=深度 D, 1=高度 H, 2=宽度 W）。

        若 MONAI 未安装则直接返回原张量（跳过增强，不抛出异常），
        以保证代码在无 MONAI 的环境下也能运行。

        参数：
          t : (1, D, H, W) float32 MRI 张量

        返回：
          增强后的 (1, D, H, W) 张量（MONAI 可用时）或原张量（MONAI 不可用时）
        """
        try:
            from monai.transforms import (
                RandFlipd, RandRotate90d, RandAffined, Compose   # MONAI 数据增强变换
            )
            transforms = Compose([
                RandFlipd(keys=["img"], prob=0.5, spatial_axis=0),  # 50% 概率沿深度轴翻转
                RandFlipd(keys=["img"], prob=0.5, spatial_axis=1),  # 50% 概率沿高度轴翻转
                RandFlipd(keys=["img"], prob=0.5, spatial_axis=2),  # 50% 概率沿宽度轴翻转
            ])
            out = transforms({"img": t})   # 以 Dict 格式传入（MONAI Dict 变换的输入格式）
            return out["img"]              # 取出增强后的张量
        except Exception:
            return t   # MONAI 不可用或增强失败时返回原始张量（静默跳过增强）


# =========================================================================== #
#  子数据集 ID 枚举（从 HuggingFace 仓库文件结构中提取，而非 load_dataset）          #
# =========================================================================== #

def list_openmind_ds_ids(repo_id: str = "AnonRes/OpenMind", max_ids: int = 300) -> List[str]:
    """
    枚举 OpenMind 仓库中的子数据集 ID（如 ds000001、ds000002 ...）。

    背景：OpenMind 在 HuggingFace 上是按 dsXXXXXX 文件夹组织的原始 NIfTI 文件集合
    （外加一份 openneuro_metadata.csv 元数据表），并不是打包成 train split 的
    Arrow/Parquet 数据集，因此无法用 load_dataset(..., split="train") 读出
    "dataset_id"/"ds_id" 字段。这里改为直接列出仓库文件，
    从顶层路径前缀中提取 dsXXXXXX 文件夹名。

    参数：
      repo_id : HuggingFace 仓库 ID（默认 "AnonRes/OpenMind"）
      max_ids : 最多枚举的子数据集数量（避免 API 请求耗时过长）

    返回：
      排序后的子数据集 ID 字符串列表（如 ["ds000001", "ds000002", ...]）
    """
    from huggingface_hub import HfApi           # HuggingFace Hub API 客户端
    api = HfApi()                               # 创建 API 实例
    with time_block("network_download/list_repo_files"):   # 计时：列出仓库文件（网络请求）
        files = api.list_repo_files(repo_id, repo_type="dataset")  # 枚举仓库中所有文件路径
    pattern = re.compile(r"^(ds\d+)/")         # 正则：匹配以 dsXXXXXX/ 开头的路径
    ids = set()                                 # 用 set 去重（同一 ds 目录下有多个文件）
    for f in files:
        m = pattern.match(f)                   # 尝试匹配文件路径的前缀
        if m:
            ids.add(m.group(1))                # 提取 dsXXXXXX 部分（第一捕获组）
            if len(ids) >= max_ids:
                break                          # 达到上限后停止（避免枚举全量文件耗时过长）
    return sorted(ids)   # 排序（确保跨机器/多次运行结果一致）


def list_ds_ids_from_metadata(
    csv_path: str,
    modality: str = "T1w",
) -> List[str]:
    """
    从本地 openneuro_metadata.csv 提取有效子数据集 ID 列表。

    过滤条件与 make_openmind_sample_loader 内部 _sample_loader_fn 一致：
      - modality 列等于指定模态（默认 "T1w"）
      - age 字段非空（脑龄预测任务要求年龄标签存在）

    这样得到的 ds_id 列表保证每个 ID 在 sample_loader_fn 中至少能返回一个样本，
    不会把空桶分配给联邦节点。

    参数：
      csv_path : 本地 openneuro_metadata.csv 的文件路径
      modality : 保留的 MRI 模态（传 None 则不按模态过滤）

    返回：
      排序后的子数据集 ID 字符串列表（覆盖 CSV 中全部满足条件的 ds_id）
    """
    from data.preprocessing import preprocess_table
    df = pd.read_csv(csv_path)
    df = preprocess_table(df)                              # 统一缺失值标记、age 转 float
    if modality is not None:
        df = df[df["modality"] == modality]                # 只保留指定模态的行
    df = df[df["age"].notna()]                             # 排除 age 缺失的受试者
    ds_ids = df["image_path"].str.split("/").str[0].dropna().unique()
    return sorted(ds_ids.tolist())


# =========================================================================== #
#  真实样本加载器（从 HuggingFace 仓库 + openneuro_metadata.csv 构建样本）          #
# =========================================================================== #

# openneuro_metadata.csv 中描述受试者人口学/健康状况的字段
# 用于生成文本条件与年龄标签（不包含扫描技术字段，避免泄露设备信息）
_PATIENT_FIELDS = ["age", "weight", "bmi", "sex", "handedness", "race", "health_status"]


def _load_openmind_metadata(repo_id: str, cache_dir: Optional[str], modality: Optional[str]) -> pd.DataFrame:
    """
    下载并解析 OpenMind 的 openneuro_metadata.csv（首次调用后命中 huggingface_hub 本地缓存）。

    CSV 列结构（真实字段）：
      unique_id, modality, image_quality_score, age, weight, bmi, sex, handedness,
      race, health_status, manufacturer, model_name, phase_encoding_direction,
      magnetic_field_strength, repetition_time, echo_time, is_brain_extract,
      derived_from, image_path, anat_mask_path, anon_mask_path

    image_path 形如 "ds000001/sub-01/anat/sub-01_T1w.nii.gz"，
    其首段即子数据集 ID（与 list_openmind_ds_ids 枚举出的 ID 一致）。

    参数：
      repo_id   : HuggingFace 仓库 ID
      cache_dir : CSV 文件的本地缓存目录（None 时使用 huggingface_hub 默认目录）
      modality  : 仅保留该模态的行（如 "T1w"，brain age 任务使用的结构像）；
                  传 None 则保留全部模态

    返回：
      经 preprocess_table 清洗后的 pd.DataFrame，新增 "ds_id" 列
    """
    from huggingface_hub import hf_hub_download   # HuggingFace 文件下载函数
    with time_block("network_download/metadata_csv"):   # 计时：元数据 CSV 下载
        csv_path = hf_hub_download(
            repo_id=repo_id,                     # HuggingFace 仓库 ID
            filename="openneuro_metadata.csv",   # CSV 文件名（仓库根目录下）
            repo_type="dataset",                 # 仓库类型
            cache_dir=cache_dir,                 # 本地缓存目录（命中时跳过下载）
        )
    df = pd.read_csv(csv_path)          # 读取 CSV 为 DataFrame
    df = preprocess_table(df)           # 清洗表格：统一列名、类型转换、处理异常值
    if modality is not None:
        df = df[df["modality"] == modality]   # 筛选指定模态的行（如只保留 T1w 结构像）
    # 从 image_path 提取子数据集 ID（取第一个 "/" 之前的部分）
    df["ds_id"] = df["image_path"].str.split("/").str[0]   # 如 "ds000001/..." → "ds000001"
    return df   # 返回包含 ds_id 列的 DataFrame


def make_openmind_sample_loader(
    repo_id: str = "AnonRes/OpenMind",    # HuggingFace 仓库 ID
    cache_dir: Optional[str] = None,      # 元数据 CSV 和 NIfTI 文件的本地缓存目录
    modality: str = "T1w",               # 使用的 MRI 模态（brain age 任务用 T1w 结构像）
    max_cache_gb: Optional[float] = 50.0,  # 惰性下载 NIfTI 文件的磁盘容量上限（GiB）
):
    """
    构建可传给 build_node_datasets(sample_loader_fn=...) 的真实数据加载函数（闭包）。

    设计思路：
      - 一次性下载 openneuro_metadata.csv 并按 ds_id 分组到字典中（O(N) 预处理）
      - 返回的闭包函数 sample_loader_fn(ds_id) → List[dict] 直接查字典（O(1)）
      - 每个样本字典包含 HuggingFace 惰性下载所需的所有信息（hf_repo_id 等），
        实际 NIfTI 文件在 OpenMindDataset.__getitem__ 中首次访问时才触发下载

    样本字典的字段：
      'hf_repo_id'          : HuggingFace 仓库 ID（供 hf_hub_download 使用）
      'hf_image_path'       : 仓库内 NIfTI 文件的相对路径
      'hf_local_dir'        : 本地 NIfTI 缓存目录（下载后存放位置）
      'hf_max_cache_bytes'  : LRU 缓存容量上限（字节），用于 _LRUDiskCache 淘汰判断
      'table_text'          : 仅由人口学/健康字段（_PATIENT_FIELDS）生成的自然语言描述
                              （不泄露扫描设备/文件路径等技术字段，用于文本条件）
      'age'                 : 年龄标签（float 或 None）
      'metadata'            : 完整的 pd.Series 行（含扫描技术字段，供偏差分析使用）

    参数：
      cache_dir    : 元数据 CSV 与惰性下载的 NIfTI 文件的本地存放根目录；
                     为 None 时 NIfTI 文件落在 _DEFAULT_MRI_LOCAL_DIR
      max_cache_gb : 惰性下载的 NIfTI 文件总占用空间上限（GiB）；
                     None 或 <=0 表示不限制（不建议在磁盘紧张的机器上使用）

    返回：
      sample_loader_fn 闭包（接受 ds_id 字符串，返回样本字典列表）
    """
    # 一次性下载并解析元数据 CSV（按 T1w 模态筛选，只保留结构像行）
    df = _load_openmind_metadata(repo_id, cache_dir, modality)
    # 按 ds_id 分组，建立 {ds_id: DataFrame} 字典（O(1) 查询每个子数据集的行）
    grouped = {ds_id: g for ds_id, g in df.groupby("ds_id")}

    # 确定 NIfTI 文件的本地存放目录
    mri_local_dir = (os.path.join(cache_dir, "mri_files")  # 指定 cache_dir 时的子目录
                     if cache_dir else _DEFAULT_MRI_LOCAL_DIR)  # 未指定时用默认目录
    # 将 GiB 转换为字节（用于 LRU 缓存的容量判断）
    max_cache_bytes = (max_cache_gb * (1024 ** 3)   # GiB → 字节
                       if max_cache_gb and max_cache_gb > 0 else None)   # 不限制时为 None

    def _sample_loader_fn(ds_id: str) -> List[dict]:
        """
        为指定子数据集 ID 构建样本字典列表（闭包，捕获 grouped、mri_local_dir 等变量）。

        只收录 age 字段非空的行：脑龄预测是 domain task，没有年龄标签的受试者
        既不参与训练，其 NIfTI 文件也不会被下载，从而节省磁盘和网络开销。

        参数：
          ds_id : 子数据集 ID（如 "ds000001"）

        返回：
          该子数据集的样本字典列表（空列表表示该 ds_id 不存在、无 T1w 文件或全部年龄缺失）
        """
        g = grouped.get(ds_id)               # 查找该子数据集的元数据 DataFrame
        if g is None or g.empty:
            return []                         # 未找到或为空时返回空列表
        samples = []
        for _, row in g.iterrows():          # 遍历该子数据集的每一行（每行对应一个 NIfTI 文件）
            age = row.get("age")             # 提取年龄值（可能为 NaN）
            # 脑龄预测任务要求年龄标签：跳过年龄为空的受试者，不触发 NIfTI 文件下载
            if pd.isna(age):
                continue
            samples.append({
                "hf_repo_id": repo_id,                         # HuggingFace 仓库 ID
                "hf_image_path": row["image_path"],             # NIfTI 文件仓库内相对路径
                "hf_local_dir": mri_local_dir,                 # 本地 NIfTI 文件缓存目录
                "hf_max_cache_bytes": max_cache_bytes,          # LRU 缓存容量（字节）
                # 只用人口学/健康字段生成文本（reindex 确保列顺序一致且不包含技术字段）
                "table_text": table_row_to_text(row.reindex(_PATIENT_FIELDS)),
                "age": float(age),   # 已确认非 NaN，直接转 float
                "metadata": row,                                # 完整行（含扫描设备信息，供偏差分析）
            })
        return samples   # 返回该子数据集中年龄非空的所有样本字典

    return _sample_loader_fn   # 返回闭包函数（作为 build_node_datasets 的 sample_loader_fn 参数）


# =========================================================================== #
#  虚拟样本生成器（单元测试和快速冒烟测试使用）                                        #
# =========================================================================== #

def _make_dummy_samples(n: int, mri_shape=(16, 16, 16), seed=0) -> List[dict]:
    """
    生成 n 个微型虚拟样本，用于单元测试和快速冒烟测试，无需任何真实数据。

    生成规则：
      - MRI 体积：标准正态随机噪声（shape=mri_shape, dtype=float32）
      - 年龄：20-80 之间的随机整数（均匀分布）
      - 性别：奇数索引为 female，偶数为 male（交替赋值保证性别平衡）
      - 健康状态：固定为 "healthy"（简化验证逻辑）

    参数：
      n         : 要生成的样本数量
      mri_shape : 随机 MRI 体积的形状（虚拟模式通常用 (16, 16, 16)）
      seed      : 随机种子（确保每次生成相同的虚拟数据集）

    返回：
      包含 n 个样本字典的列表
    """
    rng = np.random.RandomState(seed)   # 创建独立随机状态（不影响全局种子）
    samples = []
    for i in range(n):
        age = float(rng.randint(20, 80))   # 生成 20-80 之间的随机整数年龄
        samples.append({
            "mri_array": rng.randn(*mri_shape).astype(np.float32),   # 随机 MRI 噪声体积
            "table_text": f"age is {int(age)}, sex is {'male' if i % 2 == 0 else 'female'}",  # 简单文本描述
            "age": age,   # 年龄标签
            "metadata": pd.Series({
                "age": age,                                        # 年龄（用于偏差分析）
                "sex": "male" if i % 2 == 0 else "female",        # 性别（交替）
                "health_status": "healthy",                        # 健康状态（固定值）
            }),
        })
    return samples   # 返回虚拟样本列表


# =========================================================================== #
#  高层构建函数：为每个联邦节点返回训练/测试数据集                                    #
# =========================================================================== #

def build_node_datasets(
    all_ds_ids: List[str],            # 全部可用子数据集 ID 列表
    n_cross: int,                     # 分配给跨站点评估的子数据集数量
    n_private_per_node: int,          # 每个节点分配的私有子数据集数量
    n_clients: int,                   # 客户端数量
    train_split: float,               # 训练集比例（剩余为测试集）
    mri_shape: Tuple[int, int, int],  # MRI 体积目标尺寸（D, H, W）
    seed: int,                        # 随机种子（数据划分 + DataLoader shuffle）
    sample_loader_fn=None,            # 可调用对象：ds_id → List[dict]（非 dummy 模式必须提供）
    dummy_mode: bool = False,         # 是否使用虚拟随机噪声数据（跳过真实数据加载）
    n_dummy_per_ds: int = 4,          # dummy 模式下每个子数据集生成的虚拟样本数
    n_public_max: Optional[int] = None,  # "public" 角色最多分配的子数据集数量
) -> Dict[str, Dict[str, "OpenMindDataset"]]:
    """
    根据数据集 ID 分配方案，构建各联邦节点的训练/测试数据集。

    返回值结构（嵌套字典）：
      {
        'server_private': {'train': OpenMindDataset, 'test': OpenMindDataset},
        'server_public':  {'train': OpenMindDataset, 'test': OpenMindDataset},
        'client_0':       {'train': OpenMindDataset, 'test': OpenMindDataset},
        'client_1':       {'train': OpenMindDataset, 'test': OpenMindDataset},
        ...
        'cross':          {'all': OpenMindDataset},   # 跨站点评估集（不分 train/test）
      }

    参数说明：
      all_ds_ids        : 全部可用子数据集 ID 列表（来自 list_openmind_ds_ids 或虚拟 ID）
      n_cross           : 分配给跨站点评估的子数据集数量（不参与训练）
      n_private_per_node: 每个节点分配的私有子数据集数量（服务器私有 + 每客户端私有）
      n_clients         : 联邦学习客户端数量
      train_split       : 训练集比例（如 0.9 表示 90% 训练、10% 测试）
      mri_shape         : MRI 体积预处理目标尺寸（如 (96, 96, 96)）
      seed              : 随机种子（保证数据划分可复现）
      sample_loader_fn  : 真实数据加载函数（dummy_mode=False 时必须提供）
      dummy_mode        : True 时使用随机噪声替代真实 NIfTI 文件（快速测试）
      n_dummy_per_ds    : dummy_mode 下每个子数据集生成的虚拟样本数
      n_public_max      : "public" 角色最多分配的子数据集数量（防止全量下载导致磁盘满）
    """
    _build_t0 = time.perf_counter()   # 记录构建开始时间（用于耗时统计）

    # 第一步：将子数据集 ID 按角色分配
    # assign_ds_splits 返回 {role: [ds_id, ...]} 字典
    split_ids = assign_ds_splits(
        all_ds_ids, n_cross, n_private_per_node, n_clients, seed,
        n_public_max=n_public_max,   # 公开数据集上限（避免磁盘溢出）
    )

    def _load_ds(ids: List[str]) -> List[dict]:
        """
        内部辅助函数：为指定 ID 列表加载样本。

        dummy_mode=True 时生成确定性随机噪声样本（基于 ID 列表的哈希值作为种子）；
        否则对每个 ds_id 调用 sample_loader_fn 加载真实数据。

        参数：
          ids : 子数据集 ID 字符串列表

        返回：
          合并后的样本字典列表
        """
        if dummy_mode:
            # 从 ID 列表派生确定性种子（相同 ID 集合总是生成相同的虚拟样本）
            rng_seed = abs(hash(tuple(ids))) % (2**31)   # hash 值取绝对值后取模
            return _make_dummy_samples(len(ids) * n_dummy_per_ds,   # 总样本数 = ID数 × 每DS样本数
                                       mri_shape=mri_shape, seed=rng_seed)
        if sample_loader_fn is None:
            raise ValueError("非 dummy 模式必须提供 sample_loader_fn 参数")
        samples = []
        for ds_id in ids:
            samples.extend(sample_loader_fn(ds_id))   # 逐 ID 加载并合并样本
        return samples   # 返回合并后的样本列表

    result: Dict[str, Dict[str, OpenMindDataset]] = {}   # 初始化结果字典

    # ---------- 服务器私有数据集（用于服务器端 NeuroFM 本地训练和评估） ----------
    s_priv = _load_ds(split_ids["server_private"])           # 加载服务器私有样本
    tr_idx, te_idx = train_test_split_indices(len(s_priv), train_split, seed)  # 划分训练/测试
    result["server_private"] = {
        "train": OpenMindDataset([s_priv[i] for i in tr_idx], mri_shape),  # 训练集
        "test":  OpenMindDataset([s_priv[i] for i in te_idx], mri_shape),  # 测试集
    }

    # ---------- 服务器公开数据集（用于 MedMD 热身预训练 + VAE 校准） ----------
    s_pub = _load_ds(split_ids["public"])                    # 加载公开数据集样本
    tr_idx, te_idx = train_test_split_indices(len(s_pub), train_split, seed)
    result["server_public"] = {
        "train": OpenMindDataset([s_pub[i] for i in tr_idx], mri_shape),   # 训练集
        "test":  OpenMindDataset([s_pub[i] for i in te_idx], mri_shape),   # 测试集
    }

    # ---------- 各客户端私有数据集（每个客户端拥有自己的私有数据，不与其他节点共享） ----------
    for i in range(n_clients):
        c_priv = _load_ds(split_ids[f"client_{i}_private"])   # 加载第 i 个客户端的私有样本
        tr_idx, te_idx = train_test_split_indices(len(c_priv), train_split, seed)
        result[f"client_{i}"] = {
            "train": OpenMindDataset([c_priv[j] for j in tr_idx], mri_shape),  # 训练集
            "test":  OpenMindDataset([c_priv[j] for j in te_idx], mri_shape),  # 测试集
        }

    # ---------- 跨站点评估集（不参与训练，全部样本用于泛化性评估） ----------
    cross_samples = _load_ds(split_ids["cross"])             # 加载跨站点样本
    result["cross"] = {
        "all": OpenMindDataset(cross_samples, mri_shape)      # 所有样本合并（不分 train/test）
    }

    # 记录 build_node_datasets 的总耗时（含 _load_ds 中的元数据组织开销）
    # 注：真实网络 I/O（CSV/NIfTI 下载）已在 time_block("network_download/...") 中单独计时
    record_timing("data_prep/build_node_datasets", time.perf_counter() - _build_t0)
    return result   # 返回各节点的训练/测试数据集字典


def make_dataloader(
    dataset: OpenMindDataset,   # 待封装的 OpenMindDataset 对象
    batch_size: int = 1,        # 每个批次的样本数（MRI 体积大，通常为 1）
    shuffle: bool = True,       # 是否随机打乱样本顺序（训练时 True，评估时 False）
    num_workers: int = 0,       # DataLoader 工作线程数（0=主进程加载，保证可复现性）
    seed: int = 42,             # DataLoader Generator 随机种子（保证 shuffle 可复现）
) -> DataLoader:
    """
    为 OpenMindDataset 创建 PyTorch DataLoader。

    设计细节：
      - 使用固定种子的 torch.Generator 保证 shuffle 结果可复现（跨 epoch 和多次运行一致）
      - 自定义 collate_fn（_collate_fn）将样本字典列表合并为批次字典：
          MRI/年龄张量用 torch.stack 合并，文本/元数据保持为 Python list
      - num_workers=0：惰性 NIfTI 下载需要在主进程中执行（多进程 worker 中 fork
        后 HuggingFace 认证状态可能失效）

    参数：
      dataset     : 数据集对象（OpenMindDataset 或其子集）
      batch_size  : 批大小
      shuffle     : 是否打乱顺序
      num_workers : 工作线程数（推荐保持 0）
      seed        : DataLoader 随机数生成器的种子

    返回：
      配置好的 PyTorch DataLoader 对象
    """
    generator = torch.Generator()             # 创建独立的随机数生成器
    generator.manual_seed(seed)               # 设置种子（保证 shuffle 可复现）
    return DataLoader(
        dataset,                              # 数据集对象
        batch_size=batch_size,               # 批大小
        shuffle=shuffle,                      # 是否打乱
        num_workers=num_workers,              # 工作线程数
        pin_memory=False,                     # 不固定内存（避免 CUDA 设备内存压力）
        generator=generator,                  # 固定种子的随机数生成器
        collate_fn=_collate_fn,              # 自定义批次合并函数
    )


def _collate_fn(batch: list) -> dict:
    """
    自定义 collate 函数：将单个样本字典列表合并为批次字典。

    合并规则：
      - 'mri'      : 堆叠为 (B, 1, D, H, W) float32 张量（B=批大小）
      - 'age'      : 堆叠为 (B,) float32 张量（可能含 NaN）
      - 'text'     : 保持为 Python list（长度=B，文本长度各不相同，不需要 padding）
      - 'metadata' : 保持为 Python list（pd.Series 列表，供偏差分析使用）

    参数：
      batch : 长度为 B 的样本字典列表（每个元素是 __getitem__ 的返回值）

    返回：
      批次字典（keys: 'mri', 'text', 'age', 'metadata'）
    """
    mris = torch.stack([b["mri"] for b in batch])       # 沿 batch 维度堆叠 MRI 张量
    ages = torch.stack([b["age"] for b in batch])        # 沿 batch 维度堆叠年龄张量
    texts = [b["text"] for b in batch]                   # 文本保持为 list（不 pad）
    metadata = [b["metadata"] for b in batch]            # 元数据保持为 list
    return {"mri": mris, "text": texts, "age": ages, "metadata": metadata}
