"""
OpenMind 数据集加载器（基于 HuggingFace 流式接口）。

职责：
  - 按跨站点 / 服务器私有 / 客户端私有 / 公开等角色分配子数据集 ID
  - 从磁盘或内存加载 MRI 体积与表格文本
  - 返回各节点的 PyTorch Dataset 对象（供 DataLoader 使用）
"""
from __future__ import annotations
import os
import io
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

# nibabel 用于读取 NIfTI 格式的 MRI 文件；不可用时退回随机噪声体积
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# 同包内依赖
from .preprocessing import preprocess_mri, preprocess_table
from .table_to_text import table_row_to_text
from .splits import assign_ds_splits, train_test_split_indices


# --------------------------------------------------------------------------- #
#  辅助函数：子数据集质量评分                                                    #
# --------------------------------------------------------------------------- #

def _has_mostly_non_null(df: pd.DataFrame, field: str, threshold: float = 0.5) -> bool:
    """
    判断 DataFrame 中某字段的非空值比例是否超过阈值。
    用于筛选年龄、性别等关键字段覆盖率足够高的子数据集。
    """
    if field not in df.columns:
        return False
    non_null = df[field].notna().mean()
    return non_null >= threshold


def _score_ds(df: pd.DataFrame, required_fields: List[str]) -> float:
    """
    计算子数据集的质量分数：所有必要字段的非空值比例的均值。
    分数越高说明该子数据集的标注越完整。
    """
    if df is None or df.empty:
        return 0.0
    return np.mean([df[f].notna().mean() if f in df.columns else 0.0
                    for f in required_fields])


# --------------------------------------------------------------------------- #
#  核心数据集类                                                                  #
# --------------------------------------------------------------------------- #

class OpenMindDataset(Dataset):
    """
    PyTorch Dataset，封装一组 (MRI 体积, 表格文本, 年龄标签, 元数据) 元组。

    支持两种模式：
      - 内存模式：样本列表中直接存储 numpy 数组（mri_array）
      - 惰性加载模式：样本列表中存储文件路径（mri_path），按需读取 NIfTI 文件

    每个样本（dict）的必要字段：
      'mri_path'   或 'mri_array' (np.ndarray)  —— MRI 数据来源
      'table_text' (str)                         —— 表格转文本的描述字符串
      'age'        (float 或 None)               —— 年龄标签（NaN 时不参与损失计算）
      'metadata'   (pd.Series)                   —— 原始元数据行（用于偏差分析）
    """

    def __init__(
        self,
        samples: List[dict],
        mri_shape: Tuple[int, int, int] = (96, 96, 96),
        augment: bool = False,
        seed: int = 42,
    ):
        """
        参数：
          samples   : 样本字典列表（见类说明）
          mri_shape : 预处理后 MRI 体积的目标尺寸 (D, H, W)
          augment   : 是否在 __getitem__ 时进行 3D 数据增强
          seed      : 数据增强的随机种子
        """
        self.samples = samples
        self.mri_shape = tuple(mri_shape)
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        """返回数据集样本总数。"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        返回第 idx 个样本，包含：
          'mri'      : (1, D, H, W) float32 张量，值域 [-1, 1]
          'text'     : str，表格描述文本
          'age'      : 0 维 float32 张量（可能为 nan）
          'metadata' : pd.Series 或 dict，原始元数据
        """
        s = self.samples[idx]

        # 1. 加载 MRI：优先从内存数组，其次从文件路径，最后退回随机噪声
        vol = self._load_mri(s)
        # 2. 预处理：裁剪百分位、归一化到 [-1,1]、插值到目标尺寸
        mri_tensor = preprocess_mri(vol, self.mri_shape)  # 形状：(1, D, H, W)

        # 3. 可选的 3D 数据增强（翻转等 MONAI 变换）
        if self.augment:
            mri_tensor = self._augment_3d(mri_tensor)

        # 4. 构造年龄标签张量（缺失时为 nan，compute_loss 中会被遮蔽）
        age = s.get("age", None)
        age_tensor = torch.tensor(float(age), dtype=torch.float32) if age is not None else torch.tensor(float("nan"))

        return {
            "mri": mri_tensor,                          # (1, D, H, W)
            "text": s.get("table_text", "brain MRI scan"),
            "age": age_tensor,
            "metadata": s.get("metadata", {}),
        }

    # ---------------------------------------------------------------------- #
    def _load_mri(self, s: dict) -> np.ndarray:
        """
        按优先级加载 MRI 数据：
          1. 内存中的 numpy 数组（mri_array）
          2. 磁盘上的 NIfTI 文件（mri_path，需要 nibabel）
          3. 随机噪声体积（兜底 fallback）
        """
        if "mri_array" in s and s["mri_array"] is not None:
            return s["mri_array"]
        if "mri_path" in s and s["mri_path"] and os.path.exists(s["mri_path"]):
            if not HAS_NIBABEL:
                raise ImportError("加载 NIfTI 文件需要安装 nibabel 库")
            img = nib.load(s["mri_path"])
            return img.get_fdata(dtype=np.float32)
        # 兜底：返回随机噪声（仅用于单元测试）
        return np.random.randn(*self.mri_shape).astype(np.float32)

    def _augment_3d(self, t: torch.Tensor) -> torch.Tensor:
        """
        使用 MONAI 实现 3D 随机翻转增强。
        若 MONAI 未安装则直接返回原张量（不报错）。
        """
        try:
            from monai.transforms import (
                RandFlipd, RandRotate90d, RandAffined, Compose
            )
            transforms = Compose([
                RandFlipd(keys=["img"], prob=0.5, spatial_axis=0),  # 沿深度轴随机翻转
                RandFlipd(keys=["img"], prob=0.5, spatial_axis=1),  # 沿高度轴随机翻转
                RandFlipd(keys=["img"], prob=0.5, spatial_axis=2),  # 沿宽度轴随机翻转
            ])
            out = transforms({"img": t})
            return out["img"]
        except Exception:
            return t  # MONAI 不可用时跳过增强


# --------------------------------------------------------------------------- #
#  虚拟样本生成器（单元测试使用）                                                 #
# --------------------------------------------------------------------------- #

def _make_dummy_samples(n: int, mri_shape=(16, 16, 16), seed=0) -> List[dict]:
    """
    生成 n 个微型虚拟样本，用于单元测试，无需任何真实数据。

    生成规则：
      - MRI 体积：标准正态随机噪声
      - 年龄：20-80 之间的随机整数
      - 性别：奇数索引为 male，偶数为 female
    """
    rng = np.random.RandomState(seed)
    samples = []
    for i in range(n):
        age = float(rng.randint(20, 80))
        samples.append({
            "mri_array": rng.randn(*mri_shape).astype(np.float32),
            "table_text": f"age is {int(age)}, sex is {'male' if i % 2 == 0 else 'female'}",
            "age": age,
            "metadata": pd.Series({"age": age, "sex": "male" if i % 2 == 0 else "female",
                                   "health_status": "healthy"}),
        })
    return samples


# --------------------------------------------------------------------------- #
#  高层构建函数：为每个联邦节点返回训练/测试数据集                                 #
# --------------------------------------------------------------------------- #

def build_node_datasets(
    all_ds_ids: List[str],
    n_cross: int,
    n_private_per_node: int,
    n_clients: int,
    train_split: float,
    mri_shape: Tuple[int, int, int],
    seed: int,
    sample_loader_fn=None,   # 可调用对象：ds_id -> List[dict]，从真实数据源加载样本
    dummy_mode: bool = False,
    n_dummy_per_ds: int = 4,
) -> Dict[str, Dict[str, "OpenMindDataset"]]:
    """
    根据数据集 ID 分配方案，构建各联邦节点的训练/测试数据集。

    返回值结构：
      {
        'server_private': {'train': OpenMindDataset, 'test': OpenMindDataset},
        'server_public':  {'train': ..., 'test': ...},
        'client_0':       {'train': ..., 'test': ...},
        'client_1':       {'train': ..., 'test': ...},
        ...
        'cross':          {'all': OpenMindDataset},   # 跨站点评估集，不分 train/test
      }

    参数说明：
      all_ds_ids        : 全部可用子数据集 ID 列表
      n_cross           : 分配给跨站点评估的子数据集数量
      n_private_per_node: 每个节点分配的私有子数据集数量
      n_clients         : 客户端数量
      train_split       : 训练集比例
      mri_shape         : MRI 体积目标尺寸
      seed              : 随机种子
      sample_loader_fn  : 真实数据加载函数（dummy_mode=False 时必须提供）
      dummy_mode        : 是否使用虚拟数据（无需任何真实文件）
      n_dummy_per_ds    : dummy_mode 下每个子数据集生成的虚拟样本数
    """
    # 第一步：将子数据集 ID 分配到各角色（跨站点、服务器私有、客户端私有、公开）
    split_ids = assign_ds_splits(
        all_ds_ids, n_cross, n_private_per_node, n_clients, seed
    )

    def _load_ds(ids: List[str]) -> List[dict]:
        """
        内部辅助：加载指定 ID 列表对应的样本。
        dummy_mode=True 时生成随机噪声样本；否则调用 sample_loader_fn。
        """
        if dummy_mode:
            # 根据 ID 列表生成确定性的随机种子，保证每次 ID 相同时结果一致
            rng_seed = abs(hash(tuple(ids))) % (2**31)
            return _make_dummy_samples(len(ids) * n_dummy_per_ds,
                                       mri_shape=mri_shape, seed=rng_seed)
        if sample_loader_fn is None:
            raise ValueError("非 dummy 模式必须提供 sample_loader_fn 参数")
        samples = []
        for ds_id in ids:
            samples.extend(sample_loader_fn(ds_id))
        return samples

    result: Dict[str, Dict[str, OpenMindDataset]] = {}

    # ---------- 服务器私有数据集 ----------
    s_priv = _load_ds(split_ids["server_private"])
    tr_idx, te_idx = train_test_split_indices(len(s_priv), train_split, seed)
    result["server_private"] = {
        "train": OpenMindDataset([s_priv[i] for i in tr_idx], mri_shape),
        "test":  OpenMindDataset([s_priv[i] for i in te_idx], mri_shape),
    }

    # ---------- 服务器公开数据集（用于 MedMD 热身预训练） ----------
    s_pub = _load_ds(split_ids["public"])
    tr_idx, te_idx = train_test_split_indices(len(s_pub), train_split, seed)
    result["server_public"] = {
        "train": OpenMindDataset([s_pub[i] for i in tr_idx], mri_shape),
        "test":  OpenMindDataset([s_pub[i] for i in te_idx], mri_shape),
    }

    # ---------- 各客户端私有数据集 ----------
    for i in range(n_clients):
        c_priv = _load_ds(split_ids[f"client_{i}_private"])
        tr_idx, te_idx = train_test_split_indices(len(c_priv), train_split, seed)
        result[f"client_{i}"] = {
            "train": OpenMindDataset([c_priv[j] for j in tr_idx], mri_shape),
            "test":  OpenMindDataset([c_priv[j] for j in te_idx], mri_shape),
        }

    # ---------- 跨站点评估集（不划分 train/test，全部用于评估） ----------
    cross_samples = _load_ds(split_ids["cross"])
    result["cross"] = {"all": OpenMindDataset(cross_samples, mri_shape)}

    return result


def make_dataloader(
    dataset: OpenMindDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """
    为 OpenMindDataset 创建 DataLoader。

    使用固定种子的 Generator 保证每次 shuffle 结果可复现。
    collate_fn 将样本列表合并为批次张量（包含变长文本列表）。
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: list) -> dict:
    """
    自定义 collate 函数：
      - MRI 张量堆叠为 (B, 1, D, H, W)
      - 年龄张量堆叠为 (B,)
      - 文本列表保持为 Python list（长度可变，不 pad）
      - 元数据列表保持为 Python list
    """
    mris = torch.stack([b["mri"] for b in batch])
    ages = torch.stack([b["age"] for b in batch])
    texts = [b["text"] for b in batch]
    metadata = [b["metadata"] for b in batch]
    return {"mri": mris, "text": texts, "age": ages, "metadata": metadata}
