"""将 OpenNeuro 子数据集 ID 分配到跨站点 / 私有 / 公开等角色。"""
from __future__ import annotations
import random
from typing import Dict, List, Tuple


def assign_ds_splits(
    all_ds_ids: List[str],
    n_cross: int,
    n_private_per_node: int,
    n_clients: int,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    将所有子数据集 ID 随机分配到以下角色（各角色之间互不重叠）：

      'cross'               : 跨站点评估集（要求年龄 AND 性别字段覆盖率高，在调用前预过滤）
      'server_private'      : 服务器私有数据集
      'client_0_private'    : 客户端 0 的私有数据集
      'client_1_private'    : 客户端 1 的私有数据集（以此类推）
      'public'              : 剩余数据集，作为服务器的公开数据用于 MedMD 热身预训练

    分配顺序（先到先得，按 pool 剩余量分配）：
      1. 从随机打乱的池中取前 n_cross 个 -> cross
      2. 取接下来 n_private_per_node * (1 + n_clients) 个 -> 私有（服务器 + 各客户端）
      3. 剩余全部 -> public

    注意：数据质量预筛选（年龄/性别非空率检查）应在调用此函数之前完成，
    此函数只负责随机分配，不做质量过滤。

    参数：
      all_ds_ids         : 全部可用子数据集 ID 列表
      n_cross            : 分配给跨站点评估的数量
      n_private_per_node : 每个节点（服务器 + 每个客户端）的私有数据集数量
      n_clients          : 客户端数量
      seed               : 随机种子（保证可复现性）

    返回：
      字典，键为角色名，值为对应的子数据集 ID 列表
    """
    rng = random.Random(seed)
    pool = list(all_ds_ids)
    rng.shuffle(pool)           # 随机打乱，保证分配的随机性

    result: Dict[str, List[str]] = {}

    # 分配跨站点评估集
    cross = pool[:n_cross]
    pool = pool[n_cross:]
    result["cross"] = cross

    # 分配私有数据集（服务器 1 个节点 + n_clients 个客户端节点）
    total_private = n_private_per_node * (1 + n_clients)
    private_pool = pool[:total_private]
    pool = pool[total_private:]

    # 服务器取前 n_private_per_node 个
    result["server_private"] = private_pool[:n_private_per_node]

    # 各客户端依次取 n_private_per_node 个
    for i in range(n_clients):
        start = n_private_per_node * (i + 1)
        result[f"client_{i}_private"] = private_pool[start: start + n_private_per_node]

    # 剩余全部作为公开数据
    result["public"] = pool
    return result


def train_test_split_indices(n: int, train_frac: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    将 n 个样本的索引随机划分为训练集和测试集。

    参数：
      n          : 样本总数
      train_frac : 训练集比例（如 0.9 表示 90% 训练、10% 测试）
      seed       : 随机种子

    返回：
      (train_indices, test_indices) 两个整数列表
    """
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)                  # 随机打乱
    cut = int(n * train_frac)             # 划分点
    return indices[:cut], indices[cut:]   # 前段为训练集，后段为测试集
