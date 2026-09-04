"""将 OpenNeuro 子数据集 ID 分配到跨站点 / 私有 / 公开等角色。"""
from __future__ import annotations  # 允许注解中使用尚未定义的类名
import random                        # Python 内置随机模块（用于打乱和分配 ID 列表）
import warnings                      # cross 候选不足时发出警告，而非静默降级
from typing import Dict, List, Optional, Tuple  # 类型注解工具


def assign_ds_splits(
    all_ds_ids: List[str],
    n_cross: int,
    n_private_per_node: int,
    n_clients: int,
    seed: int = 42,
    quality_stats: Optional[Dict[str, Dict[str, float]]] = None,
    max_missing_ratio: float = 0.5,
    n_public_max: Optional[int] = None,
) -> Dict[str, List[str]]:
    """
    将所有子数据集 ID 随机分配到以下角色（各角色之间互不重叠）：

      'cross'               : 跨站点评估集（要求年龄字段非空，在调用前预过滤）
      'server_private'      : 服务器私有数据集
      'client_0_private'    : 客户端 0 的私有数据集
      'client_1_private'    : 客户端 1 的私有数据集（以此类推）
      'public'              : 服务器公开数据集（用于 MedMD 热身预训练），取剩余池的前
                              n_public_max 个（n_public_max 为 None 时取剩余全部）

    分配顺序（先到先得，按 pool 剩余量分配）：
      1. 从随机打乱的池中依次挑选 n_cross 个 -> cross
         （若提供 quality_stats，跳过不达标的 ID，留给私有/公开角色；见下）
      2. 取接下来 n_private_per_node * (1 + n_clients) 个 -> 私有（服务器 + 各客户端）
      3. 剩余中的前 n_public_max 个（不传则为剩余全部）-> public

    注意：数据质量预筛选（年龄非空率检查，即完全排除全空数据集）应在调用此函数之前完成，
    本函数默认只负责随机分配，不做质量过滤。

    quality_stats（可选）用于在挑选 cross 时按缺失率二次把关：
      - 传入 {ds_id: {field: missing_ratio, ...}, ...}（如 age/sex/health_status 各自的 NaN 占比）
      - 挑选 cross 时，只有该 ds_id 在 quality_stats 中出现、且所有字段的 missing_ratio
        都 <= max_missing_ratio 才会被选入 cross；不达标或未出现在 quality_stats 中的 ID
        会保留在 pool 里，仍可能被分配给私有/公开角色
      - 若合格 ID 不足 n_cross 个，cross 会少于 n_cross 并发出 UserWarning（不报错，
        以免因数据质量波动中断整个实验流程）
      - 不传（默认 None）时行为与原来完全一致，不做质量过滤

    参数：
      all_ds_ids         : 全部可用子数据集 ID 列表
      n_cross            : 分配给跨站点评估的数量
      n_private_per_node : 每个节点（服务器 + 每个客户端）的私有数据集数量
      n_clients          : 客户端数量
      seed               : 随机种子（保证可复现性）
      quality_stats      : 可选，{ds_id: {field: missing_ratio}}，用于筛选 cross 候选
      max_missing_ratio  : quality_stats 各字段缺失率的最大允许值（默认 0.5，即 50%）
      n_public_max       : public 角色的最大子数据集数量上限；为 None 时不设上限，
                            取 cross/私有分配后剩余的全部子数据集

    返回：
      字典，键为角色名，值为对应的子数据集 ID 列表
    """
    rng = random.Random(seed)   # 创建独立随机数生成器（不影响全局 random 状态）
    pool = list(all_ds_ids)     # 复制 ID 列表（避免修改原始列表）
    rng.shuffle(pool)            # 随机打乱，保证分配的随机性（种子固定时结果可复现）

    result: Dict[str, List[str]] = {}  # 存放各角色的 ID 列表

    def _passes_quality(ds_id: str) -> bool:
        """quality_stats 未提供时视为直接通过；提供时要求各字段缺失率均不超过阈值。"""
        if quality_stats is None:
            return True
        stats = quality_stats.get(ds_id)
        if stats is None:   # 未统计到该 ds_id（如枚举口径不一致），保守地视为不合格
            return False
        return all(ratio <= max_missing_ratio for ratio in stats.values())

    # 分配跨站点评估集：按随机顺序挑选，quality_stats 存在时跳过不达标的 ID
    cross: List[str] = []      # 已选入 cross 的 ID
    remaining: List[str] = []  # 未选入 cross 的 ID（不达标或已凑够 n_cross），留给私有/公开
    for ds_id in pool:
        if len(cross) < n_cross and _passes_quality(ds_id):
            cross.append(ds_id)
        else:
            remaining.append(ds_id)
    if len(cross) < n_cross:
        warnings.warn(
            f"[assign_ds_splits] 仅找到 {len(cross)}/{n_cross} 个满足质量要求"
            f"（各字段缺失率 <= {max_missing_ratio:.0%}）的 cross 候选数据集，"
            f"cross 集合将少于配置的 n_cross。",
            stacklevel=2,
        )
    pool = remaining           # 更新剩余池（未入选 cross 的 ID，含不达标的）
    result["cross"] = cross    # 写入结果字典

    # 分配私有数据集（服务器 1 个节点 + n_clients 个客户端节点）
    total_private = n_private_per_node * (1 + n_clients)  # 总私有数据集数量 = (1服务器 + n客户端) × 每节点数
    private_pool = pool[:total_private]  # 从剩余池中取出私有数据集
    pool = pool[total_private:]          # 更新剩余池

    # 服务器取前 n_private_per_node 个私有数据集
    result["server_private"] = private_pool[:n_private_per_node]

    # 各客户端依次取 n_private_per_node 个私有数据集
    for i in range(n_clients):
        start = n_private_per_node * (i + 1)                       # 第 i 个客户端的起始下标
        result[f"client_{i}_private"] = private_pool[start: start + n_private_per_node]  # 切片分配

    # public 取剩余池的前 n_public_max 个（不设上限则为剩余全部），磁盘空间由 LRU 缓存管理
    result["public"] = pool if n_public_max is None else pool[:n_public_max]
    return result   # 返回包含所有角色分配结果的字典


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
    indices = list(range(n))         # 生成 0 到 n-1 的索引列表
    rng = random.Random(seed)        # 创建独立随机数生成器
    rng.shuffle(indices)              # 随机打乱索引顺序
    cut = int(n * train_frac)         # 计算训练集结束位置（向下取整）
    return indices[:cut], indices[cut:]  # 前段为训练集索引，后段为测试集索引
