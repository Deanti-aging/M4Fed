"""
M4Fed 敏感性分析（Sensitivity Analysis）。

对三个关键超参数进行独立的网格搜索，每次仅改变一个变量，其余保持默认值：

  1. 合成数据比例（synth_ratio）：[0.1, 1.0, 10.0]
     控制每轮联邦循环和本地训练中生成样本数与真实样本数的比值。
     0.1 → 生成较少合成数据，10.0 → 生成 10 倍于真实数据的合成数据。

  2. MedMD 学习率（lr_medmd）：[1e-5, 1e-4, 1e-3]
     控制扩散模型 U-Net 的训练速度与收敛稳定性的平衡。

  3. 数据异质性（Dirichlet α）：[0.1, 0.5, 1.0]
     α → 0 时数据分布极不均匀（高度 Non-IID），α → ∞ 时趋向 IID。
     通过 dirichlet_resplit() 重新划分各节点私有数据来模拟不同异质程度。

每个配置点均运行完整的 M4Fed 协议，汇总结果保存为 CSV。
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import os, sys, copy                  # os：路径；sys：路径修改；copy：深拷贝
import numpy as np                    # NumPy（随机划分、Dirichlet 采样）
import pandas as pd                   # Pandas（汇总表格）

# 将项目根目录加入 Python 路径（使实验脚本可以直接运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import load_config                   # 加载 YAML 配置文件
from utils.seed_utils import set_seed             # 随机种子设置（保证可复现）
from utils.io_utils import ensure_dir, save_csv   # 目录创建和 CSV 保存
from data.openmind_dataset import build_node_datasets, OpenMindDataset, make_openmind_sample_loader
# build_node_datasets : 构建服务器/客户端数据集字典
# OpenMindDataset     : 数据集类（用于访问 .samples 属性）
# make_openmind_sample_loader : 创建 HuggingFace 样本加载函数
from federated.server import Server               # 服务器节点
from federated.client import Client               # 客户端节点
from federated.methods import run_m4fed           # M4Fed 完整协议
from evaluation.visualization import plot_loss_curves   # 损失曲线绘图（用于可视化扫描对比）


def _build(cfg, node_datasets, device):
    """
    快速构建服务器和 2 个客户端参与者对象（敏感性分析的辅助函数）。

    参数：
      cfg          : 当前扫描点的配置（已深拷贝，已修改目标超参数）
      node_datasets: 节点数据集字典（由 build_node_datasets 返回）
      device       : 计算设备（"cuda" 或 "cpu"）

    返回：
      (server, clients) 元组，clients 是长度为 2 的列表
    """
    # 初始化服务器（含公开数据 server_public 和私有数据 server_private）
    server = Server(cfg, node_datasets["server_private"],
                    node_datasets["server_public"], device)
    # 初始化 2 个客户端（client_0 和 client_1，只含私有数据）
    clients = [Client(i, cfg, node_datasets[f"client_{i}"], device)
               for i in range(2)]
    return server, clients   # 返回参与者元组


def _base_datasets(cfg, all_ds_ids, mri_shape):
    """
    构建基础节点数据集（作为敏感性分析各扫描点的共同起点）。

    各扫描点通过深拷贝（copy.deepcopy）此数据集后修改，避免扫描点之间数据相互污染。

    参数：
      cfg        : M4FedConfig（用于读取 hf_dataset_id、cache_dir 等数据参数）
      all_ds_ids : 所有可用的子数据集 ID 列表
      mri_shape  : MRI 体积目标分辨率（元组）

    返回：
      node_datasets 字典（keys: "server_private", "server_public", "client_0", "client_1"）
    """
    sample_loader_fn = make_openmind_sample_loader(
        cfg.data.hf_dataset_id,                 # HuggingFace 数据集仓库 ID
        cache_dir=cfg.data.cache_dir or None,   # 本地缓存目录（None 用默认缓存）
        max_cache_gb=cfg.data.max_cache_gb,     # 最大缓存容量（GB）
    )
    # 构建节点数据集（服务器 + 2 客户端；n_clients=2 固定）
    nd = build_node_datasets(
        all_ds_ids=all_ds_ids,                              # 所有数据集 ID
        n_cross=cfg.data.n_cross_ds,                         # 公开数据集数量
        n_private_per_node=cfg.data.n_private_ds_per_node,  # 每节点私有数据集数量
        n_clients=2,                                         # 敏感性分析固定 2 客户端
        train_split=cfg.data.train_split,                    # 训练/测试划分比例
        mri_shape=mri_shape,                                 # MRI 目标分辨率
        seed=cfg.data.seed,                                  # 数据划分随机种子
        sample_loader_fn=sample_loader_fn,                   # 样本加载函数
    )
    return nd, sample_loader_fn   # 同时返回 sample_loader_fn，供调用方获取 n_classes


def dirichlet_resplit(node_datasets: dict, alpha: float, seed: int) -> dict:
    """
    按 Dirichlet(α) 分布重新划分各节点的私有训练数据，模拟 Non-IID 场景。

    Dirichlet 分布原理：
      Dir([α, α, α]) 采样出三节点的数据比例向量 [p0, p1, p2]（p0+p1+p2=1）。
      - α 越小（如 0.1）：比例向量极端化（一节点获得绝大多数数据），Non-IID 程度高
      - α 越大（如 1.0）：比例向量趋于均匀（[1/3, 1/3, 1/3]），趋向 IID

    算法步骤：
      1. 收集所有节点（server_private + client_0 + client_1）的训练样本（汇总池）
      2. 用 np.random.dirichlet([α, α, α]) 采样三节点的数据比例
      3. 将比例乘以样本总量得到各节点分配数量（累积边界）
      4. 随机打乱样本索引后按累积边界切分
      5. 将切分结果回写到各节点 .samples 属性（原地修改）

    参数：
      node_datasets : build_node_datasets 返回的数据集字典（原地修改）
      alpha         : Dirichlet 浓度参数（越小越 Non-IID）
      seed          : 随机种子（确保不同扫描点之间可复现）

    返回：
      重新划分后的 node_datasets（原地修改并返回，便于链式调用）
    """
    rng = np.random.RandomState(seed)   # 创建独立随机状态（不影响全局种子）
    # 确定参与重划分的节点键名（服务器私有 + 2 客户端私有）
    nodes = ["server_private"] + [f"client_{i}" for i in range(2)]

    # 第一步：汇总所有节点的训练样本（仅从 train 集取，test 集不重划）
    all_samples = []
    for node_key in nodes:
        ds = node_datasets[node_key]["train"]   # 获取该节点的训练数据集
        if hasattr(ds, "samples"):
            all_samples.extend(ds.samples)       # 将该节点所有样本加入汇总池

    n = len(all_samples)         # 总样本数
    if n == 0:
        return node_datasets     # 空数据集时直接返回（避免除零错误）

    # 第二步：Dirichlet 采样各节点的数据比例
    proportions = rng.dirichlet([alpha] * len(nodes))   # 形状 (3,)，和为 1

    # 第三步：将比例转换为整数边界（累积和后取整，最后一节点取剩余样本）
    boundaries = (np.cumsum(proportions) * n).astype(int)  # [end_0, end_1, end_2]
    boundaries[-1] = n   # 确保最后边界恰好等于总样本数（避免浮点取整丢失样本）

    # 第四步：随机打乱所有样本索引（确保各节点收到的数据无偏序）
    indices = rng.permutation(n)   # 生成 [0, n) 的随机排列
    starts = [0] + list(boundaries[:-1])   # 各节点起始索引（[0, end_0, end_1]）

    # 第五步：按边界切分并回写各节点 .samples 属性
    for i, node_key in enumerate(nodes):
        node_idxs = indices[starts[i]:boundaries[i]]     # 该节点分到的样本索引
        new_samples = [all_samples[j] for j in node_idxs]  # 按索引取出样本
        ds = node_datasets[node_key]["train"]              # 目标训练数据集
        if hasattr(ds, "samples"):
            ds.samples = new_samples   # 原地替换样本列表（实现 Non-IID 重划分）

    return node_datasets   # 返回修改后的数据集字典


def run_sensitivity(args, cfg, all_ds_ids, mri_shape, task_specs, results_dir):
    """
    运行三组敏感性分析（合成比例 / MedMD 学习率 / Dirichlet α）并汇总结果。

    每次只改变一个超参数，其余保持 cfg 的默认值（控制变量法）。
    每个扫描点均独立运行完整的 M4Fed 三阶段协议。

    参数：
      args       : 命令行参数
      cfg        : 基础 M4FedConfig（所有扫描点的起点，不直接修改）
      all_ds_ids : 所有可用子数据集 ID 列表
      mri_shape  : MRI 体积目标尺寸（元组）
      task_specs : 任务规格字典（如 {"age": {"type": "regression", "n_classes": 1}}）
      results_dir: 结果保存根目录

    返回：
      all_sensitivity_results 字典（键为扫描点标签，值为含 param/value/MAE 等字段的字典）
    """
    sens_out = ensure_dir(os.path.join(results_dir, "sensitivity"))  # 创建敏感性分析结果目录
    device = cfg.training.device   # 计算设备

    # 构建基础节点数据集（各扫描点深拷贝此数据集，避免相互污染）
    nd_base, sample_fn = _base_datasets(cfg, all_ds_ids, mri_shape)
    # 用实际类别数更新 task_specs（覆盖 main.py 中的占位 n_classes）
    for field, n in getattr(sample_fn, "n_classes", {}).items():
        if field in task_specs and n > 0:
            task_specs[field]["n_classes"] = n

    all_sensitivity_results = {}   # 收集所有扫描点的结果

    # ==================================================================
    # 扫描 1：合成数据比例（synth_ratio）
    # 同时修改联邦循环阶段（synth_ratio_fed）和本地训练阶段（synth_ratio_local）
    # synth_ratio = 合成样本数 / 真实样本数（比值越大，合成数据影响越强）
    # ==================================================================
    for ratio in [0.1, 1.0, 10.0]:   # 三个扫描点
        tag = f"synth_ratio_{ratio}"  # 扫描点标签（用作目录名和结果键名）
        print(f"[敏感性] 合成比例 synth_ratio = {ratio}")
        set_seed(cfg.training.seed)     # 重置随机种子（每个扫描点独立初始化）
        cfg_r = copy.deepcopy(cfg)      # 深拷贝配置（避免修改原始配置）
        cfg_r.federated.synth_ratio_fed = ratio    # 修改：联邦循环合成比例
        cfg_r.federated.synth_ratio_local = ratio  # 修改：本地训练合成比例
        nd = copy.deepcopy(nd_base)    # 深拷贝数据集（保证各扫描点数据独立）
        s, c = _build(cfg_r, nd, device)   # 构建参与者
        # 运行完整 M4Fed 协议，结果保存到扫描点专属子目录
        res = run_m4fed(s, c, cfg_r, task_specs,
                        os.path.join(sens_out, tag))
        # 将扫描点信息和客户端均值指标合并到结果字典
        all_sensitivity_results[tag] = {
            "param": "synth_ratio",    # 被扫描的超参数名
            "value": ratio,            # 当前扫描点的超参数值
            **res.get("clients_mean", {}),   # 展开 MAE/RMSE/Pearson_r/R2（跨客户端均值）
        }

    # ==================================================================
    # 扫描 2：MedMD 学习率（lr_medmd）
    # 控制扩散模型 U-Net 的训练速度：学习率过大→不稳定，过小→收敛慢
    # ==================================================================
    for lr in [1e-5, 1e-4, 1e-3]:   # 三个数量级
        tag = f"lr_{lr:.0e}"          # 标签格式如 "lr_1e-05"
        print(f"[敏感性] 学习率 lr_medmd = {lr}")
        set_seed(cfg.training.seed)    # 重置随机种子
        cfg_l = copy.deepcopy(cfg)     # 深拷贝配置
        cfg_l.training.lr_medmd = lr  # 修改：MedMD 扩散模型学习率
        nd = copy.deepcopy(nd_base)   # 深拷贝数据集
        s, c = _build(cfg_l, nd, device)   # 构建参与者
        res = run_m4fed(s, c, cfg_l, task_specs,
                        os.path.join(sens_out, tag))
        all_sensitivity_results[tag] = {
            "param": "lr_medmd",   # 被扫描的超参数名
            "value": lr,           # 当前学习率值
            **res.get("clients_mean", {}),   # 展开均值指标
        }

    # ==================================================================
    # 扫描 3：数据异质性（Dirichlet α）
    # 通过 dirichlet_resplit 重新划分各节点私有数据，模拟不同 Non-IID 程度
    # α=0.1：极端 Non-IID（某节点几乎独占所有数据）
    # α=1.0：接近 IID（各节点数据量近似均等）
    # ==================================================================
    for alpha in [0.1, 0.5, 1.0]:   # 三个 Dirichlet 浓度参数
        tag = f"dirichlet_alpha_{alpha}"   # 标签格式如 "dirichlet_alpha_0.1"
        print(f"[敏感性] 数据异质性 Dirichlet α = {alpha}")
        set_seed(cfg.training.seed)     # 重置随机种子（保证 M4Fed 训练可复现）
        cfg_d = copy.deepcopy(cfg)      # 深拷贝配置（α 不改变模型配置）
        # 深拷贝数据集后按 Dirichlet(α) 重新划分各节点私有训练数据
        nd = dirichlet_resplit(copy.deepcopy(nd_base), alpha=alpha,
                               seed=cfg.training.seed)  # 使用与 M4Fed 相同的种子
        s, c = _build(cfg_d, nd, device)   # 构建参与者（使用重划分后的数据集）
        res = run_m4fed(s, c, cfg_d, task_specs,
                        os.path.join(sens_out, tag))
        all_sensitivity_results[tag] = {
            "param": "dirichlet_alpha",   # 被扫描的超参数名
            "value": alpha,               # 当前 Dirichlet α 值
            **res.get("clients_mean", {}),   # 展开均值指标
        }

    # ------------------------------------------------------------------
    # 汇总并保存结果（CSV 表格）
    # ------------------------------------------------------------------
    # 将所有扫描点结果转为 DataFrame（行为扫描点，列为 param/value/MAE/RMSE 等）
    summary_df = pd.DataFrame(list(all_sensitivity_results.values()))
    summary_df.insert(0, "tag", list(all_sensitivity_results.keys()))  # 插入标签列
    save_csv(summary_df, os.path.join(sens_out, "sensitivity_summary.csv"))   # 保存 CSV
    print("[敏感性] 结果已保存至", sens_out)
    return all_sensitivity_results   # 返回完整敏感性分析结果字典


if __name__ == "__main__":
    import argparse   # 命令行参数解析
    from data.openmind_dataset import list_openmind_ds_ids
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m4fed.yaml")   # 配置文件路径
    p.add_argument("--results_dir", default="results")          # 结果保存目录
    args = p.parse_args()   # 解析命令行参数
    cfg = load_config(args.config)   # 加载 M4FedConfig 配置
    all_ds_ids = list_openmind_ds_ids(cfg.data.hf_dataset_id)
    mri_shape = tuple(cfg.data.mri_shape)
    task_specs = {
        "age":           {"type": "regression",     "n_classes": 1},
        "sex":           {"type": "classification", "n_classes": 2},
        "health_status": {"type": "classification", "n_classes": 2},
        "race":          {"type": "classification", "n_classes": 5},
    }
    run_sensitivity(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)
