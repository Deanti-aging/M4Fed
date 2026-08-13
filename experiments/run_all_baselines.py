"""
运行全部 5 个联邦学习基线方法并生成对比图表。

5 个方法：
  M4Fed  —— 本论文提出的方法（服务器 + 2 个客户端）
  FedAvg —— 联邦平均（3 个客户端）
  FedProx—— FedAvg + 近端正则项（3 个客户端）
  FAN    —— 联邦增强网络，去中心化合成数据交换（3 个客户端）
  GPT-FL —— 服务器端文本条件生成（3 个客户端）

客户端数量差异：
  M4Fed 使用服务器 + 2 个客户端的三方结构（服务器持有公开数据）；
  其余基线使用 3 个纯客户端结构（无区分服务器的公开数据）。
  因此 FedAvg/FedProx/FAN/GPT-FL 都重新构建了 3 客户端的数据集划分。

用法：
  python experiments/run_all_baselines.py --config configs/m4fed.yaml
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import argparse                        # 命令行参数解析
import os                              # 路径操作
import sys                             # 系统路径修改
import copy                            # 深拷贝（为各方法复制配置对象）
import numpy as np                     # NumPy（统计计算）
import pandas as pd                    # Pandas（汇总表格保存）

# 将项目根目录加入 Python 路径（使实验脚本可以直接运行，不依赖 package install）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import load_config           # 加载 YAML 配置文件
from utils.seed_utils import set_seed     # 随机种子设置（保证可复现）
from utils.io_utils import ensure_dir, save_csv  # 目录创建和 CSV 保存
from data.openmind_dataset import (       # 数据集构建工具
    build_node_datasets, list_openmind_ds_ids, make_openmind_sample_loader
)
from data.splits import assign_ds_splits  # 数据集划分辅助函数
from federated.server import Server       # 服务器节点
from federated.client import Client       # 客户端节点
from federated.methods import run_m4fed, run_fedavg, run_fedprox, run_fan, run_gptfl  # 5 个方法
from evaluation.visualization import (    # 可视化函数
    plot_loss_curves, plot_metric_bars, save_figure,
    plot_scatter, plot_age_bin_mae, plot_residuals, plot_error_heatmap,
    save_all_bias_figures,
)
from evaluation.metrics import compute_domain_metrics, compute_bias_analysis   # 评估指标函数


def parse_args():
    """解析命令行参数：配置文件路径、是否虚拟模式、结果保存目录。"""
    p = argparse.ArgumentParser()   # 创建参数解析器
    p.add_argument("--config", default="configs/m4fed.yaml")   # 配置文件路径（默认 m4fed.yaml）
    p.add_argument("--results_dir", default="results")   # 结果保存根目录
    return p.parse_args()   # 解析并返回参数命名空间


def build_participants(cfg, node_datasets: dict, device: str, n_clients: int):
    """
    从预构建的节点数据集字典实例化 Server 和 Client 对象。

    参数：
      cfg          : 当前方法的配置对象（M4FedConfig）
      node_datasets: build_node_datasets 返回的节点数据集字典
      device       : 计算设备（'cuda' 或 'cpu'）
      n_clients    : 客户端数量（M4Fed 为 2，其他基线为 3）

    返回：
      (server, clients) 元组
        - server  : Server 实例（含公开数据和私有数据）
        - clients : Client 实例列表（长度 = n_clients）
    """
    # 初始化服务器（公开数据 + 私有数据）
    server = Server(
        cfg=cfg,                                               # 全局配置
        private_datasets=node_datasets["server_private"],      # 服务器私有数据集字典
        public_datasets=node_datasets["server_public"],        # 服务器公开数据集字典
        device=device,                                         # 计算设备
    )
    # 初始化各客户端（私有数据）
    clients = [
        Client(
            client_id=i,                                       # 客户端编号（0-indexed）
            cfg=cfg,                                           # 全局配置
            private_datasets=node_datasets[f"client_{i}"],    # 第 i 个客户端的私有数据集
            device=device,                                     # 计算设备
        )
        for i in range(n_clients)   # 创建 n_clients 个客户端
    ]
    return server, clients   # 返回服务器和客户端列表


def default_task_specs():
    """
    返回默认任务规格字典（SwinMultiTask 多任务预测配置）。

    包含两个任务：
      age：回归任务，输出维度 1（直接预测年龄值）
      sex：分类任务，类别数 2（0=男，1=女）

    返回：
      任务规格字典（{task_name: {type: ..., n_classes: ...}}）
    """
    return {
        "age":           {"type": "regression",     "n_classes": 1},  # 年龄回归（主任务）
        "sex":           {"type": "classification", "n_classes": 2},  # 性别分类（辅助任务）
        "health_status": {"type": "classification", "n_classes": 2},  # 健康状态分类（占位，n_classes 由数据决定）
        "race":          {"type": "classification", "n_classes": 5},  # 种族分类（占位，n_classes 由数据决定）
    }


def main():
    """主函数：解析参数 → 构建数据集 → 顺序运行 5 个方法 → 汇总结果并绘图。"""
    args = parse_args()           # 解析命令行参数

    from utils.log_utils import setup_run_logger   # 终端输出同步保存到 results_dir/run.log
    setup_run_logger(args.results_dir)

    cfg = load_config(args.config)  # 加载配置文件
    set_seed(cfg.training.seed)    # 设置全局随机种子（保证可复现）

    device = cfg.training.device   # 从配置读取计算设备
    # 若 CUDA 不可用但配置要求使用 CUDA，自动降级到 CPU（避免崩溃）
    if not _cuda_available() and device == "cuda":
        device = "cpu"             # 降级到 CPU
        cfg.training.device = "cpu"   # 同步更新配置
        print("[run_all] CUDA 不可用，已切换至 CPU。")

    results_dir = ensure_dir(args.results_dir)   # 创建并返回结果根目录
    task_specs = default_task_specs()             # 获取默认任务规格

    # ------------------------------------------------------------------ #
    # 数据集准备                                                            #
    # ------------------------------------------------------------------ #
    all_ds_ids = _load_openmind_ds_ids(cfg)   # 枚举 OpenMind 子数据集 ID
    # 各 ds_id 的 age/sex/health_status 缺失率统计，用于挑选 cross（跨站点评估）数据集时
    # 二次把关：缺失率过高的 ds_id 会被跳过，避免偏差分析（性别/健康状态×年龄误差热图）
    # 因样本量过少而不可靠；无法计算时（如离线虚拟 ID 调试）返回 None，退化为不做质量过滤
    quality_stats = _load_ds_quality_stats(cfg)
    mri_shape = tuple(cfg.data.mri_shape)      # 从配置读取目标分辨率

    sample_loader_fn = make_openmind_sample_loader(
        cfg.data.hf_dataset_id,             # HuggingFace 数据集仓库 ID
        cache_dir=cfg.data.cache_dir or None,  # 本地缓存目录（None=使用默认）
        max_cache_gb=cfg.data.max_cache_gb,  # 最大缓存大小（GB，LRU 淘汰旧文件）
    )
    # 用实际数据中统计到的类别数覆盖 task_specs 中的占位 n_classes
    for field, n in getattr(sample_loader_fn, "n_classes", {}).items():
        if field in task_specs and n > 0:
            task_specs[field]["n_classes"] = n

    # 构建 M4Fed 专用划分（服务器公开+私有 + 2 个客户端私有）
    # return_split_ids=True：额外拿到 assign_ds_splits 的角色→ds_id 分配结果，
    # 用于事后报告 cross 实际用了哪些 ds_id（见下方 cross_ds_ids.csv 保存）
    node_datasets, split_ids = build_node_datasets(
        all_ds_ids=all_ds_ids,               # 所有可用数据集 ID 列表
        n_cross=cfg.data.n_cross_ds,          # 公开数据集数量（服务器热身用）
        n_private_per_node=cfg.data.n_private_ds_per_node,  # 每个节点的私有数据集数量
        n_clients=cfg.data.n_clients if hasattr(cfg.data, "n_clients") else 2,  # M4Fed 用 2 客户端
        train_split=cfg.data.train_split,     # 训练/测试划分比例
        mri_shape=mri_shape,                  # MRI 目标分辨率
        seed=cfg.data.seed,                   # 划分随机种子
        sample_loader_fn=sample_loader_fn,
        quality_stats=quality_stats,          # ds_id → {字段: 缺失率}，None 表示不过滤
        max_missing_ratio=cfg.data.cross_max_missing_ratio,
        return_split_ids=True,
    )
    # cross 集合仅由 all_ds_ids/n_cross/seed/quality_stats 决定，与 n_clients 无关，
    # 因此 3 客户端划分复用同一批 cross ds_id，无需再次报告
    save_csv({"ds_id": split_ids["cross"]}, os.path.join(results_dir, "cross_ds_ids.csv"))
    print(f"[数据] cross 数据集（{len(split_ids['cross'])} 个 ds_id）已记录至 "
          f"{os.path.join(results_dir, 'cross_ds_ids.csv')}")

    # 构建 3 客户端划分（FedAvg/FedProx/FAN/GPT-FL 共用同一划分）
    node_datasets_3 = build_node_datasets(
        all_ds_ids=all_ds_ids,
        n_cross=cfg.data.n_cross_ds,
        n_private_per_node=cfg.data.n_private_ds_per_node,
        n_clients=3,                          # 3 客户端结构（其他基线使用）
        train_split=cfg.data.train_split,
        mri_shape=mri_shape,
        seed=cfg.data.seed,
        sample_loader_fn=sample_loader_fn,
        quality_stats=quality_stats,
        max_missing_ratio=cfg.data.cross_max_missing_ratio,
    )

    # ------------------------------------------------------------------ #
    # 依次运行各方法                                                        #
    # ------------------------------------------------------------------ #
    all_results = {}   # 收集所有方法的评估结果字典

    # ---- M4Fed（本论文提出的方法）----
    print("\n======== M4Fed ========")
    server, clients = build_participants(cfg, node_datasets, device, n_clients=2)
    set_seed(cfg.training.seed)   # 每个方法前重置随机种子（保证公平对比）
    all_results["m4fed"] = run_m4fed(server, clients, cfg, task_specs, results_dir)
    # 保存服务器 MedMD 路径（供 GPT-FL 复用热身阶段训练好的生成模型）
    m4fed_medmd_path = os.path.join(results_dir, "m4fed", "server_medmd.pt")

    # ---- FedAvg（联邦平均基线，McMahan et al., 2017）----
    print("\n======== FedAvg ========")
    cfg_fa = copy.deepcopy(cfg)          # 深拷贝配置（避免修改影响后续方法）
    cfg_fa.federated.method = "fedavg"   # 标记方法名
    cfg_fa.federated.n_clients = 3       # FedAvg 使用 3 客户端
    server3, clients3 = build_participants(cfg_fa, node_datasets_3, device, n_clients=3)
    set_seed(cfg.training.seed)   # 重置随机种子
    all_results["fedavg"] = run_fedavg(server3, clients3, cfg_fa, task_specs, results_dir)

    # ---- FedProx（联邦近端正则，Li et al., 2020）----
    print("\n======== FedProx ========")
    cfg_fp = copy.deepcopy(cfg)           # 深拷贝配置
    cfg_fp.federated.method = "fedprox"   # 标记方法名
    cfg_fp.federated.n_clients = 3        # FedProx 使用 3 客户端
    server3b, clients3b = build_participants(cfg_fp, node_datasets_3, device, n_clients=3)
    set_seed(cfg.training.seed)   # 重置随机种子
    all_results["fedprox"] = run_fedprox(server3b, clients3b, cfg_fp, task_specs, results_dir)

    # ---- FAN（联邦增强网络，去中心化合成数据交换）----
    print("\n======== FAN ========")
    cfg_fan = copy.deepcopy(cfg)          # 深拷贝配置
    cfg_fan.federated.method = "fan"      # 标记方法名
    cfg_fan.federated.n_clients = 3       # FAN 使用 3 客户端
    server3c, clients3c = build_participants(cfg_fan, node_datasets_3, device, n_clients=3)
    set_seed(cfg.training.seed)   # 重置随机种子
    all_results["fan"] = run_fan(server3c, clients3c, cfg_fan, task_specs, results_dir)

    # ---- GPT-FL（服务器端文本条件生成，He et al., 2023）----
    print("\n======== GPT-FL ========")
    cfg_gfl = copy.deepcopy(cfg)           # 深拷贝配置
    cfg_gfl.federated.method = "gptfl"     # 标记方法名
    cfg_gfl.federated.n_clients = 3        # GPT-FL 使用 3 客户端
    server3d, clients3d = build_participants(cfg_gfl, node_datasets_3, device, n_clients=3)
    set_seed(cfg.training.seed)   # 重置随机种子
    all_results["gptfl"] = run_gptfl(
        server3d, clients3d, cfg_gfl, task_specs, results_dir,
        pretrained_medmd_path=m4fed_medmd_path,   # 复用 M4Fed 热身阶段的服务器 MedMD
    )

    # ------------------------------------------------------------------ #
    # 汇总指标并生成对比图表                                                 #
    # ------------------------------------------------------------------ #
    _save_summary_and_figures(all_results, results_dir, cfg)   # 保存领域指标汇总 CSV 和柱状图
    _save_generative_summary(all_results, results_dir, cfg)    # 保存生成指标汇总 CSV 和柱状图

    # 保存各阶段耗时分布（用于论文效率对比和计算资源分析）
    from utils.timing_utils import save_timing_csv   # 局部导入（避免顶层循环依赖）
    save_timing_csv(os.path.join(results_dir, "timing_breakdown.csv"))
    print("[汇总] 耗时统计已保存至", os.path.join(results_dir, "timing_breakdown.csv"))

    # 保存被判定为不可用而排除的样本清单（重新下载后仍损坏），用于统计真实数据可用率
    from utils.data_quality_utils import save_unusable_csv
    save_unusable_csv(os.path.join(results_dir, "unusable_samples.csv"))
    print("[汇总] 不可用样本清单已保存至", os.path.join(results_dir, "unusable_samples.csv"))


def _save_summary_and_figures(all_results: dict, results_dir: str, cfg) -> None:
    """
    将所有方法的客户端均值指标汇总为 CSV，并生成各指标的柱状图。

    处理逻辑：
      - 从每个方法的 "clients_mean" 和 "clients_std" 提取跨客户端均值和标准差
      - M4Fed 的服务器端指标单独列出（标为 "M4Fed-server"）
      - 为每个指标（MAE/RMSE/Pearson_r/R2）生成带误差棒的柱状图

    参数：
      all_results : {方法名: run_xxx() 的返回字典}（5 个方法的结果）
      results_dir : 结果根目录（汇总文件保存在此）
      cfg         : 配置对象（读取 eval.figure_format 和 eval.palette）
    """
    fmt = cfg.eval.figure_format   # 图表文件格式（"svg"、"pdf"、"png"）
    palette = cfg.eval.palette     # 颜色调色板（来自配置）

    # ---- 构造汇总表格（每个方法占一行）----
    summary_rows = []   # 收集各方法的汇总数据行
    for method, res in all_results.items():
        mean = res.get("clients_mean", {})   # 跨客户端均值
        std = res.get("clients_std", {})      # 跨客户端标准差
        summary_rows.append({
            "method": method,
            "MAE_mean": mean.get("MAE", float("nan")),       # MAE 均值
            "MAE_std": std.get("MAE", float("nan")),          # MAE 标准差
            "RMSE_mean": mean.get("RMSE", float("nan")),     # RMSE 均值
            "RMSE_std": std.get("RMSE", float("nan")),        # RMSE 标准差
            "Pearson_r_mean": mean.get("Pearson_r", float("nan")),    # Pearson_r 均值
            "Pearson_r_std": std.get("Pearson_r", float("nan")),      # Pearson_r 标准差
            "R2_mean": mean.get("R2", float("nan")),          # R² 均值
            "R2_std": std.get("R2", float("nan")),             # R² 标准差
        })

    # 单独列出 M4Fed 服务器端指标（与其他方法的客户端指标并列对比，std=0 因为只有一个服务器）
    if "m4fed" in all_results:
        sr = all_results["m4fed"].get("server", {})   # 服务器评估指标
        summary_rows.append({
            "method": "M4Fed-server",
            "MAE_mean": sr.get("MAE", float("nan")), "MAE_std": 0,         # 服务器 MAE（无方差）
            "RMSE_mean": sr.get("RMSE", float("nan")), "RMSE_std": 0,      # 服务器 RMSE
            "Pearson_r_mean": sr.get("Pearson_r", float("nan")), "Pearson_r_std": 0,
            "R2_mean": sr.get("R2", float("nan")), "R2_std": 0,
        })

    summary_df = pd.DataFrame(summary_rows)   # 转为 DataFrame
    save_csv(summary_df, os.path.join(results_dir, "summary_domain_metrics.csv"))   # 保存 CSV
    print("[汇总] 领域指标汇总表已保存。")

    # ---- 绘制各指标的柱状图（含误差棒）----
    # 内部方法名 → 论文图表展示名称的映射
    bname_map = {
        "fedavg": "FedAvg", "fedprox": "FedProx", "fan": "FAN",
        "gptfl": "GPT-FL", "m4fed": "M4Fed-clients", "M4Fed-server": "M4Fed-server",
    }
    metric_means = {}   # {显示名称: {指标名: 均值}}
    metric_stds = {}    # {显示名称: {指标名: 标准差}}
    for row in summary_rows:
        m = row["method"]                       # 内部方法名
        label = bname_map.get(m, m)             # 转换为显示名称
        metric_means[label] = {
            "MAE": row.get("MAE_mean", float("nan")),
            "RMSE": row.get("RMSE_mean", float("nan")),
            "Pearson_r": row.get("Pearson_r_mean", float("nan")),
            "R2": row.get("R2_mean", float("nan")),
        }
        metric_stds[label] = {
            "MAE": row.get("MAE_std", 0),
            "RMSE": row.get("RMSE_std", 0),
            "Pearson_r": row.get("Pearson_r_std", 0),
            "R2": row.get("R2_std", 0),
        }

    figs_dir = ensure_dir(os.path.join(results_dir, "figures"))   # 创建图表目录
    for metric in ["MAE", "RMSE", "Pearson_r", "R2"]:
        plot_metric_bars(
            metric_means, metric_stds,   # 均值和标准差字典
            metric_name=metric,           # 当前指标名
            palette=palette,              # 颜色调色板
            save_path=os.path.join(figs_dir, f"bar_{metric}"),   # 保存路径
            fmt=fmt,                      # 文件格式
        )
    print("[汇总] 指标柱状图已保存至", figs_dir)


def _save_generative_summary(all_results: dict, results_dir: str, cfg) -> None:
    """
    汇总 FAN / GPT-FL / M4Fed 三个方法的生成模型指标（PSNR/SSIM/LPIPS/FID/KID），
    保存为 CSV 并为每个指标绘制带误差棒的柱状图。

    处理逻辑：
      - 从每个方法结果的 "gen" 字段读取 "clients_mean" / "clients_std"（跨客户端统计）
      - M4Fed / GPT-FL 的服务器生成指标单独列出（标为 "<方法>-server"，std=0，因只有一个服务器）
      - FedAvg / FedProx 无生成模型，自动跳过（结果中无 "gen" 字段）

    参数：
      all_results : {方法名: run_xxx() 的返回字典}
      results_dir : 结果根目录
      cfg         : 配置对象（读取 eval.figure_format 和 eval.palette）
    """
    from evaluation.generative_eval import GEN_METRIC_KEYS   # 生成指标名列表

    fmt = cfg.eval.figure_format   # 图表格式
    palette = cfg.eval.palette     # 调色板

    # 内部方法名 → 图表展示名称
    bname_map = {"fan": "FAN", "gptfl": "GPT-FL", "m4fed": "M4Fed-clients"}

    summary_rows = []   # 汇总表格行
    metric_means = {}   # {显示名称: {指标名: 均值}}（供柱状图）
    metric_stds = {}    # {显示名称: {指标名: 标准差}}

    for method in ["fan", "gptfl", "m4fed"]:
        res = all_results.get(method)
        if not res or "gen" not in res:
            continue   # 该方法无生成模型（FedAvg/FedProx）或未运行，跳过
        gen = res["gen"]
        mean = gen.get("clients_mean", {})   # 跨客户端均值
        std = gen.get("clients_std", {})      # 跨客户端标准差
        row = {"method": method}
        for k in GEN_METRIC_KEYS:
            row[f"{k}_mean"] = mean.get(k, float("nan"))
            row[f"{k}_std"] = std.get(k, float("nan"))
        summary_rows.append(row)

        label = bname_map.get(method, method)
        metric_means[label] = {k: mean.get(k, float("nan")) for k in GEN_METRIC_KEYS}
        metric_stds[label] = {k: std.get(k, 0) for k in GEN_METRIC_KEYS}

        # 服务器生成指标单独成行（M4Fed / GPT-FL 有服务器节点；std=0 因只有一个服务器）
        if "server" in gen:
            sr = gen["server"]
            srow = {"method": f"{method}-server"}
            for k in GEN_METRIC_KEYS:
                srow[f"{k}_mean"] = sr.get(k, float("nan"))
                srow[f"{k}_std"] = 0
            summary_rows.append(srow)
            slabel = f"{bname_map.get(method, method).split('-')[0]}-server"
            metric_means[slabel] = {k: sr.get(k, float("nan")) for k in GEN_METRIC_KEYS}
            metric_stds[slabel] = {k: 0 for k in GEN_METRIC_KEYS}

    if not summary_rows:
        print("[汇总] 无生成模型指标可汇总（FAN/GPT-FL/M4Fed 均未产生结果）。")
        return

    summary_df = pd.DataFrame(summary_rows)   # 转为 DataFrame
    save_csv(summary_df, os.path.join(results_dir, "summary_generative_metrics.csv"))
    print("[汇总] 生成指标汇总表已保存。")

    # 绘制各生成指标的柱状图（含误差棒）
    figs_dir = ensure_dir(os.path.join(results_dir, "figures"))
    for metric in GEN_METRIC_KEYS:
        plot_metric_bars(
            metric_means, metric_stds,
            metric_name=metric,
            palette=palette,
            save_path=os.path.join(figs_dir, f"bar_gen_{metric}"),
            fmt=fmt,
        )
    print("[汇总] 生成指标柱状图已保存至", figs_dir)


def _cuda_available() -> bool:
    """
    检查 CUDA GPU 是否可用（用于自动降级到 CPU）。

    返回：
      True 表示 CUDA 可用，False 表示无 GPU 或 PyTorch 未启用 CUDA
    """
    import torch   # 局部导入（此函数调用频率低）
    return torch.cuda.is_available()   # 检查 CUDA 设备是否可用


def _load_openmind_ds_ids(cfg) -> list:
    """
    获取 OpenMind 有效子数据集 ID 列表（T1w 模态 + age 非空）。

    优先级：
      1. 本地 openneuro_metadata.csv（cfg.data.local_metadata_csv），覆盖全量数据集，无网络开销
      2. HuggingFace 元数据接口（hf_hub_download openneuro_metadata.csv），需网络
      3. 100 个虚拟 ID（离线调试用，不可用于正式实验）

    参数：
      cfg : 配置对象（含 local_metadata_csv、hf_dataset_id、cache_dir 字段）

    返回：
      排序后的子数据集 ID 字符串列表
    """
    import os
    from data.openmind_dataset import list_ds_ids_from_metadata, _load_openmind_metadata
    # 1. 优先读取本地 CSV（全量 ds_id，无需 HuggingFace API 请求）
    local_csv = cfg.data.local_metadata_csv
    if local_csv and os.path.exists(local_csv):
        print(f"[数据] 从本地 {local_csv} 读取子数据集 ID...")
        return list_ds_ids_from_metadata(local_csv)
    # 2. 本地文件不存在，通过 HuggingFace 下载元数据 CSV
    print("[数据] 本地 CSV 不存在，从 HuggingFace 下载元数据...")
    try:
        df = _load_openmind_metadata(
            cfg.data.hf_dataset_id, cfg.data.cache_dir or None, "T1w"
        )
        df = df[df["age"].notna()]
        return sorted(df["ds_id"].unique().tolist())
    except Exception as e:
        # 枚举失败时回退到虚拟 ID（用于调试或网络受限环境）
        print(f"[数据] 无法从 HuggingFace 加载：{e}，使用虚拟 ID 代替。")
        return [f"ds{i:06d}" for i in range(100)]   # 返回 100 个虚拟 ID


def _load_ds_quality_stats(cfg):
    """
    获取各 ds_id 的 age/sex/health_status 缺失率统计，供 cross 数据集质量筛选使用
    （build_node_datasets 的 quality_stats 参数）。

    优先级与 _load_openmind_ds_ids 保持一致，确保统计的 ds_id 集合与枚举出的
    all_ds_ids 口径一致：
      1. 本地 openneuro_metadata.csv → compute_ds_quality_stats
      2. HuggingFace 元数据接口 → 复用 _load_openmind_metadata 返回的表统计
      3. 均不可用（如虚拟 ID 调试模式）→ 返回 None，assign_ds_splits 退化为不做质量过滤

    参数：
      cfg : 配置对象（含 local_metadata_csv、hf_dataset_id、cache_dir 字段）

    返回：
      {ds_id: {field: missing_ratio}} 或 None
    """
    import os
    from data.openmind_dataset import (
        compute_ds_quality_stats, _load_openmind_metadata, _missing_ratio_by_ds_id,
    )
    local_csv = cfg.data.local_metadata_csv
    if local_csv and os.path.exists(local_csv):
        return compute_ds_quality_stats(local_csv)
    try:
        df = _load_openmind_metadata(cfg.data.hf_dataset_id, cfg.data.cache_dir or None, "T1w")
        return _missing_ratio_by_ds_id(df, ("age", "sex", "health_status"))
    except Exception as e:
        print(f"[数据] 无法计算 cross 数据集质量统计：{e}，跳过质量过滤。")
        return None   # assign_ds_splits 在 quality_stats=None 时行为与不传一致


if __name__ == "__main__":
    main()   # 直接运行时调用主函数
