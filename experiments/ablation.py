"""
M4Fed 消融实验（Ablation Study）。

验证三个组件的有效性（通过逐一移除）：
  变体 A：w/o server public data    —— 服务器公开数据集置空，验证公开数据的作用
  变体 B：w/o TabPFN                —— 禁用表格补全（缺失值保持 NaN），验证 TabPFN 的作用
  变体 C：random-init SwinUNETR    —— SwinUNETR 骨干随机初始化（不使用预训练权重），
                                      验证预训练骨干对脑年龄预测的贡献

每个变体都在相同的数据集划分和超参数下运行完整的 M4Fed 协议，
唯一的区别是对应组件被移除或替换（控制变量法）。
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import os, sys, copy                  # os：路径；sys：路径修改；copy：深拷贝
import numpy as np                    # NumPy（统计计算）
import pandas as pd                   # Pandas（汇总表格）

# 将项目根目录加入 Python 路径（使实验脚本可以直接运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import load_config                   # 加载 YAML 配置文件
from utils.seed_utils import set_seed             # 随机种子设置
from utils.io_utils import ensure_dir, save_csv   # 目录创建和 CSV 保存
from data.openmind_dataset import build_node_datasets, make_openmind_sample_loader  # 数据集构建
from federated.server import Server               # 服务器节点
from federated.client import Client               # 客户端节点
from federated.methods import run_m4fed           # M4Fed 完整协议
from evaluation.visualization import plot_metric_bars   # 指标柱状图


def _build(cfg, node_datasets, device, n_clients=2):
    """
    快速构建服务器和客户端参与者对象（消融实验的辅助函数）。

    参数：
      cfg          : 配置对象（已为当前变体深拷贝并修改）
      node_datasets: 节点数据集字典（由 build_node_datasets 返回）
      device       : 计算设备（"cuda" 或 "cpu"）
      n_clients    : 客户端数量（消融实验固定为 2）

    返回：
      (server, clients) 元组
    """
    # 初始化服务器（含公开数据和私有数据）
    server = Server(cfg, node_datasets["server_private"],
                    node_datasets["server_public"], device)
    # 初始化各客户端（只含私有数据）
    clients = [Client(i, cfg, node_datasets[f"client_{i}"], device)
               for i in range(n_clients)]
    return server, clients   # 返回服务器和客户端列表


def run_ablation(args, cfg, all_ds_ids, mri_shape, task_specs, results_dir):
    """
    运行全部三个消融变体并生成对比结果。

    每个变体的运行流程：
      1. 深拷贝基础配置（避免变体间相互影响）
      2. 深拷贝基础数据集划分（保证相同的数据分布）
      3. 修改对应组件（公开数据/TabPFN/SwinUNETR 初始化）
      4. 运行完整 M4Fed 协议
      5. 记录结果

    参数：
      args       : 命令行参数
      cfg        : 基础 M4FedConfig（所有变体的起点）
      all_ds_ids : 所有子数据集 ID 列表
      mri_shape  : MRI 体积目标尺寸（来自配置）
      task_specs : 任务规格字典（预测任务的配置）
      results_dir: 结果保存根目录

    返回：
      all_ablation_results 字典（键为变体名，值为对应 run_m4fed 结果字典）
    """
    device = cfg.training.device                                      # 计算设备
    ablation_out = ensure_dir(os.path.join(results_dir, "ablation"))  # 创建消融实验结果目录

    sample_loader_fn = make_openmind_sample_loader(
        cfg.data.hf_dataset_id,                  # HuggingFace 数据集仓库 ID
        cache_dir=cfg.data.cache_dir or None,    # 本地缓存目录
        max_cache_gb=cfg.data.max_cache_gb,      # 最大缓存大小（GB）
    )
    # 用实际类别数更新 task_specs（覆盖 main.py 中的占位 n_classes）
    for field, n in getattr(sample_loader_fn, "n_classes", {}).items():
        if field in task_specs and n > 0:
            task_specs[field]["n_classes"] = n

    # 构建基础节点数据集（所有消融变体共用相同的数据集划分，保证公平对比）
    node_datasets = build_node_datasets(
        all_ds_ids=all_ds_ids,                               # 所有可用数据集 ID
        n_cross=cfg.data.n_cross_ds,                          # 公开数据集数量
        n_private_per_node=cfg.data.n_private_ds_per_node,   # 每节点私有数据集数量
        n_clients=2,                                          # 消融实验固定 2 客户端
        train_split=cfg.data.train_split,                     # 训练/测试比例
        mri_shape=mri_shape,                                  # MRI 目标分辨率
        seed=cfg.data.seed,                                   # 数据划分随机种子
        sample_loader_fn=sample_loader_fn,                    # 样本加载函数
    )

    ablation_results = {}   # 收集各变体的实验结果

    # ------------------------------------------------------------------
    # 变体 A：移除服务器公开数据（验证公开数据对 MedMD 热身的重要性）
    # 实现方式：将 server_public 数据集的 samples 列表清空（置为空列表）
    # 预期效果：MedMD 无法热身，U-Net 性能下降
    # ------------------------------------------------------------------
    print("[消融] 变体 A：无服务器公开数据")
    set_seed(cfg.training.seed)         # 重置随机种子（保证可复现）
    cfg_a = copy.deepcopy(cfg)          # 深拷贝配置（变体 A 独立）
    nd_a = copy.deepcopy(node_datasets)  # 深拷贝数据集（变体 A 独立）
    nd_a["server_public"]["train"].samples = []   # 清空公开训练集（模拟无公开数据）
    s, c = _build(cfg_a, nd_a, device)   # 构建参与者（含清空公开数据的服务器）
    ablation_results["w/o_public"] = run_m4fed(
        s, c, cfg_a, task_specs, os.path.join(ablation_out, "wo_public"))   # 运行并记录

    # ------------------------------------------------------------------
    # 变体 B：移除 TabPFN 表格补全（验证元数据填充对文本条件质量的重要性）
    # 实现方式：Monkey-patch TabPFNImputer.fit_transform 为恒等函数
    # 效果：元数据中所有 NaN 值保持不变，文本条件包含 NaN 占位符，质量下降
    # ------------------------------------------------------------------
    print("[消融] 变体 B：无 TabPFN 表格补全")
    set_seed(cfg.training.seed)         # 重置随机种子
    cfg_b = copy.deepcopy(cfg)          # 深拷贝配置（变体 B 独立）
    s, c = _build(cfg_b, node_datasets, device)   # 构建参与者（使用原始数据集）
    # Monkey-patch：替换 fit_transform 为恒等函数（lambda 直接返回 df，不做任何填充）
    from models.tabpfn_wrapper import TabPFNImputer
    TabPFNImputer.fit_transform = lambda self, df: df   # 覆盖方法（不做 NaN 填充）
    ablation_results["w/o_tabpfn"] = run_m4fed(
        s, c, cfg_b, task_specs, os.path.join(ablation_out, "wo_tabpfn"))

    # ------------------------------------------------------------------
    # 变体 C：SwinUNETR 随机初始化（验证预训练骨干权重对领域预测的重要性）
    # 实现方式：Monkey-patch Server.build_models，强制 random_init_swin=True
    # 效果：SwinUNETR 骨干从随机权重出发（而非 MONAI 预训练权重），作为消融对照
    # ------------------------------------------------------------------
    print("[消融] 变体 C：SwinUNETR 随机初始化（无预训练骨干）")
    set_seed(cfg.training.seed)          # 重置随机种子
    cfg_c = copy.deepcopy(cfg)           # 深拷贝配置（变体 C 独立）
    s, c = _build(cfg_c, node_datasets, device)   # 构建参与者（使用原始数据集）
    original_build = Server.build_models  # 保存原始 build_models 方法（实验后恢复）
    def _random_build(self, ts, **kw):
        """强制 random_init_swin=True 的 build_models 替代版本（消融专用）。"""
        original_build(self, ts, random_init_swin=True)
    Server.build_models = _random_build
    ablation_results["random_init_swin"] = run_m4fed(
        s, c, cfg_c, task_specs, os.path.join(ablation_out, "random_init"))
    Server.build_models = original_build   # 恢复原始方法（避免影响后续实验）

    # ------------------------------------------------------------------
    # 汇总并保存结果（CSV + 柱状图）
    # ------------------------------------------------------------------
    rows = []   # 收集各变体的汇总数据行
    for variant, res in ablation_results.items():
        mean = res.get("clients_mean", {})   # 跨客户端均值
        std = res.get("clients_std", {})      # 跨客户端标准差
        rows.append({"variant": variant,
                     "MAE": mean.get("MAE", float("nan")),         # 主指标
                     "RMSE": mean.get("RMSE", float("nan")),       # 辅助指标
                     "Pearson_r": mean.get("Pearson_r", float("nan")),
                     "R2": mean.get("R2", float("nan")),
                     "MAE_std": std.get("MAE", 0),                  # MAE 标准差
                     })
    summary_df = pd.DataFrame(rows)   # 转为 DataFrame
    save_csv(summary_df, os.path.join(ablation_out, "ablation_summary.csv"))   # 保存 CSV
    print("[消融] 结果已保存至", ablation_out)

    # 绘制各变体 MAE 对比柱状图（直观展示各组件的贡献）
    fmt = cfg.eval.figure_format   # 图表文件格式
    metric_means = {r["variant"]: {k: r[k] for k in ["MAE","RMSE","Pearson_r","R2"]}
                    for r in rows}   # 构造 {变体名: {指标: 值}} 的字典
    plot_metric_bars(metric_means, metric_name="MAE",
                     title="消融实验：MAE 对比",
                     save_path=os.path.join(ablation_out, "ablation_MAE"), fmt=fmt)
    return ablation_results   # 返回完整消融实验结果字典


if __name__ == "__main__":
    import argparse   # 命令行参数解析
    from data.openmind_dataset import list_openmind_ds_ids
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m4fed.yaml")   # 配置文件路径
    p.add_argument("--results_dir", default="results")          # 结果保存目录
    args = p.parse_args()   # 解析参数
    cfg = load_config(args.config)   # 加载配置
    all_ds_ids = list_openmind_ds_ids(cfg.data.hf_dataset_id)
    mri_shape = tuple(cfg.data.mri_shape)
    task_specs = {
        "age":           {"type": "regression",     "n_classes": 1},
        "health_status": {"type": "classification", "n_classes": 2},
        "race":          {"type": "classification", "n_classes": 5},
    }
    run_ablation(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)
