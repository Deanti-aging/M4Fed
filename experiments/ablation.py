"""
M4Fed 消融实验（Ablation Study）。

验证四个组件/数据条件的有效性：
  变体 A：w/o_public         —— 服务器公开数据集置空，验证公开数据的作用
  变体 B：w/o_tabpfn         —— 禁用表格补全（缺失值保持 NaN），验证 TabPFN 的作用
  变体 C：w/o_pretrained_swin —— SwinUNETR 骨干随机初始化（不使用预训练权重），
                                 验证预训练骨干对脑年龄预测的贡献
  变体 D：训练数据健康状态构成 —— 训练数据筛选为 health_status 为"健康"的受试者（去掉
                                 "有病"样本），验证训练数据健康状态分布对领域模型泛化的
                                 影响。评估统一采用 M4Fed 服务器端领域模型（与 main 一致），
                                 在（未按健康状态筛选的）服务器测试集上按 health_status
                                 拆分出三个测试切片：mixed_test（完整测试集，即"有病和
                                 健康混合"）、ill_test（仅有病）、healthy_test（仅健康），
                                 并给出两种训练数据构成的对照：
                                   healthy_train_* —— 训练集仅健康人（本变体训练所得服务器模型）
                                   mixed_train_*   —— M4Fed 完整训练（训练集含病人），权重直接
                                                      取自 results_dir/m4fed/server_swin.pt，
                                                      无需重训；文件缺失时自动跳过该对照

每个变体都在相同的数据集划分和超参数下运行完整的 M4Fed 协议，
唯一的区别是对应组件被移除/替换，或训练数据被筛选（控制变量法）。
变体 D 的 mixed_train_* 对照复用 main 已训练好的服务器模型，与其余变体共享同一套
数据集划分（cfg.data.seed），故测试样本一致、可直接比较。
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import os, sys, copy                  # os：路径；sys：路径修改；copy：深拷贝
import numpy as np                    # NumPy（统计计算）
import pandas as pd                   # Pandas（汇总表格）
import torch                          # 加载 M4Fed 服务器模型权重（变体 D mixed_train_* 对照）

# 将项目根目录加入 Python 路径（使实验脚本可以直接运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import load_config                   # 加载 YAML 配置文件
from utils.seed_utils import set_seed             # 随机种子设置
from utils.io_utils import ensure_dir, save_csv   # 目录创建、CSV 保存
from data.openmind_dataset import (               # 数据集构建
    build_node_datasets, make_openmind_sample_loader, OpenMindDataset, make_dataloader,
)
from federated.server import Server               # 服务器节点
from federated.client import Client               # 客户端节点
from federated.methods import run_m4fed           # M4Fed 完整协议
from models import SwinMultiTask                  # 加载 main 已训练的服务器领域模型（变体 D 对照）
from training.swin_trainer import SwinTrainer     # 领域模型（脑年龄预测）评估器
from evaluation.metrics import compute_domain_metrics    # 领域指标：MAE/RMSE/Pearson_r/R2
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


def _filter_samples_by_health(samples: list, allowed_labels: set) -> list:
    """
    按 health_status（TabPFN 填充前的原始标签编码，sample["health_status"]）
    筛选样本列表，只保留取值在 allowed_labels 中的样本。

    参数：
      samples        : 样本字典列表（OpenMindDataset.samples）
      allowed_labels : 允许保留的 health_status 编码值集合（float）；
                       空集合表示无法确定标签映射，返回空列表（保守处理，不误用未筛选数据）
      （用于变体 D：w/o_ill_train，训练集只保留 health_status=healthy 的样本）

    返回：
      筛选后的样本字典列表（新列表，不修改原列表）
    """
    if not allowed_labels:
        return []
    return [s for s in samples if s.get("health_status") in allowed_labels]


def _eval_domain_on_samples(swin_model, samples: list, mri_shape, cfg, device) -> dict:
    """
    在给定的样本列表（而非某节点的完整测试集）上评估领域模型（脑年龄预测）性能。

    用于消融变体 D：把节点原始测试集按 health_status 拆分为子集后，
    分别构建临时 DataLoader 并评估，不影响节点自身的 test_loader。

    参数：
      swin_model : 已训练的 SwinMultiTask 模型（推理模式，不更新参数）
      samples    : 样本字典列表（某个 health_status 子集）
      mri_shape  : MRI 目标分辨率（与训练时一致）
      cfg        : 全局配置对象（读取 batch_size 等）
      device     : 计算设备

    返回：
      {'MAE', 'RMSE', 'Pearson_r', 'R2'} 字典；样本为空时全部为 NaN
    """
    nan_metrics = {"MAE": float("nan"), "RMSE": float("nan"),
                   "Pearson_r": float("nan"), "R2": float("nan")}
    if not samples:
        return nan_metrics
    loader = make_dataloader(
        OpenMindDataset(samples, mri_shape),
        batch_size=cfg.training.batch_size,
        shuffle=False,                          # 评估不打乱顺序
        num_workers=cfg.training.num_workers,
        seed=cfg.training.seed,
    )
    trainer = SwinTrainer(swin_model, device=device, lr=0, mixed_precision=False)
    y_true, y_pred = trainer.evaluate(loader)
    if len(y_true) == 0:
        return nan_metrics
    return compute_domain_metrics(y_true, y_pred)


#: 变体 D 的三个测试切片名（顺序即汇总表 / 柱状图的展示顺序）
HEALTH_SLICE_ORDER = ["mixed_test", "ill_test", "healthy_test"]


def _eval_server_health_slices(swin_model, test_samples: list, mri_shape, cfg, device,
                               healthy_label, ill_label) -> dict:
    """
    在服务器（未按健康状态筛选的）测试集上，按 health_status 把样本拆成
    mixed / ill_only / healthy_only 三个切片，分别用同一个 SwinMultiTask
    评估脑年龄预测性能（与 main 一致，只看服务器端领域模型）。

    参数：
      swin_model    : 已训练的 SwinMultiTask（推理模式，不更新参数）
      test_samples  : 服务器测试样本字典列表（未经健康状态筛选）
      mri_shape     : MRI 目标分辨率
      cfg           : 全局配置对象
      device        : 计算设备
      healthy_label : health_status 中"健康"对应的编码值（None 表示映射未知）
      ill_label     : health_status 中"有病"对应的编码值（None 表示映射未知）

    返回：
      {"mixed_test": {...}, "ill_test": {...}, "healthy_test": {...}}，
      每个值为 {'MAE','RMSE','Pearson_r','R2'} 字典；切片为空时全部为 NaN
    """
    slice_samples = {
        "mixed_test":   test_samples,
        "ill_test":     [s for s in test_samples
                         if ill_label is not None and s.get("health_status") == ill_label],
        "healthy_test": [s for s in test_samples
                         if healthy_label is not None and s.get("health_status") == healthy_label],
    }
    return {name: _eval_domain_on_samples(swin_model, samp, mri_shape, cfg, device)
            for name, samp in slice_samples.items()}


def _load_m4fed_server_swin(candidate_paths: list, cfg, task_specs, device):
    """
    加载 main 流程已训练好的 M4Fed 服务器端 SwinMultiTask 权重
    （<...>/m4fed/server_swin.pt），用于变体 D 的 mixed_train_* 对照，
    避免为"训练集含病人"这一条件重复训练一次完整 M4Fed。

    参数：
      candidate_paths : 依次尝试的 server_swin.pt 候选路径列表
                        （通常为 [<m4fed_results_dir>/m4fed/server_swin.pt,
                                  <results_dir>/m4fed/server_swin.pt]）

    返回：
      SwinMultiTask 实例（已加载权重、已 .to(device)）；
      若所有候选路径都不存在则返回 None（调用方跳过 mixed_train_* 对照）。
    """
    model_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
    if model_path is None:
        print(f"[消融] 变体 D：以下路径均无 server_swin.pt，跳过 mixed_train_*"
              f"（M4Fed 完整训练）对照：{candidate_paths}")
        return None
    model = SwinMultiTask(
        pretrain_url=cfg.model.swinunetr_pretrain_url,
        img_size=tuple(cfg.data.mri_shape),
        feature_size=cfg.model.swinunetr_feature_size,
        task_specs=task_specs,
        device=device,
        random_init=True,                          # 骨干权重随即被下方加载覆盖，无需下载 MONAI 权重
        age_loss_weight=cfg.training.age_loss_weight,
    ).to(device)
    # 只加载形状匹配的权重（骨干 + age 回归头必然匹配；若 main 与本次的
    # race/sex 等分类头 n_classes 不同，那几个 head 权重会被跳过，不影响脑龄预测）
    state = torch.load(model_path, map_location="cpu")
    model_sd = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
    dropped = [k for k in state if k not in compatible]
    model.load_state_dict(compatible, strict=False)
    if dropped:
        print(f"[消融] 变体 D：server_swin.pt 有 {len(dropped)} 个键因形状/命名不匹配被跳过"
              f"（不影响脑龄回归头与骨干）：{dropped[:6]}{' ...' if len(dropped) > 6 else ''}")
    print(f"[消融] 变体 D：已加载 M4Fed 服务器领域模型 {model_path}（mixed_train_* 对照）")
    return model


def run_ablation(args, cfg, all_ds_ids, mri_shape, task_specs, results_dir):
    """
    运行全部四个消融变体并生成对比结果。

    每个变体的运行流程：
      1. 深拷贝基础配置（避免变体间相互影响）
      2. 深拷贝基础数据集划分（保证相同的数据分布）
      3. 修改对应组件（公开数据/TabPFN/SwinUNETR 初始化），或筛选训练数据（变体 D）
      4. 运行完整 M4Fed 协议
      5. 记录结果（A/B/C 走 clients_mean 口径；变体 D 走服务器端领域模型口径，
         在服务器测试集上按 health_status 拆成 mixed/ill/healthy 三个切片，并对比
         healthy_train_*（仅健康人训练）与 mixed_train_*（复用 main 的 M4Fed 完整训练
         权重 results_dir/m4fed/server_swin.pt，缺失则跳过）两种训练数据构成）

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
        n_public_max=cfg.data.n_public_ds_max,                # server_public 数据集数量上限
    )

    # health_status 的字符串 → 整数编码映射（由 make_openmind_sample_loader 在全量元数据上拟合，
    # 跨子数据集一致，如 {"healthy": 0, "ill": 1}），供变体 D 筛选训练数据 / 拆分测试子集使用
    health_label_map = getattr(sample_loader_fn, "label_maps", {}).get("health_status", {})
    healthy_label = health_label_map.get("healthy")   # "健康" 对应的编码值（可能为 None）
    ill_label = health_label_map.get("ill")           # "有病" 对应的编码值（可能为 None）

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
    original_fit_transform = TabPFNImputer.fit_transform   # 保存原始方法（实验后恢复）
    TabPFNImputer.fit_transform = lambda self, df: df   # 覆盖方法（不做 NaN 填充）
    ablation_results["w/o_tabpfn"] = run_m4fed(
        s, c, cfg_b, task_specs, os.path.join(ablation_out, "wo_tabpfn"))
    TabPFNImputer.fit_transform = original_fit_transform   # 恢复原始方法（避免影响后续变体 C）

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
    ablation_results["w/o_pretrained_swin"] = run_m4fed(
        s, c, cfg_c, task_specs, os.path.join(ablation_out, "wo_pretrained_swin"))
    Server.build_models = original_build   # 恢复原始方法（避免影响后续实验）

    # ------------------------------------------------------------------
    # 变体 D：训练数据健康状态构成（验证训练数据健康状态分布对领域模型的影响）
    # 实现方式：深拷贝数据集后，把服务器私有/公开训练集和各客户端私有训练集的
    # samples 筛选为 health_status == "healthy"（编码值 healthy_label）的子集；
    # 测试集保持原始（未筛选）状态，评估阶段用服务器端领域模型按 health_status
    # 拆成 mixed/ill/healthy 三个切片，并与 M4Fed 完整训练（含病人）的服务器模型对照。
    # ------------------------------------------------------------------
    print("[消融] 变体 D：训练数据仅用健康人（health_status=healthy）")
    set_seed(cfg.training.seed)          # 重置随机种子
    cfg_d = copy.deepcopy(cfg)           # 深拷贝配置（变体 D 独立）
    nd_d = copy.deepcopy(node_datasets)  # 深拷贝数据集（变体 D 独立，测试集稍后原样使用）
    healthy_allowed = {healthy_label} if healthy_label is not None else set()
    nd_d["server_private"]["train"].samples = _filter_samples_by_health(
        nd_d["server_private"]["train"].samples, healthy_allowed)   # 服务器私有训练集只留健康人
    nd_d["server_public"]["train"].samples = _filter_samples_by_health(
        nd_d["server_public"]["train"].samples, healthy_allowed)    # 服务器公开训练集只留健康人
    for i in range(2):
        nd_d[f"client_{i}"]["train"].samples = _filter_samples_by_health(
            nd_d[f"client_{i}"]["train"].samples, healthy_allowed)  # 客户端私有训练集只留健康人
    s, c = _build(cfg_d, nd_d, device)   # 构建参与者（训练集已筛选，测试集仍为原始未筛选数据）
    ablation_results["w/o_ill_train"] = run_m4fed(
        s, c, cfg_d, task_specs, os.path.join(ablation_out, "wo_ill_train"))

    # 额外评估（统一走 M4Fed 服务器端领域模型，与 main 口径一致）：
    # 在（未按健康状态筛选的）服务器测试集上，按 health_status 拆成
    # mixed / ill_only / healthy_only 三个切片，比较两种训练数据构成——
    #   healthy_train_* : 本变体所得服务器模型（训练集仅健康人）
    #   mixed_train_*   : M4Fed 完整训练所得服务器模型（训练集含病人，权重取自
    #                     <m4fed_results_dir>/m4fed/server_swin.pt，无需重训；缺失则跳过）
    server_test_samples = s.private_datasets["test"].samples   # 未筛选，两种模型共用同一批样本
    healthy_train_slices = _eval_server_health_slices(
        s.swin, server_test_samples, mri_shape, cfg_d, device, healthy_label, ill_label)
    # M4Fed 完整训练权重优先取自 --m4fed_results_dir（默认 results/main），
    # 回退到本次消融的 results_dir（同一次 run_all 里 ablation 与 m4fed 同目录时）
    m4fed_ref_dir = getattr(args, "m4fed_results_dir", None) or "results/main"
    swin_candidates = [
        os.path.join(m4fed_ref_dir, "m4fed", "server_swin.pt"),
        os.path.join(results_dir, "m4fed", "server_swin.pt"),
    ]
    m4fed_full_swin = _load_m4fed_server_swin(swin_candidates, cfg_d, task_specs, device)
    mixed_train_slices = (
        _eval_server_health_slices(m4fed_full_swin, server_test_samples, mri_shape,
                                   cfg_d, device, healthy_label, ill_label)
        if m4fed_full_swin is not None else None)

    ablation_results["w/o_ill_train"]["health_subset_eval"] = {
        "healthy_train": healthy_train_slices,   # {mixed_test/ill_test/healthy_test: 指标}
        "mixed_train":   mixed_train_slices,     # 同上；server_swin.pt 缺失时为 None
    }

    # 汇总为长表：每行一个 (训练数据构成 × 测试切片) 组合，供 CSV 和柱状图使用
    health_slice_rows = []
    for cond, slices in (("healthy_train", healthy_train_slices),
                         ("mixed_train", mixed_train_slices)):
        if slices is None:
            continue
        for sl in HEALTH_SLICE_ORDER:
            m = slices[sl]
            health_slice_rows.append({
                "variant": f"{cond}_{sl}",
                "MAE": m.get("MAE", float("nan")),
                "RMSE": m.get("RMSE", float("nan")),
                "Pearson_r": m.get("Pearson_r", float("nan")),
                "R2": m.get("R2", float("nan")),
                "MAE_std": 0,   # 单模型单次评估，无标准差（与主表列对齐用 0 占位）
            })
    save_csv(pd.DataFrame(health_slice_rows),
             os.path.join(ablation_out, "ill_train_health_slice_summary.csv"))
    print("[消融] 变体 D：健康状态测试切片评估结果已保存至", ablation_out)

    # ------------------------------------------------------------------
    # 汇总并保存结果（CSV + 柱状图）
    # ------------------------------------------------------------------
    rows = []   # 收集各变体的汇总数据行
    for variant, res in ablation_results.items():
        # 变体 D（w/o_ill_train）不走 clients_mean 口径：它按 health_status 测试切片、
        # 且统一用服务器端领域模型展开为 healthy_train_* / mixed_train_* 多行（见下方并入）
        if variant == "w/o_ill_train":
            continue
        mean = res.get("clients_mean", {})   # 跨客户端均值
        std = res.get("clients_std", {})      # 跨客户端标准差
        rows.append({"variant": variant,
                     "MAE": mean.get("MAE", float("nan")),         # 主指标
                     "RMSE": mean.get("RMSE", float("nan")),       # 辅助指标
                     "Pearson_r": mean.get("Pearson_r", float("nan")),
                     "R2": mean.get("R2", float("nan")),
                     "MAE_std": std.get("MAE", 0),                  # MAE 标准差
                     })
    # 并入变体 D 的健康状态测试切片行（服务器端领域模型口径）：
    #   healthy_train_{mixed,ill,healthy}_test —— 训练集仅健康人
    #   mixed_train_{mixed,ill,healthy}_test   —— M4Fed 完整训练（server_swin.pt；缺失则无此 3 行）
    # 其中 mixed_train_mixed_test 即 M4Fed 服务器模型在完整测试集上的脑龄预测表现，
    # 可与 run_all_baselines 的 M4Fed 行交叉核对（口径：前者服务器、后者跨客户端均值）
    rows.extend(health_slice_rows)
    summary_df = pd.DataFrame(rows)   # 转为 DataFrame
    save_csv(summary_df, os.path.join(ablation_out, "ablation_summary.csv"))   # 保存 CSV
    print("[消融] 结果已保存至", ablation_out)

    # 绘制各变体 MAE 对比柱状图（直观展示各组件的贡献）
    fmt = cfg.eval.figure_format   # 图表文件格式
    metric_means = {r["variant"]: {k: r[k] for k in ["MAE","RMSE","Pearson_r","R2"]}
                    for r in rows}   # 构造 {变体名: {指标: 值}} 的字典
    plot_metric_bars(metric_means, metric_name="MAE",
                     save_path=os.path.join(ablation_out, "ablation_MAE"), fmt=fmt)

    # 单独绘制变体 D 的健康状态测试切片 MAE 对比图
    # （healthy_train / mixed_train × mixed_test / ill_test / healthy_test）
    if health_slice_rows:
        slice_metric_means = {r["variant"]: {k: r[k] for k in ["MAE","RMSE","Pearson_r","R2"]}
                              for r in health_slice_rows}
        plot_metric_bars(slice_metric_means, metric_name="MAE",
                         save_path=os.path.join(ablation_out, "ill_train_health_slice_MAE"),
                         fmt=fmt)
    return ablation_results   # 返回完整消融实验结果字典


if __name__ == "__main__":
    import argparse   # 命令行参数解析
    from data.openmind_dataset import list_openmind_ds_ids
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m4fed.yaml")   # 配置文件路径
    p.add_argument("--results_dir", default="results")          # 结果保存目录
    p.add_argument("--m4fed_results_dir", default="results/main",
                   help="变体 D mixed_train_* 对照读取的 M4Fed 完整训练结果目录"
                        "（需含 m4fed/server_swin.pt）")
    args = p.parse_args()   # 解析参数
    from utils.log_utils import setup_run_logger   # 终端输出同步保存到 results_dir/run.log
    setup_run_logger(args.results_dir)
    cfg = load_config(args.config)   # 加载配置
    all_ds_ids = list_openmind_ds_ids(cfg.data.hf_dataset_id)
    mri_shape = tuple(cfg.data.mri_shape)
    task_specs = {
        "age":           {"type": "regression",     "n_classes": 1},
        "sex":           {"type": "classification", "n_classes": 2},
        "health_status": {"type": "classification", "n_classes": 2},
        "race":          {"type": "classification", "n_classes": 5},
    }
    run_ablation(args, cfg, all_ds_ids, mri_shape, task_specs, args.results_dir)
