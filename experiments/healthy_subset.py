"""
M4Fed 补充实验：仅使用 OpenMind 中 health_status == "healthy" 的数据训练 M4Fed。

与正式训练（experiments/run_all_baselines.py 中的 M4Fed 分支）的唯一区别：
  - 数据来源被限制为「健康受试者」子集；
  - 数据划分方式改为「按受试者把可用数据平均分成 4 份」（均分以受试者数计，同一
    受试者的多张 T1w 扫描整体绑定、不拆分），而不是按子数据集 ID 分配角色。
  - 「可用数据量」以受试者计（同时在日志中给出对应的 T1w 扫描总数）。

其余一切与 baseline 对齐：
  - 基线的样本筛选逻辑全部保留：
      * 仅 T1w 模态（make_openmind_sample_loader(modality="T1w")）
      * age 字段非空（_sample_loader_fn 内部跳过 age 缺失的行）
      * MRI 体素含 NaN/Inf 的样本淘汰（OpenMindDataset._reject_if_non_finite +
        data/known_nan_samples.csv）
      * 无法下载 / 重新下载后仍损坏的样本跳过（OpenMindDataset._load_mri 标记
        _unusable，本脚本在划分前直接剔除已知不可用样本，训练期间新发现的会写入
        results/<dir>/unusable_samples.csv）
  - 训练/测试划分比例   ：cfg.data.train_split（同 configs/m4fed.yaml）
  - 超参数             ：configs/m4fed.yaml（lr / batch_size / seed / QLoRA 等）
  - 预热轮数 / 联邦循环 / 本地训练轮数：本脚本内覆盖为 baseline 的 1 / 3 / 3
    （仅作用于本补充实验，不影响 configs/m4fed.yaml 及正式训练）
  - 训练完成后的模型性能评估、偏差因素分析、可视化：直接调用 run_m4fed，与正式
    训练完全一致（domain 指标、服务器端偏差热图、生成模型指标、损失曲线等），
    另外补充绘制 M4Fed-clients / M4Fed-server 的指标柱状图。

数据划分（按「受试者」把可用数据平均分成 4 份，均分以受试者数计）：
  第 1 份 → client_0 私有数据
  第 2 份 → client_1 私有数据
  第 3 份 → 服务器私有数据（server_private）
  第 4 份 → 服务器公共数据（server_public，全量作训练集，不划分测试集，同 baseline）
  client_0 / client_1 / server_private 各自再按 cfg.data.train_split 做「受试者级」
  train/test 划分（同一受试者的全部扫描要么都在 train、要么都在 test），
  彻底消除节点内的同受试者信息泄漏。

用法：
  python experiments/healthy_subset.py --config configs/m4fed.yaml \
      --results_dir results/healthy_subset
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import argparse                        # 命令行参数解析
import json                            # 保存队列统计为 JSON
import os                              # 路径操作
import random                          # 固定种子的样本打乱
import sys                             # sys.path 修改

import numpy as np                     # 数值统计（年龄分布等）
import pandas as pd                    # 队列人口学统计表

# 将项目根目录加入 Python 路径（使脚本可直接运行，不依赖 package install）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.env_utils import load_local_env   # 从 .env.local 加载本地密钥（TABPFN_TOKEN 等）
load_local_env()   # 必须在可能触发 tabpfn 等库读取环境变量的导入之前执行

from configs import load_config                              # 加载 YAML 配置
from utils.seed_utils import set_seed                        # 全局随机种子
from utils.io_utils import ensure_dir, save_csv              # 目录创建 / CSV 保存
from utils.log_utils import setup_run_logger                 # 终端输出同步落盘
from data.openmind_dataset import (                          # 数据集构建工具
    OpenMindDataset, make_openmind_sample_loader,
    list_ds_ids_from_metadata, _load_openmind_metadata,
)
from data.splits import train_test_split_indices             # train/test 索引划分
from federated.server import Server                          # 服务器节点
from federated.client import Client                          # 客户端节点
from federated.methods import run_m4fed                      # M4Fed 完整协议（训练+评估+偏差+可视化）
from evaluation.visualization import plot_metric_bars        # 指标柱状图
from evaluation.generative_eval import GEN_METRIC_KEYS       # 生成指标名列表

# 健康状态字段中代表「健康」的取值（大小写/前后空白不敏感）
_HEALTHY_TOKEN = "healthy"


# --------------------------------------------------------------------------- #
#  数据准备                                                                     #
# --------------------------------------------------------------------------- #

def _cuda_available() -> bool:
    """检查 CUDA 是否可用（用于自动降级到 CPU）。"""
    import torch   # 局部导入（调用频率低）
    return torch.cuda.is_available()


def _load_all_ds_ids(cfg) -> list:
    """
    枚举 OpenMind 中所有「T1w + age 非空」的子数据集 ID（与 baseline 的口径一致）。

    优先级：
      1. 本地 openneuro_metadata.csv（cfg.data.local_metadata_csv）——覆盖全量，无网络开销
      2. HuggingFace 元数据接口 ——需网络
    """
    local_csv = cfg.data.local_metadata_csv
    if local_csv and os.path.exists(local_csv):
        print(f"[healthy] 从本地 {local_csv} 枚举子数据集 ID ...")
        return list_ds_ids_from_metadata(local_csv)   # T1w + age 非空
    print("[healthy] 本地 CSV 不存在，从 HuggingFace 下载元数据 ...")
    df = _load_openmind_metadata(cfg.data.hf_dataset_id, cfg.data.cache_dir or None, "T1w")
    df = df[df["age"].notna()]
    return sorted(df["ds_id"].unique().tolist())


def _subject_key(sample: dict) -> str:
    """
    从样本的仓库内相对路径提取「受试者」标识（dsXXXX/sub-YY）。

    一个受试者可能有多张 T1w 扫描（多 session / run），人口学统计按受试者去重。
    """
    meta = sample.get("metadata", {})
    img = None
    if hasattr(meta, "get"):
        img = meta.get("image_path")
    if not img:
        # 退回用 hf_image_path（形如 "OpenMind/ds000001/sub-01/anat/..."）
        img = sample.get("hf_image_path", "")
        parts = img.split("/")
        return "/".join(parts[1:3]) if len(parts) >= 3 else img
    parts = str(img).split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else str(img)


def _is_healthy(sample: dict) -> bool:
    """判断样本的 health_status 是否为 healthy（大小写/空白不敏感）。"""
    meta = sample.get("metadata", {})
    val = meta.get("health_status") if hasattr(meta, "get") else None
    if val is None:
        return False
    try:
        if isinstance(val, float) and np.isnan(val):
            return False
    except TypeError:
        pass
    return str(val).strip().lower() == _HEALTHY_TOKEN


def collect_healthy_samples(cfg, sample_loader_fn, all_ds_ids: list) -> list:
    """
    遍历全部子数据集，收集 health_status == "healthy" 的样本字典。

    sample_loader_fn（make_openmind_sample_loader 的返回闭包）已经完成基线筛选：
      仅 T1w、age 非空、命中 data/known_unusable_samples.csv 的样本标记 _unusable。
    本函数在此基础上：
      1. 只保留 healthy 样本；
      2. 剔除已知不可用样本（_unusable=True）——对应「无法下载 / NaN・Inf」的持久化清单，
         训练期间新发现的不可用样本仍由 OpenMindDataset 在 __getitem__ 中跳过并记录。
    """
    healthy, n_scanned, n_healthy_raw, n_known_unusable = [], 0, 0, 0
    for ds_id in all_ds_ids:
        for s in sample_loader_fn(ds_id):
            n_scanned += 1
            if not _is_healthy(s):
                continue
            n_healthy_raw += 1
            if s.get("_unusable"):
                n_known_unusable += 1
                continue
            healthy.append(s)
    print(f"[healthy] 扫描 T1w+age 非空样本 {n_scanned} 个；其中 healthy {n_healthy_raw} 个；"
          f"剔除已知不可用 {n_known_unusable} 个。")
    return healthy


# --------------------------------------------------------------------------- #
#  队列统计（写入日志 + 落盘）                                                    #
# --------------------------------------------------------------------------- #

def _distribution(series: pd.Series) -> dict:
    """返回某列的取值计数与占比（含缺失值），键为字符串。"""
    counts = series.fillna("<missing>").astype(str).value_counts(dropna=False)
    total = int(counts.sum())
    return {
        str(k): {"count": int(v), "ratio": round(float(v) / total, 4) if total else 0.0}
        for k, v in counts.items()
    }


def log_cohort_stats(samples: list, out_dir: str) -> dict:
    """
    统计并记录健康受试者队列的：
      - 最终可用数据量（以受试者计；同时给出对应的 T1w 扫描总数）
      - 年龄跨度、年龄分布（10 岁一段的直方图 + 描述统计）
      - 性别分布、种族分布

    全部内容 print 到终端（经 setup_run_logger 同步写入 run.log），
    并额外保存：
      healthy_cohort_stats.json      —— 结构化统计
      healthy_cohort_subjects.csv    —— 去重后的受试者人口学明细
    """
    meta_df = pd.DataFrame([dict(s["metadata"]) if hasattr(s["metadata"], "to_dict")
                            else dict(s["metadata"]) for s in samples])
    meta_df["__subject__"] = [_subject_key(s) for s in samples]
    for col in ("age", "sex", "race", "health_status", "unique_id", "image_path"):
        if col not in meta_df.columns:
            meta_df[col] = np.nan
    meta_df["age"] = pd.to_numeric(meta_df["age"], errors="coerce")

    n_scan = len(meta_df)
    subj_df = meta_df.drop_duplicates(subset="__subject__").reset_index(drop=True)
    n_subj = len(subj_df)

    ages = subj_df["age"].dropna().to_numpy(dtype=float)
    if ages.size:
        age_desc = {
            "n_with_age": int(ages.size),
            "min": float(np.min(ages)), "max": float(np.max(ages)),
            "span": float(np.max(ages) - np.min(ages)),
            "mean": float(np.mean(ages)), "std": float(np.std(ages)),
            "median": float(np.median(ages)),
            "q1": float(np.percentile(ages, 25)), "q3": float(np.percentile(ages, 75)),
        }
        lo = int(np.floor(ages.min() / 10.0) * 10)
        hi = int(np.ceil((ages.max() + 1e-6) / 10.0) * 10)
        edges = list(range(lo, hi + 10, 10))
        hist, _ = np.histogram(ages, bins=edges)
        age_hist = {f"{edges[i]}-{edges[i + 1]}": int(hist[i]) for i in range(len(hist))}
    else:
        age_desc, age_hist = {"n_with_age": 0}, {}

    sex_dist = _distribution(subj_df["sex"])
    race_dist = _distribution(subj_df["race"])

    stats = {
        "usable_data_count_subjects": n_subj,  # 「最终可用数据多少」——以受试者计（主口径）
        "total_t1w_scans": n_scan,             # 对应的 T1w 扫描总数（同一受试者可有多张）
        "scans_per_subject_mean": round(n_scan / n_subj, 3) if n_subj else 0.0,
        "age_description_subject_level": age_desc,
        "age_histogram_subject_level_10yr": age_hist,
        "sex_distribution_subject_level": sex_dist,
        "race_distribution_subject_level": race_dist,
        "sex_distribution_scan_level": _distribution(meta_df["sex"]),
        "race_distribution_scan_level": _distribution(meta_df["race"]),
    }

    # ---- 打印到日志 ----
    print("\n" + "=" * 70)
    print("[healthy] 健康受试者队列统计")
    print("=" * 70)
    print(f"[healthy] 最终可用数据量（以受试者计）: {n_subj}")
    print(f"[healthy] 对应 T1w 扫描总数: {n_scan}  "
          f"（人均 {n_scan / n_subj:.2f} 张）" if n_subj else "")
    if age_desc.get("n_with_age"):
        print(f"[healthy] 年龄跨度: {age_desc['min']:.1f} ~ {age_desc['max']:.1f} 岁 "
              f"（跨度 {age_desc['span']:.1f} 岁）")
        print(f"[healthy] 年龄描述统计: 均值 {age_desc['mean']:.2f} ± {age_desc['std']:.2f}，"
              f"中位数 {age_desc['median']:.1f}，Q1 {age_desc['q1']:.1f}，Q3 {age_desc['q3']:.1f}，"
              f"有年龄标签 {age_desc['n_with_age']}/{n_subj}")
        print("[healthy] 年龄分布（每 10 岁一段，受试者级）:")
        for k, v in age_hist.items():
            frac = v / max(1, age_desc["n_with_age"])
            print(f"          {k:>9} 岁 : {v:5d}  ({frac:5.1%})  " + "#" * int(round(frac * 40)))
    else:
        print("[healthy] 队列中无有效年龄标签。")
    print("[healthy] 性别分布（受试者级）:")
    for k, v in sex_dist.items():
        print(f"          {k:>10} : {v['count']:5d}  ({v['ratio']:.1%})")
    print("[healthy] 种族分布（受试者级）:")
    for k, v in race_dist.items():
        print(f"          {k:>24} : {v['count']:5d}  ({v['ratio']:.1%})")
    print("=" * 70 + "\n")

    # ---- 落盘 ----
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "healthy_cohort_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    save_csv(
        subj_df[["__subject__", "unique_id", "age", "sex", "race", "health_status"]]
        .rename(columns={"__subject__": "subject"}),
        os.path.join(out_dir, "healthy_cohort_subjects.csv"),
    )
    print(f"[healthy] 队列统计已保存至 {out_dir}/healthy_cohort_stats.json 与 "
          f"healthy_cohort_subjects.csv")
    return stats


# --------------------------------------------------------------------------- #
#  按受试者 4 等分划分 + 节点数据集构建                                            #
# --------------------------------------------------------------------------- #

def group_by_subject(samples: list) -> list:
    """
    将扫描级样本按受试者（dsXXXX/sub-YY）分组。

    返回：[(subject_key, [该受试者的全部扫描样本]), ...]，顺序按受试者首次出现顺序。
    分组后，同一受试者的所有扫描在后续划分中始终整体移动——不跨客户端/节点，
    也不跨 train/test。
    """
    groups: dict = {}
    for s in samples:
        groups.setdefault(_subject_key(s), []).append(s)
    return list(groups.items())


def even_4way_split_subjects(subject_groups: list, seed: int) -> list:
    """
    把受试者随机打乱后平均分成 4 组（余数依次分给前面的组，受试者数最多相差 1）。

    「平均」以受试者数计（同一受试者的多张扫描随之整体移动），因此四份的扫描数可能
    略有差异。

    返回：[part0, part1, part2, part3]，每个 part 为 [(subject, [samples]), ...]。
    """
    if len(subject_groups) < 4:
        raise ValueError(f"健康受试者仅 {len(subject_groups)} 名，无法平均分成 4 份")
    idx = list(range(len(subject_groups)))
    random.Random(seed).shuffle(idx)
    shuffled = [subject_groups[i] for i in idx]
    n = len(shuffled)
    base, rem = divmod(n, 4)
    sizes = [base + (1 if i < rem else 0) for i in range(4)]
    parts, cur = [], 0
    for sz in sizes:
        parts.append(shuffled[cur:cur + sz])
        cur += sz
    return parts


def _flatten_subjects(subj_entries: list) -> list:
    """把 [(subject, [samples]), ...] 展平成扫描级样本列表。"""
    out = []
    for _, samps in subj_entries:
        out.extend(samps)
    return out


def build_node_datasets_from_parts(parts: list, cfg, mri_shape,
                                    register_to_template: bool = False) -> dict:
    """
    将「受试者级 4 等分」结果组装为 run_m4fed 需要的节点数据集字典。

      parts[0] → client_0 私有   （再按 train_split 做「受试者级」train/test 划分）
      parts[1] → client_1 私有   （同上）
      parts[2] → server_private  （同上）
      parts[3] → server_public   （全量 train，无 test，同 baseline）

    train/test 也按受试者划分：先对受试者列表做 train_test_split_indices，再把各自
    受试者的全部扫描展平——同一受试者要么全在 train、要么全在 test。

    register_to_template : 转发给 OpenMindDataset（见 data/registration.py）。默认
      False，与本函数改动前行为完全一致；experiments/healthy_subset_registered.py
      传 True，在这之外不改变任何划分/训练逻辑。
    """
    ts, seed = cfg.data.train_split, cfg.data.seed

    def _split(subj_entries: list) -> dict:
        tr_idx, te_idx = train_test_split_indices(len(subj_entries), ts, seed)
        tr = _flatten_subjects([subj_entries[i] for i in tr_idx])
        te = _flatten_subjects([subj_entries[i] for i in te_idx])
        return {
            "train": OpenMindDataset(tr, mri_shape, register_to_template=register_to_template),
            "test":  OpenMindDataset(te, mri_shape, register_to_template=register_to_template),
        }

    return {
        "client_0":       _split(parts[0]),
        "client_1":       _split(parts[1]),
        "server_private": _split(parts[2]),
        "server_public":  {"train": OpenMindDataset(_flatten_subjects(parts[3]), mri_shape,
                                                      register_to_template=register_to_template)},
    }


def _save_split_assignment(parts: list, out_dir: str) -> None:
    """记录每名受试者（及其每张扫描）被分到哪个角色，便于复现与核对。"""
    role_names = ["client_0_private", "client_1_private", "server_private", "server_public"]
    rows = []
    for role, part in zip(role_names, parts):
        for subject, samps in part:
            for s in samps:
                meta = s.get("metadata", {})
                rows.append({
                    "role": role,
                    "subject": subject,
                    "unique_id": meta.get("unique_id") if hasattr(meta, "get") else None,
                    "age": meta.get("age") if hasattr(meta, "get") else None,
                    "sex": meta.get("sex") if hasattr(meta, "get") else None,
                    "race": meta.get("race") if hasattr(meta, "get") else None,
                })
    save_csv(pd.DataFrame(rows), os.path.join(out_dir, "healthy_split_assignment.csv"))
    detail = ", ".join(
        f"{r}={len(p)} 名/{sum(len(x[1]) for x in p)} 张"
        for r, p in zip(role_names, parts)
    )
    print(f"[healthy] 4 等分（按受试者，受试者数/扫描数）: {detail}")
    print(f"[healthy] 划分明细已保存至 {out_dir}/healthy_split_assignment.csv")


# --------------------------------------------------------------------------- #
#  评估结果可视化补充（与 run_all_baselines._save_summary_and_figures 对齐）        #
# --------------------------------------------------------------------------- #

def _save_metric_bars(results: dict, cfg, out_dir: str) -> None:
    """为 M4Fed-clients（跨客户端均值±std）与 M4Fed-server 绘制 domain / 生成指标柱状图。"""
    figs = ensure_dir(os.path.join(out_dir, "figures"))
    fmt, palette = cfg.eval.figure_format, cfg.eval.palette

    # ---- domain 指标 ----
    dmean = {
        "M4Fed-clients": results.get("clients_mean", {}),
        "M4Fed-server":  results.get("server", {}),
    }
    dstd = {
        "M4Fed-clients": results.get("clients_std", {}),
        "M4Fed-server":  {k: 0.0 for k in results.get("server", {})},
    }
    rows = [{"node": "M4Fed-clients", **results.get("clients_mean", {})},
            {"node": "M4Fed-clients-std", **results.get("clients_std", {})},
            {"node": "M4Fed-server", **results.get("server", {})}]
    for c in results:
        if c.startswith("client_") and isinstance(results[c], dict) and "MAE" in results[c]:
            rows.append({"node": c, **results[c]})
    save_csv(pd.DataFrame(rows), os.path.join(out_dir, "summary_domain_metrics.csv"))
    for metric in ["MAE", "RMSE", "Pearson_r", "R2"]:
        plot_metric_bars(dmean, dstd, metric_name=metric, palette=palette,
                         save_path=os.path.join(figs, f"bar_{metric}"), fmt=fmt)

    # ---- 生成指标 ----
    gen = results.get("gen", {})
    if gen:
        gmean = {"M4Fed-clients": gen.get("clients_mean", {}),
                 "M4Fed-server":  gen.get("server", {})}
        gstd = {"M4Fed-clients": gen.get("clients_std", {}),
                "M4Fed-server":  {k: 0.0 for k in gen.get("server", {})}}
        grows = [{"node": "M4Fed-clients", **gen.get("clients_mean", {})},
                 {"node": "M4Fed-clients-std", **gen.get("clients_std", {})},
                 {"node": "M4Fed-server", **gen.get("server", {})}]
        save_csv(pd.DataFrame(grows), os.path.join(out_dir, "summary_generative_metrics.csv"))
        for metric in GEN_METRIC_KEYS:
            plot_metric_bars(gmean, gstd, metric_name=metric, palette=palette,
                             save_path=os.path.join(figs, f"bar_gen_{metric}"), fmt=fmt)
    print(f"[healthy] 指标柱状图与汇总 CSV 已保存至 {out_dir}")


# --------------------------------------------------------------------------- #
#  主流程                                                                       #
# --------------------------------------------------------------------------- #

def run_healthy_subset(args, cfg, rounds_override=(1, 3, 3),
                        register_to_template: bool = False,
                        noise_override=None,
                        age_zscore: bool = False) -> dict:
    """
    健康受试者子集上的 M4Fed 补充实验主流程。

    rounds_override : (warm_rounds, federated_rounds, local_rounds)，仅覆盖内存中的
      cfg，不修改 configs/m4fed.yaml（不影响正式训练）。默认 (1, 3, 3)，即
      baseline 轮数；传 None 则不覆盖，直接使用 cfg 中的值。
    register_to_template : 转发给 build_node_datasets_from_parts /
      OpenMindDataset（见 data/registration.py），默认 False。
    noise_override : (noise_strength_fed, noise_strength_local_min,
      noise_strength_local_max)，仅覆盖内存中的 cfg.model，不修改
      configs/m4fed.yaml。默认 None 则不覆盖，直接使用 cfg 中的值（baseline：
      联邦循环阶段固定 0.75，本地训练阶段 np.random.uniform(0.35, 0.70)）；
      experiments/healthy_subset_low_noise.py 传 (0.05, 0.05, 0.05)，即联邦循环
      与本地训练阶段的合成噪声强度都降至 0.05。
    age_zscore : 转发给 run_m4fed，对领域模型的年龄训练标签做 z-score 标准化（各节点用
      自身真实训练集的均值/标准差，评估时反变换回原始年龄）。默认 False；
      experiments/healthy_subset_registered_age_zscore.py 传 True。
    """
    out_dir = ensure_dir(args.results_dir)

    if rounds_override is not None:
        cfg.federated.warm_rounds, cfg.federated.federated_rounds, cfg.federated.local_rounds = \
            rounds_override

    if noise_override is not None:
        (cfg.model.noise_strength_fed, cfg.model.noise_strength_local_min,
         cfg.model.noise_strength_local_max) = noise_override

    mri_shape = tuple(cfg.data.mri_shape)

    device = cfg.training.device
    if not _cuda_available() and device == "cuda":
        device = cfg.training.device = "cpu"
        print("[healthy] CUDA 不可用，已切换至 CPU。")

    set_seed(cfg.training.seed)   # 固定全局随机种子（同 baseline）

    # ---- 1. 收集健康受试者样本（保留全部基线筛选逻辑） ----
    sample_loader_fn = make_openmind_sample_loader(
        cfg.data.hf_dataset_id,
        cache_dir=cfg.data.cache_dir or None,
        max_cache_gb=cfg.data.max_cache_gb,
    )
    all_ds_ids = _load_all_ds_ids(cfg)
    healthy_samples = collect_healthy_samples(cfg, sample_loader_fn, all_ds_ids)
    subject_groups = group_by_subject(healthy_samples)
    if len(subject_groups) < 4:
        raise RuntimeError(f"可用健康受试者过少（{len(subject_groups)}），无法开展实验")
    print(f"[healthy] 健康受试者 {len(subject_groups)} 名，对应 T1w 扫描 {len(healthy_samples)} 张")

    # ---- 2. 记录队列统计（可用数据量以受试者计 / 年龄跨度 / 年龄・性别・种族分布） ----
    log_cohort_stats(healthy_samples, out_dir)

    # ---- 3. 按受试者平均划分为 4 份并组装节点数据集
    #        （同一受试者的全部扫描不跨份、不跨 train/test） ----
    parts = even_4way_split_subjects(subject_groups, seed=cfg.data.seed)
    _save_split_assignment(parts, out_dir)
    node_datasets = build_node_datasets_from_parts(parts, cfg, mri_shape,
                                                    register_to_template=register_to_template)

    # ---- 4. 构建参与者（服务器 + 2 客户端），超参 / 轮数全部取自 cfg（同 baseline） ----
    server = Server(cfg, node_datasets["server_private"],
                    node_datasets["server_public"], device)
    clients = [Client(i, cfg, node_datasets[f"client_{i}"], device) for i in range(2)]

    task_specs = {
        "age":           {"type": "regression",     "n_classes": 1},
        "sex":           {"type": "classification", "n_classes": 2},
        "health_status": {"type": "classification", "n_classes": 2},  # 占位，下方按数据覆盖
        "race":          {"type": "classification", "n_classes": 5},  # 占位，下方按数据覆盖
    }
    for field, n in getattr(sample_loader_fn, "n_classes", {}).items():
        if field in task_specs and n > 0:
            task_specs[field]["n_classes"] = n

    print(f"[healthy] 预热轮数={cfg.federated.warm_rounds}  "
          f"联邦循环轮数={cfg.federated.federated_rounds}  "
          f"本地训练轮数={cfg.federated.local_rounds}  "
          f"train_split={cfg.data.train_split}  seed={cfg.training.seed}")
    print(f"[healthy] 合成噪声强度：联邦循环阶段(noise_strength_fed)="
          f"{cfg.model.noise_strength_fed}  "
          f"本地训练阶段采样区间(noise_strength_local_min~max)="
          f"[{cfg.model.noise_strength_local_min}, {cfg.model.noise_strength_local_max}]")

    print(f"[healthy] 年龄标签 z-score 标准化：{'开启' if age_zscore else '关闭'}")

    # ---- 5. 运行完整 M4Fed 协议：训练 + 性能评估 + 偏差因素分析 + 可视化（同正式训练） ----
    set_seed(cfg.training.seed)
    results = run_m4fed(server, clients, cfg, task_specs, out_dir, age_zscore=age_zscore)

    # ---- 6. 补充：M4Fed-clients / M4Fed-server 指标柱状图与汇总表 ----
    _save_metric_bars(results, cfg, out_dir)

    # ---- 7. 与正式训练一致地保存耗时分布与不可用样本清单 ----
    from utils.timing_utils import save_timing_csv
    from utils.data_quality_utils import save_unusable_csv
    save_timing_csv(os.path.join(out_dir, "timing_breakdown.csv"))
    save_unusable_csv(os.path.join(out_dir, "unusable_samples.csv"))
    nan_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "data", "known_nan_samples.csv")
    if os.path.exists(nan_csv):
        try:
            n_nan = max(0, sum(1 for _ in open(nan_csv, encoding="utf-8")) - 1)
            print(f"[healthy] 累计记录的 NaN/Inf 不可用样本数（data/known_nan_samples.csv）: {n_nan}")
        except OSError:
            pass

    sm = results.get("clients_mean", {})
    sv = results.get("server", {})
    print(f"[healthy] 完成。客户端均值 MAE={sm.get('MAE', float('nan')):.3f} "
          f"RMSE={sm.get('RMSE', float('nan')):.3f} r={sm.get('Pearson_r', float('nan')):.3f} "
          f"R2={sm.get('R2', float('nan')):.3f} | 服务器 MAE={sv.get('MAE', float('nan')):.3f}")
    print(f"[healthy] 全部结果已保存至 {out_dir}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="M4Fed 补充实验：仅健康受试者子集")
    p.add_argument("--config", default="configs/m4fed.yaml", help="YAML 配置文件路径")
    p.add_argument("--results_dir", default="results/healthy_subset", help="结果保存目录")
    return p.parse_args()


def main():
    args = parse_args()
    log_path = setup_run_logger(args.results_dir)
    print(f"[healthy] 终端日志将同步保存至 {log_path}")
    cfg = load_config(args.config)
    run_healthy_subset(args, cfg)


if __name__ == "__main__":
    main()
