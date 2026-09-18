"""
M4Fed 补充实验预处理脚本：把健康受试者子集（与 experiments/healthy_subset.py
完全一致的样本筛选口径：仅 T1w、age 非空、health_status == "healthy"、剔除
已知不可用样本）的全部 T1w 扫描仿射配准到 MNI152 标准模板，并落盘缓存。

为什么单独一个脚本、不放在训练脚本里边训边配准：
  - 配准（SimpleITK 仿射，见 data/registration.py）是纯 CPU 任务，和训练用的
    A100 GPU 完全解耦，单独跑可以：
      * 不让 GPU 卡等一个和它无关的 CPU 密集步骤；
      * 用尽可能多的 CPU 核并行（FARM 分区共 32 核，本脚本默认用 --workers）；
      * 支持断点续跑（已缓存的样本直接跳过）、单个样本失败不拖垮整批。
  - experiments/healthy_subset_registered.py 训练时直接读配准结果磁盘缓存
    （data.registration.DEFAULT_REGISTERED_CACHE_DIR），跑本脚本和跑训练
    脚本互不依赖同一次 SLURM 作业，可以分开提交、分开重跑。

用法：
  python experiments/register_healthy_subset_mri.py --config configs/m4fed.yaml \
      --workers 16
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import argparse                        # 命令行参数解析
import os                              # 路径操作
import sys                             # sys.path 修改
import time                            # 进度/耗时统计
from multiprocessing import Pool       # CPU 并行配准

import numpy as np                     # 保存 .npy 缓存
import pandas as pd                    # 失败清单落盘

# 将项目根目录加入 Python 路径（使脚本可直接运行，不依赖 package install）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.env_utils import load_local_env   # 从 .env.local 加载本地密钥（HF_TOKEN 等）
load_local_env()   # 必须在触发 huggingface_hub 下载之前执行

from configs import load_config                              # 加载 YAML 配置
from utils.io_utils import ensure_dir, save_csv              # 目录创建 / CSV 保存
from utils.log_utils import setup_run_logger                 # 终端输出同步落盘
from data.openmind_dataset import OpenMindDataset, make_openmind_sample_loader
from data.registration import (                              # 配准与缓存路径工具
    register_to_template, registered_cache_path, get_template_path,
    DEFAULT_REGISTERED_CACHE_DIR,
)
# 复用 healthy_subset.py 里的样本筛选逻辑，保证本脚本配准的样本集合与训练时
# 完全一致（同一份 all_ds_ids 枚举 + 同一套 healthy/可用性过滤）
from experiments.healthy_subset import _load_all_ds_ids, collect_healthy_samples


def _register_one(sample: dict) -> tuple:
    """
    子进程工作函数：（按需下载 +）配准 + 写缓存单个样本。

    在 Pool 子进程内执行，只接收/返回可 pickle 的简单数据（样本字典本身
    已经是 make_openmind_sample_loader 产出的普通 dict，可直接跨进程传递）。

    返回 (identifier, status, detail)：
      status ∈ {"cached", "done", "failed"}
        - cached : 配准缓存已存在，本次跳过（用于统计，不重复配准）
        - done   : 本次成功配准并写入缓存
        - failed : 下载或配准失败，detail 为错误信息（不会写入跨运行共享的
                   data/known_unusable_samples.csv——配准失败是本预处理特有
                   的问题，不代表该样本对其它未配准的实验也不可用；训练脚本
                   会话内会重试一次，仍失败则在本次运行内标记为不可用）
    """
    identifier = sample.get("hf_image_path")
    cache_path = registered_cache_path(identifier)
    if os.path.exists(cache_path):
        return identifier, "cached", ""
    try:
        mri_path = OpenMindDataset._resolve_hf_path(sample)   # 惰性下载（命中缓存则跳过）
        vol = register_to_template(mri_path)
    except Exception as e:   # noqa: BLE001 —— 批处理脚本：任何单样本失败都不应中断整批
        return identifier, "failed", f"{type(e).__name__}: {e}"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp{os.getpid()}"
    with open(tmp_path, "wb") as f:   # 用文件对象写入，避免 np.save 自动追加 .npy 后缀
        np.save(f, vol)
    os.replace(tmp_path, cache_path)   # 原子重命名，避免并发/中断留下半截缓存文件
    return identifier, "done", ""


def parse_args():
    p = argparse.ArgumentParser(description="预先把健康受试者子集的 MRI 配准到 MNI152 标准模板")
    p.add_argument("--config", default="configs/m4fed.yaml", help="YAML 配置文件路径")
    p.add_argument("--results_dir", default="results/register_healthy_subset_mri",
                    help="日志/失败清单保存目录（配准结果本身缓存在 data/mri_registered_cache）")
    p.add_argument("--workers", type=int, default=8,
                    help="并行配准的进程数（纯 CPU 任务，可设到 SLURM 分配的 cpus-per-task）")
    return p.parse_args()


def main():
    args = parse_args()
    log_path = setup_run_logger(args.results_dir)
    print(f"[register] 终端日志将同步保存至 {log_path}")
    cfg = load_config(args.config)

    print("[register] 检查/下载 MNI152 标准模板 ...")
    template_path = get_template_path()
    print(f"[register] 模板路径: {template_path}")
    print(f"[register] 配准结果缓存目录: {DEFAULT_REGISTERED_CACHE_DIR}")

    # ---- 枚举与 healthy_subset.py 完全一致的健康受试者样本集合 ----
    sample_loader_fn = make_openmind_sample_loader(
        cfg.data.hf_dataset_id,
        cache_dir=cfg.data.cache_dir or None,
        max_cache_gb=cfg.data.max_cache_gb,
    )
    all_ds_ids = _load_all_ds_ids(cfg)
    healthy_samples = collect_healthy_samples(cfg, sample_loader_fn, all_ds_ids)
    n_total = len(healthy_samples)
    print(f"[register] 待配准样本总数: {n_total}（与 healthy_subset.py 口径一致）")
    print(f"[register] 使用 {args.workers} 个进程并行配准 ...")

    out_dir = ensure_dir(args.results_dir)
    t0 = time.time()
    n_cached = n_done = n_failed = 0
    failures = []
    with Pool(processes=args.workers) as pool:
        for i, (identifier, status, detail) in enumerate(
            pool.imap_unordered(_register_one, healthy_samples, chunksize=4), start=1
        ):
            if status == "cached":
                n_cached += 1
            elif status == "done":
                n_done += 1
            else:
                n_failed += 1
                failures.append({"identifier": identifier, "reason": detail})
            if i % 50 == 0 or i == n_total:
                elapsed = time.time() - t0
                print(f"[register] 进度 {i}/{n_total}  "
                      f"已缓存(跳过)={n_cached} 新完成={n_done} 失败={n_failed}  "
                      f"已耗时 {elapsed / 60:.1f} 分钟")

    if failures:
        save_csv(pd.DataFrame(failures), os.path.join(out_dir, "registration_failures.csv"))
        print(f"[register] {len(failures)} 个样本配准失败，明细见 "
              f"{out_dir}/registration_failures.csv")

    print(f"[register] 完成。已缓存(跳过)={n_cached}  新完成={n_done}  失败={n_failed}  "
          f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"[register] 可以运行 experiments/healthy_subset_registered.py 开始训练")


if __name__ == "__main__":
    main()
