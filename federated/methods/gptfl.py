"""
GPT-FL 基线（He et al., 10.1109/CVPRW67362.2025.00164）。

训练配置：
  - 1 个 server + 3 个 clients，每个 client 有 20 个子数据集的 private data
  - 生成模型（generative model）：M4Fed 热身阶段在 server 上预训练好的 MedMD
  - 领域模型（domain model）：SwinMultiTask，参数随机初始化

训练过程：
  热身阶段（Warm-up）：
    1. 每个 client 将本地 private data 的 age 标签值发送给 server
    2. server 合并 age 标签值并删除重复项
    3. server 使用 MedMD 仅以 "age is <age_val>" 为 condition（不使用 img）
       推理生成 synthetic 3D MRI，数量等于所有 clients 的 private data 数量之和
    4. server 用 synthetic data 预训练 SwinMultiTask（domain model）
    5. server 将预训练好的 SwinMultiTask 分发给每个 client

  联邦循环阶段（Federated Loop）：
    - 训练轮数 = federated_rounds
    - 训练方式同 FedAvg（服务器广播全局 SwinMultiTask → 客户端在 private data 上本地训练 → FedAvg 聚合）
"""
from __future__ import annotations
import copy, os
import numpy as np
import pandas as pd
from typing import List

from training.swin_trainer import SwinTrainer
from evaluation.metrics import compute_domain_metrics
from evaluation.generative_eval import (                 # 生成模型评估与汇总工具
    evaluate_generative_model, aggregate_metrics, save_summary, save_loss_summary,
)
from data.openmind_dataset import OpenMindDataset, make_dataloader
from utils.io_utils import save_csv, save_model, ensure_dir, load_model
from utils.timing_utils import record_timing
import time


def run_gptfl(
    server,                              # Server 实例（持有预训练 MedMD，负责热身阶段合成与预训练）
    clients: List,                       # Client 实例列表
    cfg,                                 # 全局 M4FedConfig 配置对象
    task_specs: dict,                    # 预测任务规格字典
    results_dir: str,                    # 结果保存根目录
    pretrained_medmd_path: str = None,   # M4Fed 热身阶段保存的服务器 MedMD 权重路径（可选）
) -> dict:
    """
    运行 GPT-FL 联邦训练协议。

    参数：
      server               : Server 实例（持有 MedMD，热身阶段生成 synthetic data 并预训练 SwinMultiTask）
      clients              : Client 实例列表
      cfg                  : M4FedConfig 全局配置
      task_specs           : 预测任务规格字典
      results_dir          : 结果保存根目录
      pretrained_medmd_path: M4Fed 热身阶段保存的服务器 MedMD 权重路径。
                             若存在，加载该权重以获得更好的生成质量；
                             若不存在，服务器 MedMD 随机初始化。

    返回：
      {
        'client_0': 指标字典,
        'client_1': 指标字典,
        'clients_mean': 均值,
        'clients_std': 标准差,
      }
    """
    device = cfg.training.device
    out = ensure_dir(os.path.join(results_dir, "gptfl"))

    # 所有客户端随机初始化 SwinMultiTask（GPT-FL domain model 随机初始化，后由 server 分发预训练权重）
    for c in clients:
        c.build_models(task_specs, random_init_swin=True)

    # 服务器：SwinMultiTask 随机初始化，MedMD 加载 M4Fed 预训练权重（若可用）
    server.build_models(task_specs, random_init_swin=True)
    if pretrained_medmd_path and os.path.exists(pretrained_medmd_path):
        load_model(server.medmd, pretrained_medmd_path, strict=False)
        print(f"[GPT-FL] 成功加载预训练 MedMD：{pretrained_medmd_path}")
    else:
        print("[GPT-FL] 未找到预训练 MedMD，使用随机初始化 MedMD。")

    _train_t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 热身阶段（Warm-up）                                                    #
    # ------------------------------------------------------------------ #
    print("[GPT-FL] === 热身阶段 ===")

    # Step 1：从每个 client 收集本地 private data 的 age 标签值，server 合并并去重
    all_ages = set()
    for c in clients:
        for batch in c.train_loader:
            ages = batch["age"].numpy()
            all_ages.update(int(a) for a in ages if not np.isnan(a))
    all_ages = sorted(all_ages)
    print(f"[GPT-FL] 收集到 {len(all_ages)} 个唯一 age 标签值")

    # Step 2：server 以纯文本条件（"age is <age_val>"，不使用 img）生成 synthetic data
    # 合成数量 = 所有 clients 的 private data 数量之和
    n_total = max(1, sum(c.n_private_train() for c in clients))
    print(f"[GPT-FL] 目标合成数量：{n_total}（所有 clients private data 之和）")

    # 循环遍历 age 标签值，均匀覆盖各年龄段
    age_cycle = (all_ages * (n_total // max(len(all_ages), 1) + 1))[:n_total]
    server_syn = []
    for idx, age_val in enumerate(age_cycle):
        batch_mri = server.medmd.synthesise_from_label(
            age_label=age_val,
            batch_size=1,
            mri_shape=tuple(cfg.data.mri_shape),
            ddim_steps=cfg.model.ddim_steps,
        )
        server_syn.append({
            "mri_array": batch_mri[0].cpu().numpy(),
            "table_text": f"age is {age_val}",
            "age": float(age_val),
            "metadata": {},
        })
        if (idx + 1) % max(1, n_total // 10) == 0:
            print(f"[GPT-FL]   已生成 {idx+1}/{n_total} 个 synthetic 样本")

    # Step 3：server 在 synthetic data 上预训练 SwinMultiTask（domain model）
    syn_ds = OpenMindDataset(server_syn, tuple(cfg.data.mri_shape))
    syn_loader = make_dataloader(syn_ds, cfg.training.batch_size,
                                 shuffle=True, seed=cfg.training.seed)
    server_dom_trainer = SwinTrainer(
        server.swin, device=device,
        lr=cfg.training.lr_swin,
        mixed_precision=cfg.training.mixed_precision,
        use_qlora=False,
    )
    warmup_losses = []
    print(f"[GPT-FL] 在 synthetic data 上预训练 SwinMultiTask（{cfg.federated.warm_rounds} epochs）")
    for ep in range(cfg.federated.warm_rounds):
        loss = server_dom_trainer.train_epoch(syn_loader)
        warmup_losses.append({"phase": "warmup", "round": ep, "loss": loss})
        print(f"[GPT-FL]   warmup epoch {ep+1}/{cfg.federated.warm_rounds}，loss={loss:.4f}")

    # Step 4：server 将预训练好的 SwinMultiTask 分发给每个 client
    global_state = copy.deepcopy(server.swin.state_dict())
    for c in clients:
        c.swin.load_state_dict(copy.deepcopy(global_state))
    print("[GPT-FL] SwinMultiTask 预训练权重已分发给所有 clients")

    # ------------------------------------------------------------------ #
    # 联邦循环阶段（Federated Loop，训练方式同 FedAvg）                        #
    # ------------------------------------------------------------------ #
    print("[GPT-FL] === 联邦循环阶段（FedAvg） ===")
    n_rounds = cfg.federated.federated_rounds
    all_losses = {f"client_{c.client_id}": [] for c in clients}

    for rnd in range(n_rounds):
        print(f"[GPT-FL] 第 {rnd+1}/{n_rounds} 轮")

        local_states = []
        local_ns = []

        for c in clients:
            c.swin.load_state_dict(copy.deepcopy(global_state))

            trainer = SwinTrainer(
                c.swin, device=device,
                lr=cfg.training.lr_swin,
                mixed_precision=cfg.training.mixed_precision,
                use_qlora=False,
            )
            losses = trainer.train(c.train_loader, n_epochs=1)
            all_losses[f"client_{c.client_id}"].append(
                {"phase": "federated", "round": rnd, "loss": losses[0]}
            )
            local_states.append(copy.deepcopy(c.swin.state_dict()))
            local_ns.append(c.n_train_samples())

        # FedAvg 加权平均聚合
        global_state = _fedavg_aggregate(local_states, local_ns)

    record_timing("gptfl/train", time.perf_counter() - _train_t0)

    # ---- 领域模型评估阶段（脑年龄预测：MAE / RMSE / Pearson r / R²） ----
    _eval_t0 = time.perf_counter()
    results = {}
    for c in clients:
        c.swin.load_state_dict(copy.deepcopy(global_state))
        t = SwinTrainer(c.swin, device=device, lr=0, mixed_precision=False)
        y_true, y_pred = t.evaluate(c.test_loader)
        results[f"client_{c.client_id}"] = compute_domain_metrics(y_true, y_pred)
    record_timing("gptfl/eval", time.perf_counter() - _eval_t0)

    # 汇总跨客户端领域指标的均值和标准差
    client_keys = [f"client_{c.client_id}" for c in clients]
    agg = _agg({k: results[k] for k in client_keys})
    results["clients_mean"] = agg["mean"]
    results["clients_std"] = agg["std"]

    # ---- 生成模型评估阶段（PSNR / SSIM / LPIPS / FID / KID） ----
    # GPT-FL 采用集中式生成：生成模型仅存在于 server（clients 消费其合成数据，
    # 自身不训练生成模型）。因此以 server.medmd 作为唯一生成模型，分别对
    # server 与各 client 的真实测试数据评估，得到"分 server 和 client"的生成指标。
    _gen_t0 = time.perf_counter()
    gen_results = {}
    gen_results["server"] = evaluate_generative_model(
        server.medmd, server.test_loader, cfg, device)   # server 生成模型 vs server 数据
    for c in clients:
        gen_results[f"client_{c.client_id}"] = evaluate_generative_model(
            server.medmd, c.test_loader, cfg, device)     # 共享生成模型 vs client 数据
    gen_agg = aggregate_metrics({k: gen_results[k] for k in client_keys})
    gen_results["clients_mean"] = gen_agg["mean"]
    gen_results["clients_std"] = gen_agg["std"]
    results["gen"] = gen_results
    record_timing("gptfl/gen_eval", time.perf_counter() - _gen_t0)

    # ---- 保存结果到磁盘 ----
    save_csv(pd.DataFrame(warmup_losses),
             os.path.join(out, "server_warmup_losses.csv"))
    save_csv(pd.DataFrame([gen_results["server"]]),
             os.path.join(out, "server_gen_metrics.csv"))   # server 生成指标
    for c in clients:
        key = f"client_{c.client_id}"
        save_csv(pd.DataFrame(all_losses[key]),
                 os.path.join(out, f"{key}_losses.csv"))
        save_csv(pd.DataFrame([results[key]]),
                 os.path.join(out, f"{key}_metrics.csv"))
        save_csv(pd.DataFrame([gen_results[key]]),
                 os.path.join(out, f"{key}_gen_metrics.csv"))   # client 生成指标
        if cfg.eval.save_model:
            save_model(c.swin, os.path.join(out, f"{key}_swin.pt"))   # 领域模型权重

    if cfg.eval.save_model:
        save_model(server.medmd, os.path.join(out, "server_medmd.pt"))  # 生成模型权重（server）

    # 跨客户端汇总 CSV：领域指标、生成指标、逐轮损失的 mean/std
    save_summary(results["clients_mean"], results["clients_std"],
                 os.path.join(out, "clients_domain_summary.csv"))
    save_summary(gen_results["clients_mean"], gen_results["clients_std"],
                 os.path.join(out, "clients_gen_summary.csv"))
    save_loss_summary({k: all_losses[k] for k in client_keys},
                      os.path.join(out, "clients_loss_summary.csv"))

    return results


def _fedavg_aggregate(states: list, ns: list) -> dict:
    """FedAvg 加权平均聚合：w_global = Σ_k (n_k / N) · w_k"""
    total = sum(ns)
    agg = copy.deepcopy(states[0])
    for key in agg:
        agg[key] = sum(s[key].float() * (n / total) for s, n in zip(states, ns))
    return agg


def _agg(d):
    """计算指标字典的均值和标准差。"""
    from collections import defaultdict
    vals = defaultdict(list)
    for v in d.values():
        for k, val in v.items():
            vals[k].append(val)
    return {
        "mean": {k: float(np.nanmean(v)) for k, v in vals.items()},
        "std":  {k: float(np.nanstd(v))  for k, v in vals.items()},
    }
