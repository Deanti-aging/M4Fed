# M4Fed Project Framework

## 1. Project Overview

M4Fed is a federated learning framework for **brain age prediction** (regression) that couples:
- A 3D Latent Diffusion generative model **MedMD** (img2img, text-conditioned) for synthetic MRI synthesis
- A domain foundation model **NeuroFM** fine-tuned via multi-task learning
- A novel federated protocol with warm-up, generative data exchange, and QLoRA-based client adaptation

---

## 2. Project Directory Structure

```
M4Fed/
├── PROJECT_FRAMEWORK.md          # This file
├── requirements.txt              # Conda/pip dependencies
├── main.py                       # Unified CLI entry point
├── configs/
│   ├── __init__.py
│   ├── base_config.py            # Dataclass-based config
│   ├── m4fed.yaml
│   ├── fedavg.yaml
│   ├── fedprox.yaml
│   ├── fan.yaml
│   └── gptfl.yaml
├── data/
│   ├── __init__.py
│   ├── openmind_dataset.py       # HuggingFace streaming loader & splitter
│   ├── preprocessing.py          # MRI resize / normalize; table normalise
│   ├── splits.py                 # Assign ds-IDs to server/clients/cross
│   └── table_to_text.py          # Table row → natural-language string
├── models/
│   ├── __init__.py
│   ├── medmd/
│   │   ├── __init__.py
│   │   ├── medmd.py              # MedMD pipeline (VAE + UNet + BioMedCLIP)
│   │   ├── vae_wrapper.py        # Frozen MONAI AutoencoderKL loader
│   │   └── conditioning.py       # BioMedCLIP text encoder wrapper
│   ├── neurofm/
│   │   ├── __init__.py
│   │   └── neurofm_multitask.py  # NeuroFM + regression task heads
│   └── tabpfn_wrapper.py         # TabPFN-based table imputer
├── federated/
│   ├── __init__.py
│   ├── base_participant.py       # Abstract server/client base
│   ├── server.py                 # Server logic
│   ├── client.py                 # Client logic
│   └── methods/
│       ├── __init__.py
│       ├── m4fed.py              # Full M4Fed protocol
│       ├── fedavg.py             # FedAvg baseline
│       ├── fedprox.py            # FedProx baseline
│       ├── fan.py                # FAN baseline
│       └── gptfl.py              # GPT-FL baseline
├── training/
│   ├── __init__.py
│   ├── medmd_trainer.py          # MedMD training loop (LDM)
│   └── neurofm_trainer.py        # NeuroFM multi-task training loop
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # MAE, RMSE, Pearson r, R², PSNR, SSIM, LPIPS, FID, KID
│   └── visualization.py          # Loss curves, bar charts, scatter plots, heatmaps
├── utils/
│   ├── __init__.py
│   ├── qlora_utils.py            # QLoRA PEFT helpers
│   ├── seed_utils.py             # Deterministic seed setting
│   └── io_utils.py               # CSV saving, model checkpointing
├── experiments/
│   ├── __init__.py
│   ├── run_all_baselines.py      # Run all 5 methods end-to-end
│   ├── ablation.py               # 3 ablation variants
│   ├── sensitivity.py            # Synthetic-data qty, lr, non-iid α
│   └── cross_site.py             # Cross-site generalisation
├── tests/
│   ├── __init__.py
│   └── unit_test.py              # Tiny-data smoke tests
└── results/                      # Auto-created; CSV + figure outputs
```

---

## 3. Architecture Details

### 3.1 MedMD (img2img LDM)

| Component | Model | Status |
|-----------|-------|--------|
| VAE encoder/decoder | MONAI `AutoencoderKL` (3D, `katielink` zoo) | **Frozen** |
| Denoising U-Net | MONAI `DiffusionModelUNet` (3D) | **Trained** |
| Text encoder | Microsoft `BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | **Frozen** |

**img2img noise schedule (DDIM)**:
- `noise_strength s ∈ (0,1]` → start denoising from timestep `t_start = int(s × T)`
- Noisy latent: `z_{t} = √ᾱ_t · z₀ + √(1−ᾱ_t) · ε`,  `ε ~ N(0,I)`
- Federated loop synthesis: `s > 0.7` (high diversity)
- Local training synthesis: `s ∈ [0.35, 0.70]` (moderate similarity)

**3D cross-attention injection**: text embeddings (shape `[B, L, D_clip]`) are projected to match U-Net hidden dim and injected via spatial-cross-attention at each resolution level.

### 3.2 NeuroFM Multi-task

- Backbone: `NeuroAI-UofG/NeuroFM` (3D brain MRI transformer)
- Task heads: one `Linear` per table field (MSE loss for continuous, CE for categorical)
- Primary task: `age` prediction (brain age)
- QLoRA applied to backbone during client fine-tuning

### 3.3 TabPFN Imputer

- Model: `Prior-Labs/tabpfn_2_5`
- Per-column imputation: train on observed rows, predict missing rows
- Applied to `openneuro_metadata.csv` before text conversion

---

## 4. Federated Protocol (M4Fed)

### Participants
| Node | Data | Models |
|------|------|--------|
| Server | public + private (20 ds) | MedMD (pre-trained) + NeuroFM |
| Client 1 | private (20 ds) | MedMD (copy) + NeuroFM |
| Client 2 | private (20 ds) | MedMD (copy) + NeuroFM |

### Stages

**Warm-up**
1. All nodes run TabPFN table completion on local metadata
2. Server pre-trains MedMD on public data for `warm_rounds` epochs → distributes to clients

**Federated Loop** (repeat `federated_rounds` times)
1. Each client synthesises `N_local / 10` volumes (`s > 0.7`) → sends to server
2. Server fully fine-tunes MedMD on received synthetic data
3. Server synthesises `N_server_private / 10` volumes → sends to each client
4. Each client QLoRA fine-tunes MedMD on received synthetic data

**Local Training** (run `local_rounds` epochs)
1. Each node synthesises `N_local` volumes (`s ∈ [0.35, 0.70]`)
2. Server fully fine-tunes NeuroFM on public + private + synthetic (multi-task)
3. Each client QLoRA fine-tunes NeuroFM on private + synthetic (multi-task)

---

## 5. Loss Functions

### 5.1 MedMD (Denoising Loss)

$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{z_0, \varepsilon \sim \mathcal{N}(0,I), t} \left[ \left\| \varepsilon - \varepsilon_\theta(z_t, t, \mathbf{c}) \right\|_2^2 \right]$$

where $z_0 = \mathcal{E}(x)$ is the VAE-encoded latent, $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon$, and $\mathbf{c}$ is the BioMedCLIP text embedding.

### 5.2 NeuroFM Multi-task

$$\mathcal{L}_{\text{MT}} = \sum_{k=1}^{K} \lambda_k \, \mathcal{L}_k$$

- Continuous field (e.g. age): $\mathcal{L}_k = \text{MSE}(\hat{y}_k, y_k)$
- Categorical field: $\mathcal{L}_k = \text{CrossEntropy}(\hat{y}_k, y_k)$
- Weights $\lambda_k = 1$ (uniform) by default; age head uses $\lambda_{\text{age}} = 2$

### 5.3 FedProx Client Local Objective

$$\mathcal{L}_{\text{FedProx}} = \mathcal{L}_{\text{local}}(w) + \frac{\mu}{2} \|w - w^t\|^2$$

where $w^t$ is the last global model broadcast by the server ($\mu = 0.001$).

### 5.4 FedAvg Aggregation

$$w^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \, w_k^t$$

where $n_k$ is the number of training samples on client $k$ and $n = \sum_k n_k$.

---

## 6. Evaluation Metrics

| Category | Metrics |
|----------|---------|
| Domain model (brain age) | MAE, RMSE, Pearson *r*, R² |
| Generative model (image quality) | PSNR, SSIM, LPIPS, FID, KID |
| Bias analysis | True-vs-Predicted scatter, Age-bin MAE, Residual plot, Error heatmap (age×sex), Error heatmap (age×health\_status) |

---

## 7. Conda / Pip Dependencies

```
# conda channels: conda-forge, pytorch, nvidia
# Core
python=3.10
pytorch>=2.1 torchvision torchaudio pytorch-cuda=12.1

# Medical imaging
monai[all]>=1.3
monai-generative>=0.2   # or install from source

# HuggingFace ecosystem
transformers>=4.40
datasets>=2.18
huggingface_hub>=0.22
peft>=0.10              # QLoRA / LoRA
accelerate>=0.29

# TabPFN
tabpfn>=2.0             # Prior-Labs tabpfn_2_5

# Image quality metrics
lpips>=0.1.4
torch-fidelity>=0.3     # FID, KID
scikit-image>=0.22      # SSIM, PSNR

# Scientific / ML
scikit-learn>=1.4
scipy>=1.12
pandas>=2.2
numpy>=1.26
nibabel>=5.2            # NIfTI I/O
nilearn>=0.10

# Visualisation
matplotlib>=3.8
seaborn>=0.13
svglib

# Utilities
tqdm
pyyaml
omegaconf
bitsandbytes>=0.43      # 4-bit quantisation for QLoRA
einops>=0.7
```

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| HuggingFace streaming datasets | Avoids downloading full dataset to local disk |
| Small batch size (≤2 for 3D volumes) | 16 GB RAM constraint |
| `torch.cuda.amp` mixed precision | Reduces memory footprint |
| `gradient_checkpointing` on U-Net and NeuroFM | Trades compute for memory |
| QLoRA (`r=16`, `lora_alpha=32`, `4-bit`) on clients | Reduces client-side fine-tuning memory |
| Frozen VAE and BioMedCLIP | Stable latent space; fewer trainable parameters |
| DDIM sampler (50 steps default) | Faster inference vs. DDPM |
| Seed fixed globally before every experiment | Full reproducibility |
| MRI resampled to `(96, 96, 96)` | Fits in memory at batch=1 |

---

## 9. Hyperparameter Defaults

| Parameter | Value |
|-----------|-------|
| `warm_rounds` | 5 |
| `federated_rounds` | 10 |
| `local_rounds` | 10 |
| `batch_size` | 1 |
| `lr_medmd` | 1e-4 |
| `lr_neurofm` | 5e-5 |
| `qlora_r` | 16 |
| `qlora_alpha` | 32 |
| `qlora_dropout` | 0.05 |
| `diffusion_timesteps T` | 1000 |
| `ddim_steps` | 50 |
| `noise_strength_fed` | 0.75 |
| `noise_strength_local_min` | 0.35 |
| `noise_strength_local_max` | 0.70 |
| `fedprox_mu` | 0.001 |
| `train_split` | 0.9 |
| `seed` | 42 |
| `n_private_ds_per_node` | 20 |
| `n_cross_ds` | 5 |

---

## 10. Claude Code 生成记录（执行过程审查日志）

> 记录日期：2026-06-12  
> 执行工具：Claude Sonnet 4.6（claude-sonnet-4-6）  
> 工作目录：`c:\Academic\Paper\20260408_FedMiff\Code\M4Fed`

### 10.1 执行步骤概述

| 步骤 | 操作 | 产出文件 |
|------|------|---------|
| 1 | 检查工作区（`ls`），确认为空仓库（仅有 LICENSE） | — |
| 2 | 创建全部子目录（`mkdir -p`） | `configs/ data/ models/medmd/ models/neurofm/ federated/methods/ training/ evaluation/ utils/ experiments/ tests/ results/` |
| 3 | 编写框架文档 | `PROJECT_FRAMEWORK.md` |
| 4 | 编写依赖清单 | `requirements.txt` |
| 5 | 编写配置模块 | `configs/__init__.py`, `configs/base_config.py`, 5 × YAML |
| 6 | 编写数据层 | `data/__init__.py`, `openmind_dataset.py`, `preprocessing.py`, `splits.py`, `table_to_text.py` |
| 7 | 编写模型层 | `models/medmd/{__init__,medmd,vae_wrapper,conditioning}.py`, `models/neurofm/{__init__,neurofm_multitask}.py`, `models/tabpfn_wrapper.py` |
| 8 | 编写工具层 | `utils/{__init__,seed_utils,qlora_utils,io_utils}.py` |
| 9 | 编写训练层 | `training/{__init__,medmd_trainer,neurofm_trainer}.py` |
| 10 | 编写联邦层 | `federated/{__init__,base_participant,server,client}.py`, `federated/methods/{__init__,m4fed,fedavg,fedprox,fan,gptfl}.py` |
| 11 | 编写评估层 | `evaluation/{__init__,metrics,visualization}.py` |
| 12 | 编写实验脚本 | `experiments/{__init__,run_all_baselines,ablation,sensitivity,cross_site}.py` |
| 13 | 编写单元测试 | `tests/{__init__,unit_test}.py` |
| 14 | 编写主入口 | `main.py` |
| 15 | **Bug 修复**：`federated/methods/m4fed.py` — Step-1 客户端合成调用签名错误（将错误传入的 model 对象作为第一个位置参数传给 `synthesise_dataset`，改为正确地创建独立的 `MedMDTrainer(c.medmd, ...)` 并调用）| `federated/methods/m4fed.py` |
| 16 | **Bug 修复**：`federated/server.py` — 移除多余的 `from configs.base_config import ModelConfig` 无效导入；将 `from models import MedMD, NeuroFMMultiTask` 改为直接子模块导入以避免循环引用 | `federated/server.py`, `federated/client.py` |
| 17 | **协议修正**：`_synth_local` 函数新增 `source_loader` 参数，使 Server 的本地合成阶段仅使用 `private_train_loader`（符合协议描述"private data推理生成"），而非混入 public data | `federated/methods/m4fed.py` |
| 18 | 语法检查尝试（`python3 -m py_compile`），结论：当前机器尚无 Python 环境，无法执行 | — |
| 19 | 保存项目记忆文件 | `~/.claude/projects/.../memory/project_m4fed.md` |
| 20 | 追加本执行记录至 `PROJECT_FRAMEWORK.md` | 本节 |

**总计：49 个文件，约 3,200 行代码**

---

### 10.2 各文件职责简述

#### configs/
- **base_config.py**：用 Python dataclass 定义 `DataConfig / ModelConfig / FederatedConfig / TrainingConfig / EvalConfig / M4FedConfig`，支持 YAML 覆盖加载。

#### data/
- **openmind_dataset.py**：`OpenMindDataset`（PyTorch Dataset）+ `build_node_datasets`（按节点分配 cross/private/public 数据集）+ `make_dataloader`。内置 `_make_dummy_samples` 生成假数据用于单测。
- **preprocessing.py**：MRI 3D 体积归一化（clip 0.5/99.5th percentile → [-1,1]）+ 表格清洗。
- **splits.py**：`assign_ds_splits` 按 n_cross/n_private_per_node/n_clients 分配 ds_id；`train_test_split_indices`。
- **table_to_text.py**：将 `openneuro_metadata.csv` 的一行转为自然语言字符串（`"age is 32, sex is female, ..."`）。

#### models/medmd/
- **vae_wrapper.py**：尝试加载 MONAI `AutoencoderKL`，失败时退回 `_MinimalVAE`（8行简单3D VAE），始终冻结参数。
- **conditioning.py**：`BioMedCLIPEncoder` 封装 BioMedCLIP text model，`forward(texts) → (B, seq_len, hidden_dim)`，失败时返回随机嵌入。
- **medmd.py**：`DDIMScheduler`（add_noise / get_timesteps_for_strength / ddim_step）+ `_SimpleUNet3D`（fallback）+ `MedMD`（`forward` 计算 LDM 损失；`synthesise` img2img；`synthesise_from_label` 纯文本生成 for GPT-FL）。

#### models/neurofm/
- **neurofm_multitask.py**：`_Backbone3DCNN`（fallback）+ `_TaskHead`（单任务线性头）+ `NeuroFMMultiTask`（多任务损失，age 权重×2，NaN 掩码处理）。

#### models/tabpfn_wrapper.py
- 按列进行缺失值填补：连续字段用 `TabPFNRegressor`，类别字段用 `TabPFNClassifier`；库不可用时退回中位数/众数。

#### training/
- **medmd_trainer.py**：`train_epoch`（混合精度 AMP + grad clip）+ `synthesise_dataset`（批量 img2img 生成，返回 sample dict 列表）。
- **neurofm_trainer.py**：`train_epoch`（多任务损失 + FedProx proximal term）+ `evaluate`（返回 `y_true, y_pred` numpy 数组）。

#### federated/
- **base_participant.py**：抽象基类，提供 `get/set_neurofm_state`、`get/set_medmd_unet_state`、`n_train_samples` 等公共接口。
- **server.py**：`Server` 持有 public+private 合并 DataLoader；`add_synthetic_data_to_loader` 返回混合 DataLoader。
- **client.py**：`Client` 仅持有 private DataLoader。

#### federated/methods/
- **m4fed.py**：完整三阶段协议（Warm-up → Federated Loop → Local Training）；`_eval_node` 统一评估；结果写 CSV。
- **fedavg.py**：`_fedavg_aggregate`（sample-size 加权平均）；每轮 broadcast → local update → aggregate。
- **fedprox.py**：同 FedAvg，`NeuroFMTrainer(proximal_mu=0.001, global_state=...)` 注入近端项。
- **fan.py**：warm-up 阶段 MONAI augment（随机翻转）+ 客户端互发合成数据 + 本地训练。
- **gptfl.py**：Server 用 `synthesise_from_label(age_val)` 纯 label 生成 public-sized 合成集；客户端接收后与本地数据融合训练。

#### evaluation/
- **metrics.py**：`compute_domain_metrics`（scipy/sklearn）；`compute_image_metrics`（skimage PSNR/SSIM + lpips 中间切片）；`compute_fid_kid`（torch-fidelity，保存临时 PNG）；`compute_bias_analysis`（age-bin MAE、残差、性别/健康状态 pivot heatmap）。
- **visualization.py**：所有图均使用 Carlito 字体（自动 fallback）+ 6色 palette；`plot_loss_curves`（阴影区间表示 std）；`plot_metric_bars`（误差棒）；5种偏差分析图；`save_figure`（svg/pdf）。

#### experiments/
- **run_all_baselines.py**：顺序运行5个方法 → 汇总 CSV → 4张指标柱状图。
- **ablation.py**：3 variants（w/o public data / w/o TabPFN / random-init NeuroFM）各自运行 M4Fed。
- **sensitivity.py**：9组配置（synth ratio × 3 + lr × 3 + Dirichlet α × 3）+ `dirichlet_resplit` 重新分配数据。
- **cross_site.py**：加载 Server NeuroFM checkpoint → 在 cross dataset 上推理评估 → 保存偏差分析图。

#### tests/unit_test.py
- 17 个独立测试函数，使用 `mri_shape=(8,8,8)` + 4 样本；覆盖从数据预处理到完整 M4Fed/FedAvg/FedProx 单轮的全链路。运行：`python main.py --mode test`。

---

### 10.3 已知限制与注意事项

| 类别 | 说明 |
|------|------|
| 依赖安装 | 需先按 `requirements.txt` 建立 conda/pip 环境；MONAI GenerativeModels 需从源码安装（`pip install git+https://github.com/Project-MONAI/GenerativeModels.git`） |
| NeuroFM 加载 | `NeuroAI-UofG/NeuroFM` 需要 `trust_remote_code=True`；若模型结构与预期不符，自动退回 `_Backbone3DCNN` |
| VAE 权重 | `katielink` 收藏中的具体 repo 名需确认；目前 `vae_repo` 默认值为占位符，实际使用需更新 `configs/m4fed.yaml` 中的 `vae_repo` 字段 |
| HuggingFace streaming | OpenMind 数据集的字段名（`dataset_id` / `ds_id` 等）需与实际 schema 对齐，可能需调整 `openmind_dataset.py` 中的字段名 |
| FID/KID | 需要 `torch-fidelity`；样本数 <50 时 KID 会自动降低 `kid_subset_size` |
| Python 环境 | 截至 2026-06-12 本机尚无 Python 环境，未能运行语法检查；安装环境后建议先执行 `python main.py --mode test` 验证 |
