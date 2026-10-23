# M4Fed: Multi-Source Federated Learning with Medical Foundation Models for Multi-Modal Medical Data

This repository contains the official implementation of **M4Fed** (**M**ulti-source **M**ulti-modal **M**edical Foundation-model **Fed**erated Learning). It also includes the four federated baselines and every experiment reported in the paper:

> **Multi-Source Federated Learning with Medical Foundation Models for Multi-Modal Medical Data on Brain Age Estimation**
> Xinguang Wang, Nguyen Linh Trung, Oliver Y. Chen, Shujian Yu, Alvaro Javier Vargas Guerrero, Quang Manh Doan, Tianzheng Hu, Hai Anh Nguyen, Stijn Denissen, Jeroen Van Schependom, Guy Nagels
> *Manuscript under review.*

M4Fed is evaluated on **brain-age estimation** from T1-weighted MRI, using the public multi-site [OpenMind](https://huggingface.co/datasets/AnonRes/OpenMind) corpus. Two cohorts are used: an *unselected* cohort (3,622 subjects from 85 OpenNeuro sub-datasets) and a *healthy-controls* cohort (4,571 subjects from 64 sub-datasets), aged 4–84.

For a file-by-file description of the code, see [PROJECT_FRAMEWORK.md](PROJECT_FRAMEWORK.md).

---

## About the paper

Medical AI draws on three resources that rarely meet in one training pipeline:

- **Private clinical data**, kept inside hospitals by privacy regulation (HIPAA, GDPR);
- **Public repositories**, which are large but heterogeneous and sparsely annotated;
- **Pretrained foundation models**, which are capable but still need adaptation to a target population.

Using all three together runs into three constraints at once. Raw patient data cannot be centralized. Exchanging the weights of a foundation-scale model every round costs too much bandwidth. And hospital nodes rarely have the GPUs to fine-tune such a model end to end. Existing federated methods each handle part of this:

| Method | Privacy | Communication | Edge compute | Private data | Public data | Foundation model |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| FedAvg / FedProx | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| FedDF | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| FedPEFT / OpenFedLLM | ✓ | △ | ✓ | ✓ | ✗ | ✓ |
| FAN | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| GPT-FL | ✓ | △ | ✗ | ✓ | ✓ | ✓ |
| **M4Fed** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

M4Fed covers all of them in one framework. Its main results:

- The **M4Fed server** has the lowest brain-age MAE of all methods (**8.87 yr**, 13.7% below the best baseline, GPT-FL) and is the only method with a positive R² (0.178).
- **M4Fed clients** rank subjects by age better than every baseline (Pearson r **0.442** vs. 0.248 for GPT-FL), although their MAE is not lower.
- QLoRA cuts client-side trainable parameters by **81.9%** for the domain model (9.25M → 1.67M) and by **92.6%** for the diffusion U-Net (7.54M → 0.56M).
- At a synthesis ratio of ρ_f = 0.01, the per-round upload is **42.6% smaller** than FedAvg's, with no loss in server accuracy. The payload does not grow with model size, so the gap widens for larger backbones: for a 1B-parameter domain model, M4Fed's upload at ρ_f = 0.1 would be about 19.5× smaller than FedAvg's.
- On held-out acquisition sites, the server's MAE (9.42 yr) is still lower than every baseline's in-federation MAE.
- On the healthy-controls cohort, affine MNI152 registration and per-node z-scoring of age labels further improve the result. A centralized model trained without privacy constraints reaches 3.51 yr MAE, an upper reference rather than a competing method.

## About M4Fed

### Three models on every node

The server and each client all run the same three models:

| Role | Model | What it does | Trainable parts |
|---|---|---|---|
| Imputation model | **TabPFN** (`Prior-Labs/tabpfn_2_5`) | Fills in missing metadata fields (age, weight, BMI, sex, handedness, health status, …) in-context, separately on each node | None (in-context inference) |
| Generative model | **MedMD**: a text-conditioned 3D latent diffusion model | Synthesizes 96³ brain MRI volumes by img2img DDIM, conditioned on the imputed metadata written out as text | Denoising U-Net only. The MONAI AutoencoderKL and the BiomedCLIP text encoder stay frozen |
| Domain model | **SwinMTL**: a Swin UNETR encoder with multi-task MLP heads | Predicts age (primary task, loss weight 2) plus auxiliary fields (sex, health status, race) | Server: full fine-tuning. Clients: QLoRA adapters (r = 16, 4-bit NF4 backbone) and task heads |

All three are pluggable: each can be replaced by a model better suited to another sub-domain.

### Three-phase protocol

1. **Warm-up.** Each node imputes its own metadata with TabPFN. The server then pretrains MedMD on public data and sends the U-Net weights to every client once.
2. **Federated loop.** Each client synthesizes ⌈ρ_f · n_k⌉ high-noise volumes (σ = 0.75) from its private data and sends them to the server, which fine-tunes its MedMD on them. The server then synthesizes volumes from its own private data and broadcasts them, and clients QLoRA-fine-tune their MedMD on them. **Nodes exchange synthetic volumes only: no raw images, raw metadata or gradients.**
3. **Local training.** Each node uses MedMD to augment its private data at moderate noise (σ ~ U[0.35, 0.70], ρ_l = 1.0), then fine-tunes SwinMTL on it. The server uses full-parameter updates; clients train QLoRA adapters only.

Communication is set by the synthesis ratio ρ_f rather than by model size, and edge compute is limited to low-rank adapters.

### Baselines

All baselines are re-implemented in the same codebase, data pipeline and evaluation harness, using the same SwinMTL domain model.

| Method | Reference | Implementation |
|---|---|---|
| FedAvg | McMahan et al., 2017 | [federated/methods/fedavg.py](federated/methods/fedavg.py) |
| FedProx | Li et al., 2020 | [federated/methods/fedprox.py](federated/methods/fedprox.py) |
| FAN | Wang et al., 2026 | [federated/methods/fan.py](federated/methods/fan.py) |
| GPT-FL | Zhang et al., 2025 | [federated/methods/gptfl.py](federated/methods/gptfl.py) |
| **M4Fed** (ours) | this paper | [federated/methods/m4fed.py](federated/methods/m4fed.py) |

M4Fed uses one server and two clients. The baselines use three clients, and their server only aggregates and broadcasts. GPT-FL reuses the generator that M4Fed's server pretrains during warm-up.

## Repository structure

```
M4Fed/
├── main.py                  # Entry point: --mode test | run_all | ablation | sensitivity | cross_site | visualize
├── configs/                 # Dataclass defaults (base_config.py) + one YAML per method
├── data/                    # OpenMind loader (lazy HF download + LRU disk cache), sub-dataset splits,
│                            #   preprocessing, table-to-text, MNI152 registration, known bad-sample lists
├── models/
│   ├── medmd/               # MedMD: VAE wrapper, BiomedCLIP conditioning, latent diffusion U-Net + DDIM
│   ├── swin/                # SwinMTL (class SwinMultiTask): Swin UNETR encoder + multi-task heads
│   └── tabpfn_wrapper.py    # TabPFN metadata imputation
├── federated/               # Server, client, and the five federated methods
├── training/                # MedMD and SwinMTL trainers
├── evaluation/              # Regression metrics, generative metrics (PSNR/SSIM/LPIPS/FID/KID), bias plots
├── experiments/             # Every experiment in the paper (see the table below)
├── utils/                   # QLoRA, seeding, logging, timing, data-quality logs, .env loader
├── tests/unit_test.py       # Smoke tests on tiny dummy data
├── run_*.sh                 # SLURM job scripts (run_*_etro.sh: variants used on our cluster)
└── openneuro_metadata.csv   # OpenMind metadata table (one row per scan)
```

## Reproducing the results

### 1. Environment

The experiments ran with Python 3.11 on one NVIDIA A100 GPU per job (64 GB RAM, 8 CPU cores).

```bash
conda create -n m4fed python=3.11 -y
conda activate m4fed

# Install the CUDA build of PyTorch first (pick the CUDA tag that matches your driver;
# see https://pytorch.org/get-started/locally/). Example for CUDA 12.1:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Then install the remaining dependencies
pip install -r requirements.txt
```

> On Windows, the default PyPI `torch` wheel is CPU-only, so install PyTorch from the CUDA index *before* running `pip install -r requirements.txt`. `bitsandbytes` is optional: without it, QLoRA falls back to standard LoRA without 4-bit quantization.

### 2. Credentials

Create a `.env.local` file in the repository root. It is git-ignored and loaded automatically by every entry point:

```
TABPFN_TOKEN=tabpfn_sk_xxxxxxxx   # required for TabPFN v2.5 weights
HF_TOKEN=hf_xxxxxxxx              # HuggingFace token (OpenMind data and pretrained models)
```

### 3. Data

You do not need to download anything by hand:

- MRI volumes are fetched one at a time from [`AnonRes/OpenMind`](https://huggingface.co/datasets/AnonRes/OpenMind) into `data/mri_cache/` the first time they are read.
- Pretrained weights for the VAE, BiomedCLIP, Swin UNETR (self-supervised) and TabPFN go to `data/model_cache/`.
- The MNI152 template for the registration experiments is fetched by nilearn into `data/mni_template/`, and registered volumes are cached in `data/mri_registered_cache/`.

Only T1w scans with a recorded age are used. The full OpenMind repository is about 1 TB; two settings in [configs/base_config.py](configs/base_config.py) keep disk use bounded:

- `data.n_public_ds_max` (default 20) caps the number of public sub-datasets;
- `data.max_cache_gb` (default 80) caps the MRI cache, evicting the least recently used files first.

Scans known to be corrupted or to contain non-finite voxels are skipped automatically (see [data/known_unusable_samples.csv](data/known_unusable_samples.csv) and [data/known_nan_samples.csv](data/known_nan_samples.csv)).

### 4. Smoke test (optional, a few minutes)

```bash
python main.py --mode test                       # all base suites on tiny dummy data
python tests/unit_test.py --suite baseline       # or one suite: baseline | ablation | sensitivity | cross_site | visualization
```

### 5. Paper experiments

All runs read their hyperparameters from [configs/m4fed.yaml](configs/m4fed.yaml) on top of the defaults in [configs/base_config.py](configs/base_config.py): 2 clients + 1 server, R_w / R_f / R_l = 1 / 3 / 3, ρ_f = 0.1, ρ_l = 1.0, seed 42. Some healthy-cohort scripts override the round counts in memory; the table says which.

**Run the main comparison first.** Cross-site evaluation, visualization, the health-status ablation and the prediction export all reuse its checkpoints in `results/main/`.

**Unselected cohort** (sub-datasets assigned to nodes by ID)

| Paper section | Command | Output |
|---|---|---|
| Main comparison with baselines; generative quality; communication and compute cost | `python main.py --mode run_all --config configs/m4fed.yaml --results_dir results/main` | `results/main/` |
| Ablation on M4Fed components, and ablation on health status (unselected-cohort rows) | `python main.py --mode ablation --config configs/m4fed.yaml --results_dir results/ablation` | `results/ablation/` |
| Sensitivity to ρ_f, MedMD learning rate and Dirichlet α | `python main.py --mode sensitivity --config configs/m4fed.yaml --results_dir results/sensitivity` | `results/sensitivity/` |
| Cross-site generalization | `python main.py --mode cross_site --config configs/m4fed.yaml --results_dir results/main` | `results/main/` |
| Synthesized MRI slices | `python main.py --mode visualize --config configs/m4fed.yaml --results_dir results/main` | `results/main/visualization/` |
| Per-scan predictions and scatter plots (no retraining) | `python experiments/export_test_predictions.py --config configs/m4fed.yaml --results_dir results/main` | `results/main/<method>/` |

**Healthy-controls cohort** (subjects split into four equal blocks: server-public, server-private, client 0, client 1)

| Paper section | Command | Rounds R_w/R_f/R_l | Output |
|---|---|---|---|
| Healthy-controls reference run (end-to-end, unregistered) | `python experiments/healthy_subset.py --config configs/m4fed.yaml --results_dir results/healthy_subset` | 1/3/3 | `results/healthy_subset/` |
| Noise strength σ = 0.05 | `python experiments/healthy_subset_low_noise.py --config configs/m4fed.yaml --results_dir results/healthy_subset_low_noise` | 1/3/3 | `results/healthy_subset_low_noise/` |
| Registration, step 1: register every scan to MNI152 (CPU only, resumable) | `python experiments/register_healthy_subset_mri.py --config configs/m4fed.yaml --results_dir results/register_healthy_subset_mri --workers 16` | – | `data/mri_registered_cache/` |
| Registration, step 2: train on registered volumes | `python experiments/healthy_subset_registered.py --config configs/m4fed.yaml --results_dir results/healthy_subset_registered_133` | 1/3/3 | `results/healthy_subset_registered_133/` |
| Label z-score standardization (registered volumes) | `python experiments/healthy_subset_registered_age_zscore.py --config configs/m4fed.yaml --results_dir results/healthy_subset_registered_age_zscore --baseline_dir results/healthy_subset_registered_133` | 1/3/3 | `results/healthy_subset_registered_age_zscore/` |
| Centralized training (upper reference) | `python experiments/centralized_healthy_registered_age_zscore.py --config configs/m4fed.yaml --results_dir results/centralized_healthy_registered_age_zscore --split_dir results/healthy_subset_registered_133 --epochs 10` | 10 epochs | `results/centralized_healthy_registered_age_zscore/` |

The z-score and centralized runs reuse the subject split saved in `results/healthy_subset_registered_133/healthy_split_assignment.csv`, so all three runs share the same test scans. To export per-scan predictions from a finished healthy-cohort run, use `experiments/export_healthy_subset_predictions.py --results_dir <dir>` (add `--no_register` for unregistered runs).

On a SLURM cluster, submit the matching scripts in this order:

```bash
sbatch run_main.sh                                   # must finish before the next four
sbatch run_cross_site.sh
sbatch run_visualize.sh
sbatch run_ablation.sh
sbatch run_export_test_predictions_etro.sh
sbatch run_sensitivity.sh
sbatch run_healthy_subset.sh
sbatch run_healthy_subset_low_noise_etro.sh
sbatch run_register_healthy_subset_mri_etro.sh       # must finish before the registered runs
sbatch run_healthy_subset_registered_etro.sh         # writes to results/healthy_subset_registered;
                                                     #   later scripts expect results/healthy_subset_registered_133
sbatch run_healthy_subset_registered_age_zscore_etro.sh
sbatch run_centralized_healthy_registered_age_zscore_etro.sh
```

Before submitting, edit the `cd` path, conda setup and partition name at the top of each script to match your cluster.

### 6. Outputs

Each results directory contains:

- domain metrics per node (MAE, RMSE, Pearson r, R²) and a `summary_domain_metrics.csv`, where client metrics are mean ± std across clients, as in the paper;
- generative metrics (PSNR, SSIM, LPIPS, FID, KID) for methods with a generative model;
- per-round loss curves, bias plots (age-bin MAE, residuals, age × sex and age × health-status heatmaps);
- logged communication bytes, client FLOPs and a `timing_breakdown.csv`;
- model checkpoints (`<node>_swin.pt`, `<node>_medmd.pt`) and an `unusable_samples.csv` listing scans skipped during the run.

> **Note on runtime.** The main comparison trains five methods on 3D volumes, so the SLURM scripts request up to 4 days on one A100. Exact numbers can vary slightly across GPU types and library versions, even with a fixed seed.

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026m4fed,
  title   = {Multi-Source Federated Learning with Medical Foundation Models for Multi-Modal Medical Data on Brain Age Estimation},
  author  = {Wang, Xinguang and Trung, Nguyen Linh and Chen, Oliver Y. and Yu, Shujian and Vargas Guerrero, Alvaro Javier and Doan, Quang Manh and Hu, Tianzheng and Nguyen, Hai Anh and Denissen, Stijn and Van Schependom, Jeroen and Nagels, Guy},
  journal = {Under review},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

This work was supported in part by the Research Foundation Flanders (grant 1805620N) and the China Scholarship Council (grant 202408230018).

The code builds on [MONAI](https://monai.io/) (AutoencoderKL, DiffusionModelUNet, Swin UNETR), [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), [TabPFN](https://github.com/PriorLabs/TabPFN), [PEFT](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), [SimpleITK](https://simpleitk.org/) and [nilearn](https://nilearn.github.io/). It uses the [OpenMind](https://huggingface.co/datasets/AnonRes/OpenMind) dataset derived from [OpenNeuro](https://openneuro.org/).
