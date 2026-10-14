# M4Fed: Multi-Source Federated Learning with Medical Foundation Models for Multi-Modal Medical Data

This repository contains the official implementation of **M4Fed** (**M**ulti-source **M**ulti-modal **M**edical Foundation-model **Fed**erated Learning), together with the four federated baselines and every experiment reported in the paper:

> **Multi-Source Federated Learning with Medical Foundation Models for Multi-Modal Medical Data**
> *(manuscript under review)*

The framework is evaluated on **brain-age estimation** from T1-weighted MRI using the public multi-site [OpenMind](https://huggingface.co/datasets/AnonRes/OpenMind) corpus (7,753 subjects aged 4–84 from 153 OpenNeuro centers).

---

## About the paper

Medical AI depends on three resources that rarely meet in one training pipeline:

- **Private clinical data**, locked inside hospitals by privacy regulation (HIPAA, GDPR);
- **Public repositories**, which are large but heterogeneous and sparsely annotated;
- **Pretrained foundation models**, which are capable but still need adaptation to a target population.

Combining all three runs into three constraints at once: raw patient data cannot be centralized, exchanging the weights of a foundation-scale model every round costs too much bandwidth, and hospital nodes rarely have the GPUs to fine-tune such a model end to end. Existing federated methods address these one at a time: FedAvg/FedProx handle private data, FedDF adds public data, FedPEFT/OpenFedLLM add foundation models, and GPT-FL adds synthetic data. None of them handles all three resources under all three constraints.

M4Fed covers all of them in a single framework. Its main results:

- The **M4Fed server** reaches the lowest brain-age MAE of all methods (**8.87 yr**, 13.7% below the best baseline) and is the only method with a positive R².
- **M4Fed clients** rank subjects by age better than every baseline (Pearson r **0.442** vs. 0.248 for the best baseline).
- QLoRA cuts client-side trainable parameters by **81.9%** for the domain model and **92.6%** for the diffusion U-Net.
- At a synthesis ratio of ρ_f = 0.01, the per-round upload is **42.6% smaller** than FedAvg's, with no loss in server accuracy. Because the payload does not grow with model size, the gap widens for larger backbones (about 19.5× smaller than FedAvg for a 1B-parameter model at ρ_f = 0.1).

## About M4Fed

### Three models on every node

The server and each client all run the same three models:

| Role | Model | What it does | Trainable parts |
|---|---|---|---|
| Imputation model | **TabPFN** (`Prior-Labs/tabpfn_2_5`) | Fills in missing metadata fields (age, sex, BMI, handedness, health status, …) in-context, separately on each node | none (in-context inference) |
| Generative model | **MedMD**: a text-conditioned 3D latent diffusion model | Synthesizes 96³ brain MRI volumes by img2img DDIM, conditioned on the imputed metadata written out as text | Denoising U-Net only. The MONAI AutoencoderKL and the BiomedCLIP text encoder stay frozen |
| Domain model | **SwinMTL**: a Swin UNETR encoder with multi-task MLP heads | Predicts age (primary task, weight 2) plus auxiliary fields | Server: full fine-tuning. Clients: QLoRA adapters (r=16, 4-bit backbone) and task heads |

### Three-phase protocol

1. **Warm-up.** Each node imputes its own metadata with TabPFN. The server then pretrains MedMD on public data and sends the weights to every client once.
2. **Federated loop.** Clients synthesize ⌈ρ_f · n_k⌉ high-noise volumes (σ = 0.75) from their private data and send them to the server, which fine-tunes its MedMD on them. The server then synthesizes from its own private data and broadcasts the result, and clients QLoRA-fine-tune their MedMD on it. **Nodes exchange synthetic volumes only: no raw data, raw metadata or gradients.**
3. **Local training.** Each node uses MedMD to augment its private data at moderate noise (σ ~ U[0.35, 0.70], ρ_l = 1.0) and then fine-tunes SwinMTL on it. The server uses full-parameter updates; clients use QLoRA only.

Communication is set by the synthesis ratio ρ_f rather than by model size, and edge compute is limited to low-rank adapters. All three models are pluggable, so each can be swapped for a model better suited to another sub-domain.

### Baselines

| Method | Reference | Implementation |
|---|---|---|
| FedAvg | McMahan et al., 2017 | [federated/methods/fedavg.py](federated/methods/fedavg.py) |
| FedProx | Li et al., 2020 | [federated/methods/fedprox.py](federated/methods/fedprox.py) |
| FAN | Wang et al., 2026 | [federated/methods/fan.py](federated/methods/fan.py) |
| GPT-FL | Zhang et al., 2025 | [federated/methods/gptfl.py](federated/methods/gptfl.py) |
| **M4Fed** (ours) | this paper | [federated/methods/m4fed.py](federated/methods/m4fed.py) |

## Repository structure

```
M4Fed/
├── main.py                  # Unified entry point (--mode run_all | test | ablation | sensitivity | cross_site | visualize)
├── configs/                 # YAML configs for M4Fed and each baseline + dataclass defaults (base_config.py)
├── data/                    # OpenMind loader (lazy HF download + LRU disk cache), splits, preprocessing,
│                            #   table-to-text serialization, MNI152 registration, known bad-sample lists
├── models/
│   ├── medmd/               # MedMD: VAE wrapper, BiomedCLIP conditioning, latent diffusion U-Net
│   ├── swin/                # SwinMTL: Swin UNETR multi-task predictor
│   └── tabpfn_wrapper.py    # TabPFN metadata imputation
├── federated/               # Server, client, and the five federated methods
├── training/                # MedMD and SwinMTL trainers
├── evaluation/              # Regression metrics, generative metrics (FID/KID/SSIM/LPIPS), plots
├── experiments/             # Main comparison, ablation, sensitivity, cross-site, healthy-subset studies
├── utils/                   # QLoRA, seeding, logging, timing, I/O helpers
├── tests/unit_test.py       # Smoke tests for every experiment suite
├── run_*.sh                 # SLURM job scripts (run_*_etro.sh are variants for a second cluster)
└── openneuro_metadata.csv   # OpenMind metadata table (one row per scan)
```

## Reproducing the results

### 1. Environment

The experiments were run with Python 3.10+ and a single NVIDIA A100/H100 GPU (64 GB RAM, 8 CPU cores per job).

```bash
conda create -n m4fed python=3.11 -y
conda activate m4fed

# Install the CUDA build of PyTorch first (pick the CUDA tag that matches your driver;
# see https://pytorch.org/get-started/locally/). Example for CUDA 12.1:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Then install the remaining dependencies
pip install -r requirements.txt
```

> On Windows, the default PyPI `torch` wheel is CPU-only, so always install PyTorch from the CUDA index *before* running `pip install -r requirements.txt`. `bitsandbytes` is optional: without it, QLoRA falls back to standard LoRA without 4-bit quantization.

### 2. Credentials

Create a `.env.local` file in the repository root. It is git-ignored and loaded automatically by every entry point:

```
TABPFN_TOKEN=tabpfn_sk_xxxxxxxx   # required for TabPFN v2.5 weights
HF_TOKEN=hf_xxxxxxxx              # HuggingFace token (OpenMind data and pretrained models)
```

### 3. Data

You do not need to download anything by hand. On first access, MRI volumes are fetched one by one from [`AnonRes/OpenMind`](https://huggingface.co/datasets/AnonRes/OpenMind) into `data/mri_cache/`. Pretrained weights for the VAE, BiomedCLIP, Swin UNETR SSL and TabPFN go to `data/model_cache/`.

The full OpenMind repository is about 1 TB. Two settings in [configs/base_config.py](configs/base_config.py) keep disk use bounded:

- `data.n_public_ds_max` (default 20) caps the number of public sub-datasets;
- `data.max_cache_gb` (default 50) caps the MRI cache, evicting the least recently used files first.

Scans known to be corrupted or to contain non-finite voxels are excluded automatically (see [data/known_unusable_samples.csv](data/known_unusable_samples.csv) and [data/known_nan_samples.csv](data/known_nan_samples.csv)).

### 4. Smoke test (optional, a few minutes)

```bash
python main.py --mode test                       # all base suites on tiny dummy data
python tests/unit_test.py --suite baseline       # or a single suite: baseline | ablation | sensitivity | cross_site | visualization
```

### 5. Paper experiments

All hyperparameters default to [configs/m4fed.yaml](configs/m4fed.yaml): 2 clients + 1 server, R_w / R_f / R_l = 1 / 3 / 3, ρ_f = 0.1, ρ_l = 1.0, seed 42. Each `run_*.sh` is a SLURM script; to run it without SLURM, use the `python` line it contains. **Run the main comparison first**, because cross-site evaluation and visualization reuse its checkpoints.

| Paper section | Command | Output |
|---|---|---|
| Main comparison with baselines (Table: main results), empirical communication / compute cost, generative quality | `python main.py --mode run_all --config configs/m4fed.yaml --results_dir results/main` | `results/main/` |
| Ablation on M4Fed components | `python main.py --mode ablation --config configs/m4fed.yaml --results_dir results/ablation` | `results/ablation/` |
| Sensitivity to ρ_f, MedMD learning rate and Dirichlet data skew | `python main.py --mode sensitivity --config configs/m4fed.yaml --results_dir results/sensitivity` | `results/sensitivity/` |
| Cross-site generalization (*requires `results/main/m4fed/server_swin.pt`*) | `python main.py --mode cross_site --config configs/m4fed.yaml --results_dir results/main` | `results/main/` |
| Synthesized MRI slices (*requires run_all checkpoints*) | `python main.py --mode visualize --config configs/m4fed.yaml --results_dir results/main` | `results/main/` |
| Ablation on health status (healthy-controls cohort) | `python experiments/healthy_subset.py --config configs/m4fed.yaml --results_dir results/healthy_subset` | `results/healthy_subset/` |
| Noise strength σ = 0.05 vs. default | `python experiments/healthy_subset_low_noise.py --config configs/m4fed.yaml --results_dir results/healthy_subset_low_noise` | `results/healthy_subset_low_noise/` |
| MNI152 registration vs. end-to-end: step 1, register (CPU only) | `python experiments/register_healthy_subset_mri.py --config configs/m4fed.yaml --results_dir results/register_healthy_subset_mri --workers 16` | `data/mri_registered_cache/` |
| MNI152 registration vs. end-to-end: step 2, train | `python experiments/healthy_subset_registered.py --config configs/m4fed.yaml --results_dir results/healthy_subset_registered` | `results/healthy_subset_registered/` |

On a SLURM cluster, the same experiments can be submitted in order with:

```bash
sbatch run_main.sh            # must finish before the next two
sbatch run_cross_site.sh
sbatch run_visualize.sh
sbatch run_ablation.sh
sbatch run_sensitivity.sh
sbatch run_healthy_subset.sh
```

Before submitting, edit the `cd` path, conda setup and partition names at the top of each script to match your cluster.

Each results directory contains per-node metrics (MAE, RMSE, Pearson r, R²), generative metrics (FID, KID, SSIM, LPIPS), logged communication bytes and client FLOPs, loss curves, and figures. Client metrics are reported as mean ± std across client nodes, as in the paper.

> **Note on runtime.** The main comparison trains five methods on 3D volumes, so the provided SLURM scripts request a 2–4 day wall-time limit on one A100. Exact numbers can vary slightly across GPU types and library versions, even with a fixed seed.

## Citation

If you find this work useful, please cite:

```bibtex
@article{m4fed,
  title   = {Multi-Source Federated Learning with Medical Foundation Models for Multi-Modal Medical Data},
  author  = {Xinguang Wang et al.},
  journal = {Under review},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

This work builds on [MONAI](https://monai.io/) (AutoencoderKL, DiffusionModelUNet, Swin UNETR), [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), [TabPFN](https://github.com/PriorLabs/TabPFN), [PEFT](https://github.com/huggingface/peft) and [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), and uses the [OpenMind](https://huggingface.co/datasets/AnonRes/OpenMind) dataset derived from [OpenNeuro](https://openneuro.org/).
