# Generative AI for Modeling Single-Cell Data

Flow matching models for generating synthetic scRNA-seq expression profiles, benchmarked on the PBMC 3K dataset.

## Overview

Single-cell RNA sequencing (scRNA-seq) gives gene-level expression measurements for thousands of individual cells, but real datasets are expensive to collect and biased toward common cell types. Generative models can produce realistic synthetic cells that augment scarce data and support downstream method development.

This repository implements and compares three flow-matching pipelines on the PBMC 3K dataset (2,638 immune cells, 2,000 highly variable genes), in both unconditional and Leiden-cluster-conditional settings:

- **FM-Gene** — flow matching directly in the 2,000-dim gene space.
- **FM-PCA** — flow matching in a 20-dim PCA latent space, decoded by inverse PCA.
- **FM-AE** — flow matching in a 20-dim autoencoder latent space, decoded by the AE.

Generated cells are evaluated against two established baselines (scGAN, scDiffusion) along five complementary tiers — point-level fidelity (Mean / Std MSE), population-level distribution similarity (MMD, Wasserstein, KL), biological-signal preservation (Common-DEGs), generalization (memorization ratio), and qualitative manifold preservation (joint UMAP). Final results are summarized in the BIS 681 written report and the final-presentation deck under `Documentation/` and `Presentation/`.

## Companion repository — cscGAN baseline

The cscGAN baseline used throughout the comparisons in this repo lives in a separate repository because it requires a different runtime (TensorFlow 1.15 inside Docker / CUDA 11.8) than the PyTorch-based Flow Matching code here:

> **[ZhuoyuanJiang/Single_Cell_scGAN](https://github.com/ZhuoyuanJiang/Single_Cell_scGAN)** — fork of [imsb-uke/scGAN](https://github.com/imsb-uke/scGAN), retrained on PBMC 3K with two bug fixes (ReLU/standardized-data mismatch, learning-rate schedule). The final cscGAN comparison result (epoch-4000 checkpoint) is the row you'll see in this repo's evaluation tables.

## Project Structure

```
GenAI_SingleCell/
├── config.py                # All hyperparameters and shared paths
├── src/
│   ├── preprocess.py        # Data loading, QC, normalization, HVG selection, splits
│   ├── models_ae.py         # Autoencoder model and training utilities
│   ├── models_flow.py       # Flow matching model and training/sampling utilities
│   └── metrics.py           # Evaluation metrics (MMD, Wasserstein, KL, Common-DEGs, UMAP, memorization)
├── notebooks/               # Numbered, run in order
│   ├── 01_preprocess_pbmc3k # Download & preprocess PBMC 3K, save train/val/test splits
│   ├── 02_pipeline_pca_flow # Pipeline 1: PCA latent space + flow matching
│   ├── 03_pipeline_ae_flow  # Pipeline 2: Autoencoder latent space + flow matching
│   ├── 04_pipeline_gene_flow# Pipeline 3: Direct gene-space flow matching
│   ├── 05_evaluation        # Compare all three pipelines side by side
│   └── 06_conditional_flow  # Conditional flow matching (all three variants)
├── scripts/                 # Standalone utility scripts
│   └── generate_leiden_labels.py    # Reproduce Leiden cluster labels
├── Presentation/            # Final presentation deck and report assets
├── data/                    # Raw .h5ad files (auto-downloaded, gitignored)
├── artifacts/               # Generated outputs (gitignored)
│   ├── data/                # Preprocessed arrays, scalers, splits
│   ├── models/              # Trained model checkpoints (.pt, .joblib)
│   └── figures/             # Saved plots
├── teammates/               # Reference notebooks from teammates (gitignored)
├── requirements.txt         # Pip dependencies
└── environment.yml          # Full conda environment export
```

## Setup

Tested on Linux / WSL2 with an NVIDIA GPU (CUDA-capable, e.g. RTX 4070). Training runs on CPU but is much slower; the GPU is recommended for FM-Gene and the AE.

```bash
# 1. Clone the repo
git clone https://github.com/<your-fork>/GenAI_SingleCell.git
cd GenAI_SingleCell

# 2a. Conda (recommended — reproduces the exact environment)
conda env create -f environment.yml
conda activate GenAI_single_cell

# 2b. Or pip (use a fresh venv)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

After install, launch Jupyter and run the notebooks in numerical order — see [Running the Pipelines](#running-the-pipelines) below. Notebook 01 downloads PBMC 3K automatically into `data/`; everything generated downstream lands under `artifacts/`. Both directories are gitignored, so a fresh clone produces no leftover files.

## Running the Pipelines

Run the notebooks in order. Notebook 01 downloads the data and creates the train/val/test split; notebooks 02–04 each train a different pipeline; notebook 05 compares them; notebook 06 runs the conditional variants of all three pipelines.

```bash
# From the repo root:
jupyter notebook notebooks/01_preprocess_pbmc3k.ipynb
# then 02, 03, 04, 05, 06 ...
```

All outputs (processed data, models, figures) are saved under `artifacts/`.

## Three Pipelines

1. **PCA + Flow Matching** — Reduce to 20-dim PCA, train flow model in latent space, reconstruct via inverse PCA.
2. **Autoencoder + Flow Matching** — Learn a 20-dim AE latent space, train flow model there, decode back.
3. **Direct Gene-Space Flow Matching** — Train flow model directly on 2000 HVG expression values.

## Config

All hyperparameters live in `config.py`. Change them there and every notebook picks them up — no need to edit notebooks individually.

## Acknowledgments

This was a BIS 681 (Statistical Practice Capstone, Spring 2026) group effort. Thanks to teammates **Barrett Li, Dawn (Ziyan) Zhong, Joshua Lu, Qize Zhang, Yaoxin Xiao, and Yiran Huo** for their support and contributions throughout the project.
