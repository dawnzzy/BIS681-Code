# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: GenAI_single_cell
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 — Pipeline: Direct Gene-Space Flow Matching
#
# **Pipeline:** Gene Expression (2000d, standardized) → Flow Matching → inverse StandardScaler → Gene Expression
#
# This is the simplest pipeline: flow matching operates directly on the full 2000-dimensional
# standardized gene expression space, without any dimensionality reduction.
#
# The network needs to be wider/deeper to handle the higher dimensionality,
# and training takes longer, but no information is lost through compression.

# %%
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import torch

import config as cfg
from src.preprocess import set_seed, load_processed
from src.models_flow import train_flow_matching
from src.metrics import distribution_metrics, plot_umap_overlay, plot_training_curves

set_seed()
print(f"Device: {cfg.DEVICE}")

# %% [markdown]
# ## 1. Load Preprocessed Data

# %%
data = load_processed()
X_all = data["X_all"]
X_train_s = data["X_train_s"]
X_all_s = data["X_all_s"]
scaler = data["scaler"]
hvg_names = data["hvg_names"]

print(f"Train: {X_train_s.shape}")
print(f"Input dimension for flow matching: {X_train_s.shape[1]}")

# %% [markdown]
# ## 2. Train Flow Matching in Gene Space
#
# Using a wider network (1024 hidden) and more layers (6) to handle 2000 dimensions.
# Training for 800 epochs.

# %%
dim_gene = X_train_s.shape[1]
print(f"Training flow matching in {dim_gene}-dim gene space...")
print(f"Network: hidden={cfg.FM_GENE_HIDDEN}, layers={cfg.FM_GENE_LAYERS}, epochs={cfg.FM_GENE_EPOCHS}")

set_seed()
flow_gene, loss_gene = train_flow_matching(
    Z_train=X_train_s,
    dim=dim_gene,
    device=cfg.DEVICE,
    hidden=cfg.FM_GENE_HIDDEN,
    n_layers=cfg.FM_GENE_LAYERS,
    lr=cfg.FM_GENE_LR,
    batch_size=cfg.FM_GENE_BATCH_SIZE,
    n_epochs=cfg.FM_GENE_EPOCHS,
    print_every=100,
)

plot_training_curves(loss_gene, title="Flow Matching Loss (Gene Space)")
plt.show()

# %% [markdown]
# ## 3. Generate Synthetic Cells
#
# Sample from noise → integrate through learned flow → inverse StandardScaler

# %%
n_gen = len(X_all)
print(f"Generating {n_gen} synthetic cells...")

set_seed()
X_gen_gene_s = flow_gene.sample(n_gen, dim_gene, cfg.DEVICE, n_steps=cfg.FM_SAMPLE_STEPS)

# Map back: inverse StandardScaler
X_gen_gene = scaler.inverse_transform(X_gen_gene_s).astype(np.float32)

print(f"Generated shape: {X_gen_gene.shape}")
print(f"Value range: [{X_gen_gene.min():.2f}, {X_gen_gene.max():.2f}]")

# %% [markdown]
# ## 4. Quick Evaluation

# %%
metrics = distribution_metrics(X_all, X_gen_gene)
print(f"Gene Mean MSE: {metrics['Mean MSE']:.6f}")
print(f"Gene Std MSE:  {metrics['Std MSE']:.6f}")

plot_umap_overlay(X_all, X_gen_gene, title="Gene-Space Flow Matching: Real vs Generated")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Save Results

# %%
os.makedirs(cfg.MODEL_DIR, exist_ok=True)
torch.save(flow_gene.state_dict(), os.path.join(cfg.MODEL_DIR, "flow_gene.pt"))
np.save(os.path.join(cfg.DATA_DIR, "X_gen_gene.npy"), X_gen_gene)

print("Saved: flow_gene.pt, X_gen_gene.npy")
