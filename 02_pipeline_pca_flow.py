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
# # 02 — Pipeline: PCA + Flow Matching
#
# **Pipeline:** Gene Expression → PCA (reduce to 20d) → Flow Matching → PCA inverse → Gene Expression
#
# This pipeline uses PCA as a linear dimensionality reduction step before applying flow matching.
# PCA captures the top principal components of variance, providing a compact representation.
# Flow matching then learns to generate new samples in this PCA space, which are mapped back to
# gene space via `pca.inverse_transform()`.

# %%
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
X_val_s = data["X_val_s"]
X_test_s = data["X_test_s"]
X_all_s = data["X_all_s"]
scaler = data["scaler"]
hvg_names = data["hvg_names"]

print(f"Train: {X_train_s.shape}, Val: {X_val_s.shape}, Test: {X_test_s.shape}")

# %% [markdown]
# ## 2. PCA Dimensionality Reduction
#
# PCA finds orthogonal directions of maximum variance in the data.
# We reduce from 2000 dimensions to 20, fit on training data only.

# %%
pca = PCA(n_components=cfg.LATENT_DIM, svd_solver="full", random_state=cfg.SEED)
Z_train_pca = pca.fit_transform(X_train_s).astype(np.float32)
Z_val_pca = pca.transform(X_val_s).astype(np.float32)
Z_test_pca = pca.transform(X_test_s).astype(np.float32)

# Reconstruction quality
X_recon = pca.inverse_transform(Z_test_pca)
pca_recon_mse = float(np.mean((X_test_s - X_recon) ** 2))

print(f"PCA latent shape: {Z_train_pca.shape}")
print(f"Cumulative variance explained: {pca.explained_variance_ratio_.sum():.4f}")
print(f"PCA reconstruction MSE (test, standardized): {pca_recon_mse:.4f}")

# %% [markdown]
# ## 3. Train Flow Matching in PCA Space

# %%
dim_pca = Z_train_pca.shape[1]
print(f"Training flow matching in {dim_pca}-dim PCA space...")

set_seed()
flow_pca, loss_pca = train_flow_matching(
    Z_train=Z_train_pca,
    dim=dim_pca,
    device=cfg.DEVICE,
    hidden=cfg.FM_LATENT_HIDDEN,
    n_layers=cfg.FM_LATENT_LAYERS,
    lr=cfg.FM_LATENT_LR,
    batch_size=cfg.FM_LATENT_BATCH_SIZE,
    n_epochs=cfg.FM_LATENT_EPOCHS,
    print_every=50,
)

plot_training_curves(loss_pca, title="Flow Matching Loss (PCA Space)")
plt.show()

# %% [markdown]
# ## 4. Generate Synthetic Cells
#
# Sample from noise → integrate through learned flow → PCA inverse transform → inverse StandardScaler

# %%
n_gen = len(X_all)
print(f"Generating {n_gen} synthetic cells...")

set_seed()
Z_gen_pca = flow_pca.sample(n_gen, dim_pca, cfg.DEVICE, n_steps=cfg.FM_SAMPLE_STEPS)

# Map back: PCA inverse → unstandardize
X_gen_pca_s = pca.inverse_transform(Z_gen_pca).astype(np.float32)
X_gen_pca = scaler.inverse_transform(X_gen_pca_s).astype(np.float32)

print(f"Generated shape: {X_gen_pca.shape}")
print(f"Value range: [{X_gen_pca.min():.2f}, {X_gen_pca.max():.2f}]")

# %% [markdown]
# ## 5. Quick Evaluation

# %%
metrics = distribution_metrics(X_all, X_gen_pca)
print(f"Gene Mean MSE: {metrics['Mean MSE']:.6f}")
print(f"Gene Std MSE:  {metrics['Std MSE']:.6f}")

plot_umap_overlay(X_all, X_gen_pca, title="PCA + Flow Matching: Real vs Generated")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Save Results

# %%
import torch, joblib

os.makedirs(cfg.MODEL_DIR, exist_ok=True)
torch.save(flow_pca.state_dict(), os.path.join(cfg.MODEL_DIR, "flow_pca.pt"))
joblib.dump(pca, os.path.join(cfg.MODEL_DIR, "pca.joblib"))
np.save(os.path.join(cfg.DATA_DIR, "X_gen_pca.npy"), X_gen_pca)

print("Saved: flow_pca.pt, pca.joblib, X_gen_pca.npy")
