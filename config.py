"""Shared configuration for the scRNA-seq flow matching project."""

from __future__ import annotations

import os
import torch

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "artifacts", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "artifacts", "figures")

# ── Preprocessing ────────────────────────────────────────────────────────────
N_HVG = 2000
MIN_GENES = 200
MAX_GENES = 2500
MIN_CELLS = 3
MAX_MT_PCT = 5.0
NORMALIZE_TARGET = 1e4
HVG_FLAVOR = "seurat_v3"

# ── Data Split ───────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# test = 1 - train - val

# ── Dimensionality Reduction ─────────────────────────────────────────────────
LATENT_DIM = 20  # shared by PCA and AE for fair comparison
PCA_VIS_COMPS = 50  # for visualization / neighbors

# ── Autoencoder ──────────────────────────────────────────────────────────────
AE_HIDDEN_DIMS = [512, 128]  # encoder layers (decoder mirrors)
AE_DROPOUT = 0.3
AE_LR = 1e-3
AE_WEIGHT_DECAY = 1e-4
AE_BATCH_SIZE = 256
AE_MAX_EPOCHS = 500
AE_PATIENCE = 50
AE_SCHEDULER_FACTOR = 0.5
AE_SCHEDULER_PATIENCE = 15

# ── Flow Matching (gene space) ───────────────────────────────────────────────
FM_GENE_HIDDEN = 1024
FM_GENE_LAYERS = 6
FM_GENE_LR = 1e-3
FM_GENE_EPOCHS = 800
FM_GENE_BATCH_SIZE = 256

# ── Flow Matching (latent space — PCA / AE) ──────────────────────────────────
FM_LATENT_HIDDEN = 256
FM_LATENT_LAYERS = 4
FM_LATENT_LR = 1e-3
FM_LATENT_EPOCHS = 500
FM_LATENT_BATCH_SIZE = 256

# ── Conditional Flow Matching (PCA latent space + cluster label) ─────────────
CFM_HIDDEN = 256
CFM_LAYERS = 4
CFM_LR = 1e-3
CFM_EPOCHS = 500
CFM_BATCH_SIZE = 256

# ── Sampling ─────────────────────────────────────────────────────────────────
FM_SAMPLE_STEPS = 100

# ── Evaluation ───────────────────────────────────────────────────────────────
MMD_SUBSAMPLE = 500
MMD_PCA_DIMS = 50
MARKER_GENES = ["CST3", "NKG7", "MS4A1", "LYZ", "PPBP", "CD3D", "FCER1A", "FCGR3A", "CD79A"]
