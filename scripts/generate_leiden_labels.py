#!/usr/bin/env python
"""Generate Leiden cluster labels for the preprocessed PBMC 3K dataset.

This script runs Leiden clustering on the same preprocessed data from
notebook 01 and saves the labels to artifacts/data/leiden_labels.npy.

These labels are used by:
- notebook 06 (conditional flow matching) for cluster-conditioned generation
- the scGAN project (cscGAN training) for conditional GAN generation

The preprocessing steps here (QC, normalization, HVG selection, PCA,
neighbors) mirror notebook 01 exactly. The only addition is the
Leiden clustering step at the end.

Usage:
    conda run -n GenAI_single_cell python scripts/generate_leiden_labels.py
"""
import os
import sys
import numpy as np
import scanpy as sc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

LEIDEN_RESOLUTION = 1.0

print("Loading PBMC 3K dataset...")
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()

# QC — same filters as notebook 01
sc.pp.filter_cells(adata, min_genes=cfg.MIN_GENES)
sc.pp.filter_genes(adata, min_cells=cfg.MIN_CELLS)
adata = adata[adata.obs.n_genes < cfg.MAX_GENES]
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata = adata[adata.obs.pct_counts_mt < cfg.MAX_MT_PCT]
print(f"After QC: {adata.shape}")

# Normalize — same as notebook 01
sc.pp.normalize_total(adata, target_sum=cfg.NORMALIZE_TARGET)
sc.pp.log1p(adata)

# HVG selection — same as notebook 01
adata_raw = adata.copy()
sc.pp.highly_variable_genes(adata_raw, n_top_genes=cfg.N_HVG, flavor=cfg.HVG_FLAVOR)
adata = adata[:, adata_raw.var.highly_variable]
print(f"After HVG: {adata.shape}")

# Scale, PCA, neighbors, Leiden
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=cfg.PCA_VIS_COMPS)
sc.pp.neighbors(adata, n_pcs=40)
sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION)

labels = adata.obs["leiden"].astype(int).values
n_clusters = len(np.unique(labels))
print(f"Leiden clustering (resolution={LEIDEN_RESOLUTION}): {n_clusters} clusters")
for c in sorted(np.unique(labels)):
    print(f"  Cluster {c}: {(labels == c).sum()} cells ({(labels == c).mean() * 100:.1f}%)")

# Save
out_path = os.path.join(cfg.DATA_DIR, "leiden_labels.npy")
np.save(out_path, labels)
print(f"\nSaved: {out_path} (shape {labels.shape})")
