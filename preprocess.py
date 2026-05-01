"""Data loading, QC, normalization, HVG selection, train/val/test split."""

from __future__ import annotations

import os
import random

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


def set_seed(seed: int = cfg.SEED) -> None:
    """Set random seeds for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_qc() -> sc.AnnData:
    """Load PBMC 3K, compute QC metrics, filter cells/genes, normalize, select HVGs.

    Returns
    -------
    adata_hvg : AnnData
        Filtered, normalized, log-transformed AnnData subsetted to HVGs.
        - ``.X`` contains log-normalized expression.
        - ``.layers["counts"]`` contains raw counts (pre-normalization).
        - ``.layers["log_norm"]`` contains log-normalized values (same as .X).
    """
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()

    # QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=cfg.MIN_GENES)
    sc.pp.filter_genes(adata, min_cells=cfg.MIN_CELLS)
    adata = adata[adata.obs["n_genes_by_counts"] < cfg.MAX_GENES, :].copy()
    adata = adata[adata.obs["pct_counts_mt"] < cfg.MAX_MT_PCT, :].copy()

    # Store raw counts before normalization
    adata.layers["counts"] = adata.X.copy()

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=cfg.NORMALIZE_TARGET)
    sc.pp.log1p(adata)
    adata.layers["log_norm"] = adata.X.copy()

    # HVG selection on raw counts
    sc.pp.highly_variable_genes(
        adata,
        flavor=cfg.HVG_FLAVOR,
        n_top_genes=cfg.N_HVG,
        layer="counts",
    )

    # Subset to HVGs
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    return adata_hvg


def extract_feature_matrix(adata_hvg: sc.AnnData) -> tuple[np.ndarray, list[str]]:
    """Extract dense float32 feature matrix from log-normalized HVG data.

    Returns (X, hvg_names) where X has shape (n_cells, n_hvg).
    """
    X = adata_hvg.layers["log_norm"]
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    hvg_names = adata_hvg.var_names.tolist()
    return X, hvg_names


def split_and_scale(
    X: np.ndarray,
    seed: int = cfg.SEED,
    train_frac: float = cfg.TRAIN_FRAC,
    val_frac: float = cfg.VAL_FRAC,
) -> dict:
    """80/10/10 train/val/test split + StandardScaler fit on train only.

    Returns a dict with keys:
        X_train, X_val, X_test (raw log-normalized)
        X_train_s, X_val_s, X_test_s, X_all_s (standardized)
        scaler, train_idx, val_idx, test_idx
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    X_all_s = scaler.transform(X).astype(np.float32)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_s": X_train_s,
        "X_val_s": X_val_s,
        "X_test_s": X_test_s,
        "X_all_s": X_all_s,
        "scaler": scaler,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def save_processed(
    X: np.ndarray,
    hvg_names: list[str],
    split: dict,
    data_dir: str = cfg.DATA_DIR,
) -> None:
    """Save processed arrays, scaler, and split indices to disk."""
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "X_all.npy"), X)
    np.save(os.path.join(data_dir, "X_train_s.npy"), split["X_train_s"])
    np.save(os.path.join(data_dir, "X_val_s.npy"), split["X_val_s"])
    np.save(os.path.join(data_dir, "X_test_s.npy"), split["X_test_s"])
    np.save(os.path.join(data_dir, "X_all_s.npy"), split["X_all_s"])
    np.save(os.path.join(data_dir, "train_idx.npy"), split["train_idx"])
    np.save(os.path.join(data_dir, "val_idx.npy"), split["val_idx"])
    np.save(os.path.join(data_dir, "test_idx.npy"), split["test_idx"])

    import json
    with open(os.path.join(data_dir, "hvg_names.json"), "w") as f:
        json.dump(hvg_names, f)

    import joblib
    joblib.dump(split["scaler"], os.path.join(data_dir, "scaler.joblib"))


def load_processed(data_dir: str = cfg.DATA_DIR) -> dict:
    """Load processed data from disk. Returns dict with all arrays and metadata."""
    import json
    import joblib

    data = {
        "X_all": np.load(os.path.join(data_dir, "X_all.npy")),
        "X_train_s": np.load(os.path.join(data_dir, "X_train_s.npy")),
        "X_val_s": np.load(os.path.join(data_dir, "X_val_s.npy")),
        "X_test_s": np.load(os.path.join(data_dir, "X_test_s.npy")),
        "X_all_s": np.load(os.path.join(data_dir, "X_all_s.npy")),
        "train_idx": np.load(os.path.join(data_dir, "train_idx.npy")),
        "val_idx": np.load(os.path.join(data_dir, "val_idx.npy")),
        "test_idx": np.load(os.path.join(data_dir, "test_idx.npy")),
        "scaler": joblib.load(os.path.join(data_dir, "scaler.joblib")),
    }
    with open(os.path.join(data_dir, "hvg_names.json")) as f:
        data["hvg_names"] = json.load(f)

    return data
