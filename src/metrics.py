"""Evaluation metrics and visualization utilities."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ── Distribution Metrics ─────────────────────────────────────────────────────

def distribution_metrics(X_real: np.ndarray, X_gen: np.ndarray) -> dict:
    """Compute gene-wise mean MSE and std MSE between real and generated data."""
    mean_mse = float(np.mean((X_real.mean(0) - X_gen.mean(0)) ** 2))
    std_mse = float(np.mean((X_real.std(0) - X_gen.std(0)) ** 2))
    return {"Mean MSE": mean_mse, "Std MSE": std_mse}


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    n_subsample: int = cfg.MMD_SUBSAMPLE,
    n_pca: int = cfg.MMD_PCA_DIMS,
    seed: int = cfg.SEED,
) -> float:
    """Compute MMD with multi-scale Gaussian kernel in PCA-reduced space."""
    rng = np.random.default_rng(seed)

    # Subsample for efficiency
    if len(X) > n_subsample:
        X = X[rng.choice(len(X), n_subsample, replace=False)]
    if len(Y) > n_subsample:
        Y = Y[rng.choice(len(Y), n_subsample, replace=False)]

    # PCA reduction
    n_comps = min(n_pca, min(X.shape[1], n_subsample))
    pca = PCA(n_components=n_comps, random_state=seed)
    XY = np.vstack([X, Y])
    XY_pca = pca.fit_transform(XY)
    X_p, Y_p = XY_pca[: len(X)], XY_pca[len(X) :]

    # Adaptive bandwidth from k-nearest-neighbor distances
    dists = pairwise_distances(X_p)
    np.fill_diagonal(dists, np.inf)
    knn_dists = np.sort(dists, axis=1)[:, :25].mean(axis=1)
    median_dist = float(np.median(knn_dists))
    sigmas = [median_dist / 2, median_dist, median_dist * 2]

    # Multi-scale Gaussian kernel MMD
    mmd = 0.0
    for sigma in sigmas:
        gamma = -1.0 / (sigma ** 2)
        Kxx = np.exp(gamma * pairwise_distances(X_p, metric="sqeuclidean")).mean()
        Kyy = np.exp(gamma * pairwise_distances(Y_p, metric="sqeuclidean")).mean()
        Kxy = np.exp(gamma * pairwise_distances(X_p, Y_p, metric="sqeuclidean")).mean()
        mmd += Kxx + Kyy - 2 * Kxy

    return float(mmd)


def per_gene_wasserstein(X_real: np.ndarray, X_gen: np.ndarray) -> float:
    """Average per-gene 1D Wasserstein-1 distance (Earth Mover's Distance).

    For each of the ``n_genes`` columns, compute the 1D Wasserstein distance
    between the real and generated empirical distributions, then average across
    genes. Standard benchmark metric for scRNA-seq generative models (lower is
    better).
    """
    from scipy.stats import wasserstein_distance

    assert X_real.shape[1] == X_gen.shape[1], "gene dimension mismatch"
    n_genes = X_real.shape[1]
    scores = np.empty(n_genes, dtype=np.float64)
    for g in range(n_genes):
        scores[g] = wasserstein_distance(X_real[:, g], X_gen[:, g])
    return float(scores.mean())


def per_gene_kl_divergence(
    X_real: np.ndarray,
    X_gen: np.ndarray,
    bins: int = 100,
    eps: float = 1e-10,
) -> float:
    """Average per-gene KL divergence between real and generated histograms.

    For each gene, both real and generated samples are binned into ``bins``
    buckets over a **shared** range ``[min(p, q), max(p, q)]`` so that the two
    histograms live on the same support. Otherwise the KL numbers would be
    nonsense — each histogram would cover a different interval.

    Lower is better (0 = identical distributions).
    """
    from scipy.stats import entropy

    assert X_real.shape[1] == X_gen.shape[1], "gene dimension mismatch"
    n_genes = X_real.shape[1]
    scores = np.empty(n_genes, dtype=np.float64)
    for g in range(n_genes):
        p = X_real[:, g]
        q = X_gen[:, g]
        lo = float(min(p.min(), q.min()))
        hi = float(max(p.max(), q.max()))
        if hi <= lo:  # degenerate gene (all zeros in both); KL = 0
            scores[g] = 0.0
            continue
        p_hist, _ = np.histogram(p, bins=bins, range=(lo, hi), density=True)
        q_hist, _ = np.histogram(q, bins=bins, range=(lo, hi), density=True)
        scores[g] = float(entropy(p_hist + eps, q_hist + eps))
    return float(scores.mean())


def assign_labels_by_nn(
    X_gen: np.ndarray,
    X_real: np.ndarray,
    real_labels: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Assign cluster labels to unlabeled generated cells via k-NN on real data.

    Used for unconditional generators (FM-Gene / FM-PCA / FM-AE) so that the
    per-cluster DEG analysis can still be run on their outputs.
    """
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_real, real_labels)
    return knn.predict(X_gen)


def common_degs(
    X_real: np.ndarray,
    real_labels: np.ndarray,
    X_gen: np.ndarray,
    gen_labels: np.ndarray | None = None,
    hvg_names: list[str] | None = None,
    top_n: int = 100,
    method: str = "wilcoxon",
) -> dict:
    """Average overlap of top-N marker genes between real and generated data.

    For each cluster present in ``real_labels``, run
    :func:`scanpy.tl.rank_genes_groups` on both real and generated matrices,
    take the top ``top_n`` marker genes per cluster, and compute the Jaccard-
    style overlap ratio ``|intersection| / top_n``. Returns both the
    per-cluster ratios and the mean across clusters (higher is better).

    If ``gen_labels`` is ``None``, labels are inferred for generated cells via
    a k-NN classifier fit on ``(X_real, real_labels)`` so this metric still
    works for unconditional generators.
    """
    import scanpy as sc
    import anndata as ad

    if gen_labels is None:
        gen_labels = assign_labels_by_nn(X_gen, X_real, real_labels)

    assert len(real_labels) == len(X_real), "real label/data length mismatch"
    assert len(gen_labels) == len(X_gen), "gen label/data length mismatch"

    # var_names matter for overlap; if not given, use positional indices
    if hvg_names is None:
        hvg_names = [f"g{i}" for i in range(X_real.shape[1])]

    def _top_markers(X: np.ndarray, labels: np.ndarray) -> dict:
        adata = ad.AnnData(X=np.asarray(X, dtype=np.float32))
        adata.var_names = list(hvg_names)
        adata.obs["cell_type"] = pd.Categorical(labels.astype(str))
        sc.tl.rank_genes_groups(adata, "cell_type", method=method, n_genes=top_n)
        names = adata.uns["rank_genes_groups"]["names"]
        return {cl: set(names[cl][:top_n]) for cl in names.dtype.names}

    real_markers = _top_markers(X_real, real_labels)
    gen_markers = _top_markers(X_gen, gen_labels)

    per_cluster = {}
    for cl in sorted(real_markers.keys()):
        if cl not in gen_markers:  # cluster missing in generated (e.g. cluster 8 rare)
            per_cluster[cl] = 0.0
            continue
        common = real_markers[cl] & gen_markers[cl]
        per_cluster[cl] = float(len(common) / top_n)

    return {
        "per_cluster": per_cluster,
        "mean": float(np.mean(list(per_cluster.values()))),
    }


def memorization_check(
    X_gen: np.ndarray,
    X_train: np.ndarray,
    n_subsample: int = 500,
    seed: int = cfg.SEED,
) -> dict:
    """Check if generated samples are memorizing training data.

    Compares nearest-neighbor distances:
    - gen→train: for each generated sample, distance to nearest training sample
    - train→train: for each training sample, distance to nearest other training sample

    If gen→train distances are much smaller than train→train, the model may be memorizing.
    If they are similar or larger, the model is generating novel samples.
    """
    rng = np.random.default_rng(seed)

    if len(X_gen) > n_subsample:
        X_gen = X_gen[rng.choice(len(X_gen), n_subsample, replace=False)]
    if len(X_train) > n_subsample:
        X_train = X_train[rng.choice(len(X_train), n_subsample, replace=False)]

    # gen→train NN distances
    d_gen_train = pairwise_distances(X_gen, X_train)
    nn_gen_train = d_gen_train.min(axis=1)

    # train→train NN distances (exclude self)
    d_train_train = pairwise_distances(X_train)
    np.fill_diagonal(d_train_train, np.inf)
    nn_train_train = d_train_train.min(axis=1)

    return {
        "gen_to_train_nn_mean": float(nn_gen_train.mean()),
        "gen_to_train_nn_median": float(np.median(nn_gen_train)),
        "train_to_train_nn_mean": float(nn_train_train.mean()),
        "train_to_train_nn_median": float(np.median(nn_train_train)),
        "ratio_mean": float(nn_gen_train.mean() / nn_train_train.mean()),
    }


# ── Visualization ────────────────────────────────────────────────────────────

def plot_umap_overlay(
    X_real: np.ndarray,
    X_gen: np.ndarray,
    title: str = "Real vs Generated",
    n_vis: int = 1000,
    ax: plt.Axes | None = None,
    seed: int = cfg.SEED,
) -> plt.Axes:
    """UMAP scatter plot overlaying real and generated data."""
    import umap

    rng = np.random.default_rng(seed)
    r_idx = rng.choice(len(X_real), min(n_vis, len(X_real)), replace=False)
    g_idx = rng.choice(len(X_gen), min(n_vis, len(X_gen)), replace=False)

    combined = np.vstack([X_real[r_idx], X_gen[g_idx]])
    # PCA first for stability
    pca = PCA(n_components=min(50, combined.shape[1]), random_state=seed)
    combined_pca = pca.fit_transform(combined)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed, n_jobs=1)
    emb = reducer.fit_transform(combined_pca)

    n_r = len(r_idx)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(emb[:n_r, 0], emb[:n_r, 1], s=5, alpha=0.4, c="#4C72B0", label="Real")
    ax.scatter(emb[n_r:, 0], emb[n_r:, 1], s=5, alpha=0.4, c="#DD8452", label="Generated")
    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    return ax


def plot_marker_correlation(
    datasets: dict[str, np.ndarray],
    hvg_names: list[str],
    markers: list[str] | None = None,
) -> plt.Figure:
    """Plot marker gene correlation heatmaps for multiple datasets side-by-side."""
    if markers is None:
        markers = cfg.MARKER_GENES
    marker_in_hvg = [g for g in markers if g in hvg_names]
    col_idx = [hvg_names.index(g) for g in marker_in_hvg]

    if not col_idx:
        print("No marker genes found in HVG list.")
        return plt.figure()

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    im = None
    for ax, (name, data) in zip(axes, datasets.items()):
        subset = data[:, col_idx]
        corr = np.corrcoef(subset.T)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(marker_in_hvg)))
        ax.set_yticks(range(len(marker_in_hvg)))
        ax.set_xticklabels(marker_in_hvg, rotation=90, fontsize=8)
        ax.set_yticklabels(marker_in_hvg, fontsize=8)
        ax.set_title(name)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r")
    fig.suptitle("Marker Gene Correlation", fontsize=13)
    fig.tight_layout()
    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    best_epoch: int | None = None,
    title: str = "Training Curves",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot training (and optionally validation) loss curves."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train")
    if val_losses is not None:
        ax.plot(val_losses, label="Val")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best epoch={best_epoch+1}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    return ax
