"""
STEP 1 - Feature-activation t-SNE heatmap

Overlays one eGeMAPS feature on a speaker's embedding t-SNE: a k-NN heatmap of
the feature value over the plane plus the emotion scatter, to see which regions
the feature lights up.

Output: 1_classical_features/output/<speaker>/heatmap_<feature>.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# Make the shared package importable when launched from the repository root.
import sys
import pathlib

_ROOT = next(
    p for p in pathlib.Path(__file__).resolve().parents
    if (p / "common" / "__init__.py").exists()
)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)  # resolve every relative path from the project root

from common.paths import output_dir

OUT = output_dir(__file__)

# ============================== CONFIGURATION ==============================
GT_CSV = "data/ESD/ESD_GT.csv"
EGEMAPS_CSV = "data/ESD/ESD_eGeMAPS.csv"        # feature cache from egemaps_feature_extraction.py
NPZ_TEMPLATE = "data/ESD/wav2vec2-lg-xlsr-en-speech-emotion-recognition/{speaker}_embeddings.npz"
# ==========================================================================


def tsne_scatter_plus_knn_heatmap(
    emb_npz_path: str,
    spk_df: pd.DataFrame,
    speaker: str = "0017",
    out_path: str = "data/ESD/plots/0017_tsne_scatter_plus_knn_heatmap.png",
    seed: int = 42,
    perplexity: int = 30,
    n_iter: int = 1000,
    tsne_metric: str = "euclidean",

    k: int = 5,                 # KNN locality (as requested)
    grid_size: int = 300,       # 250-450 good range
    empty_value: float = 0.0,   # value for empty regions
    radius_quantile: float = 0.90,  # threshold radius derived from data density
    agg: str = "mean",          # "mean" or "median"
    FEATURE_COL = None,
):
    # ---------- Load embeddings ----------
    npz = np.load(emb_npz_path, allow_pickle=True)
    embs = npz["embs"]
    labels = npz["labels"]
    paths = np.array(npz["paths"]).squeeze()
    filenames = np.array([os.path.basename(str(p)) for p in paths])

    # ---------- Join feature by filename ----------
    spk_df = spk_df.copy()
    spk_df["filename"] = spk_df["filename"].astype(str)
    spk_df["__base__"] = spk_df["filename"].apply(lambda x: os.path.basename(x))

    if FEATURE_COL not in spk_df.columns:
        raise KeyError(f"'{FEATURE_COL}' not found in spk_df columns.")

    feat_map = spk_df.set_index("__base__")[FEATURE_COL]
    feat = pd.Series(filenames).map(feat_map).to_numpy(dtype=float)

    # remove rows where feature missing
    ok = ~np.isnan(feat)
    if ok.sum() < len(feat):
        print(f"[WARN] Missing feature values due to filename mismatch: {len(feat) - ok.sum()} / {len(feat)}")

    embs = embs[ok]
    labels = labels[ok]
    feat = feat[ok]

    # ---------- Standardize embeddings ----------
    X = StandardScaler().fit_transform(embs)

    # ---------- t-SNE ----------
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=n_iter,  # 'n_iter' renamed to 'max_iter' in scikit-learn >=1.5
        metric=tsne_metric,
        verbose=0,
    )
    XY = tsne.fit_transform(X)
    xs, ys = XY[:, 0], XY[:, 1]

    # ---------- Label encoding (for legend) ----------
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    label_names = le.classes_

    # ---------- Determine plotting bounds ----------
    pad_x = 0.05 * (xs.max() - xs.min() + 1e-9)
    pad_y = 0.05 * (ys.max() - ys.min() + 1e-9)
    x_min, x_max = xs.min() - pad_x, xs.max() + pad_x
    y_min, y_max = ys.min() - pad_y, ys.max() + pad_y

    # ---------- Build grid over TSNE plane ----------
    gx = np.linspace(x_min, x_max, grid_size)
    gy = np.linspace(y_min, y_max, grid_size)
    Xg, Yg = np.meshgrid(gx, gy)
    grid_points = np.column_stack([Xg.ravel(), Yg.ravel()])  # (G,2)

    # ---------- KNN in TSNE space (points) ----------
    # We will query nearest neighbors for each grid cell
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(XY)

    dists, idxs = nn.kneighbors(grid_points, return_distance=True)  # (G,k)

    # ---------- Define "empty space" threshold radius ----------
    # Use distribution of real-point nearest-neighbor distances to estimate typical density.
    nn1 = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(XY)
    d12, _ = nn1.kneighbors(XY, return_distance=True)
    # d12[:,0]=0 self, d12[:,1]=nearest other point
    typical_nn_dist = d12[:, 1]
    radius = np.quantile(typical_nn_dist, radius_quantile)

    # Any grid cell whose nearest real point is farther than radius -> empty_value
    nearest_dist = dists[:, 0]
    is_empty = nearest_dist > radius

    # ---------- Aggregate feature over KNN neighborhood ----------
    neigh_feat = feat[idxs]  # (G,k)

    if agg == "mean":
        vals = np.mean(neigh_feat, axis=1)
    elif agg == "median":
        vals = np.median(neigh_feat, axis=1)
    else:
        raise ValueError("agg must be 'mean' or 'median'.")

    vals[is_empty] = empty_value

    heat = vals.reshape(grid_size, grid_size)

    # ---------- Plot: top heatmap, bottom scatter ----------
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)

    # TOP: KNN heatmap (empty zones = 0)
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(
        heat,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
        interpolation="nearest",
    )
    ax0.set_title(
        f"t-SNE KNN heatmap (k={k} in t-SNE) — Speaker {speaker}\n"
        f"{FEATURE_COL}",
        fontsize=11,
        fontweight="bold",
    )
    ax0.set_xticks([])
    ax0.set_yticks([])
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label(FEATURE_COL)

    # BOTTOM: scatter TSNE
    ax1 = fig.add_subplot(gs[1, 0])
    for lab in label_names:
        m = labels == lab
        ax1.scatter(xs[m], ys[m], s=18, alpha=0.65, label=str(lab), edgecolors="none")

    ax1.set_title(f"t-SNE scatter — Speaker {speaker}", fontsize=11, fontweight="bold")
    ax1.legend(title="Emotion", fontsize=8, title_fontsize=9, loc="best")
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # same extents (so it's literally the projection)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_path)
    return {
        "XY": XY,
        "heatmap": heat,
        "radius": radius,
        "out_path": out_path,
        "num_empty_cells": int(is_empty.sum()),
        "empty_fraction": float(is_empty.mean()),
    }


# ---------------- Example usage ----------------
esd_gt = pd.read_csv(GT_CSV, sep=";")
esd_feat = pd.read_csv(EGEMAPS_CSV, sep=";")
df = esd_gt.merge(esd_feat, on="filename")

speakers = [str(s).zfill(4) for s in range(1, 21) if int(s) != 17]
for speaker in speakers:

    spk_df = df[df["speaker"] == int(speaker)]

    features = df.columns[6:]

    for feature in features:
        res = tsne_scatter_plus_knn_heatmap(
            emb_npz_path=NPZ_TEMPLATE.format(speaker=speaker),
            spk_df=spk_df,
            speaker=speaker,
            out_path=str(OUT / speaker / f"heatmap_{feature}.png"),
            perplexity=30,
            seed=42,
            n_iter=1000,
            k=5,
            grid_size=320,
            empty_value=0.0,
            radius_quantile=0.90,  # increase (0.95) => fewer empty areas; decrease (0.85) => more zeros
            agg="median",
            FEATURE_COL=feature
        )

        print("empty_fraction:", res["empty_fraction"])
