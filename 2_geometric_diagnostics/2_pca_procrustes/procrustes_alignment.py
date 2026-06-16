"""
STEP 2.3 - Procrustes alignment + PCA before/after

Aligns the RAVDESS emotion directions to the ESD ones with an orthogonal
Procrustes rotation R, reports per-class cosine before vs after, evaluates
cross-dataset transfer (no adaptation / Procrustes / oracle ESD-CV), and draws a
PCA arrow plot of the directions before and after alignment.

Output: 2_geometric_diagnostics/2_pca_procrustes/output/ ->
pca_directions_before_after.png, procrustes_results.json
"""

import os
import sys
import pathlib

_ROOT = next(
    p for p in pathlib.Path(__file__).resolve().parents
    if (p / "common" / "__init__.py").exists()
)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)  # resolve every relative path from the project root

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from common.io import (
    load_embeddings_filenames, load_esd_gt, load_ravdess_gt, merge_embeddings_with_gt,
)
from common.labels import normalize_label
from common.paths import output_dir

SEED = 42
PROBE_C = 1.0
PROBE_MAX_ITER = 5000
SHARED_EMOTIONS = sorted(["angry", "happy", "neutral", "sad", "surprised"])
EMOTION_COLORS = {
    "angry": "#E53935", "happy": "#FB8C00", "neutral": "#757575",
    "sad": "#1E88E5", "surprised": "#43A047",
}

RAV_NPZ = "data/RAVDESS/wav2vec2-lg-xlsr-en-speech-emotion-recognition/{speaker}_embeddings.npz"
ESD_NPZ = "data/ESD/wav2vec2-lg-xlsr-en-speech-emotion-recognition/{speaker}_embeddings.npz"
RAV_GT_CSV = "data/RAVDESS/RAVDESS_GT.csv"
ESD_GT_CSV = "data/ESD/ESD_GT.csv"
RAV_ACTORS = [str(s).zfill(4) for s in range(1, 25)]
ESD_SPEAKERS = [str(s).zfill(4) for s in range(1, 21)]
OUT = output_dir(__file__)


def _logreg():
    return LogisticRegression(multi_class="multinomial", solver="lbfgs",
                              C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=SEED)


def _load(npz_template, speakers, gt, meta_cols):
    """Load embeddings, join GT, normalize emotion, keep shared emotions."""
    X_raw, fnames, _ = load_embeddings_filenames(npz_template, speakers)
    X, meta = merge_embeddings_with_gt(X_raw, fnames, gt, meta_cols=meta_cols)
    meta = meta.copy()
    meta["emotion"] = meta["emotion"].map(normalize_label)
    keep = meta["emotion"].isin(SHARED_EMOTIONS).values
    return X[keep], meta[keep].reset_index(drop=True)


def _arrow_panel(ax, W_rav_2d, W_esd_2d, var, title):
    origin = np.zeros(2)
    for i, emo in enumerate(SHARED_EMOTIONS):
        c = EMOTION_COLORS[emo]
        for W2d, ls, lw in ((W_rav_2d, "-", 2.5), (W_esd_2d, "--", 2.0)):
            ax.annotate("", xy=tuple(W2d[i]), xytext=tuple(origin),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, linestyle=ls, mutation_scale=16))
    lim = float(np.max(np.abs(np.vstack([W_rav_2d, W_esd_2d]))) * 1.25) or 1.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#CCCCCC", lw=0.8, zorder=0)
    ax.axvline(0, color="#CCCCCC", lw=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title(title, fontsize=12, fontweight="bold")


def plot_before_after(W_rav_n, W_esd_n, W_rav_aligned, out_path):
    from matplotlib.lines import Line2D

    pca_b = PCA(n_components=2).fit(np.vstack([W_rav_n, W_esd_n]))
    pca_a = PCA(n_components=2).fit(np.vstack([W_rav_aligned, W_esd_n]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    _arrow_panel(ax1, pca_b.transform(W_rav_n), pca_b.transform(W_esd_n),
                 pca_b.explained_variance_ratio_, "Before alignment")
    _arrow_panel(ax2, pca_a.transform(W_rav_aligned), pca_a.transform(W_esd_n),
                 pca_a.explained_variance_ratio_, "After Procrustes alignment")

    legend = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-", label="RAVDESS"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="ESD"),
    ] + [Line2D([0], [0], color=EMOTION_COLORS[e], lw=2, marker="o", markersize=6,
                linestyle="None", label=e.capitalize()) for e in SHARED_EMOTIONS]
    ax1.legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle("Emotion-separating directions: RAVDESS vs ESD", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    le = LabelEncoder().fit(SHARED_EMOTIONS)

    # 1) Load embeddings + labels
    print("[1] Loading RAVDESS + ESD embeddings...")
    X_rav, meta_rav = _load(RAV_NPZ, RAV_ACTORS,
                            load_ravdess_gt(RAV_GT_CSV),
                            ["filename", "actor_id", "emotion"])
    X_esd, meta_esd = _load(ESD_NPZ, ESD_SPEAKERS,
                            load_esd_gt(ESD_GT_CSV),
                            ["filename", "speaker", "emotion"])
    y_rav = le.transform(meta_rav["emotion"].values)
    y_esd = le.transform(meta_esd["emotion"].values)
    speakers_esd = meta_esd["speaker"].values
    print(f"    RAVDESS: {X_rav.shape}, ESD: {X_esd.shape}, dim={X_rav.shape[1]}")

    # 2) Standardize (per-dataset for probes; common for cross-dataset eval)
    X_rav_std = StandardScaler().fit_transform(X_rav)
    X_esd_std = StandardScaler().fit_transform(X_esd)
    scaler_common = StandardScaler().fit(X_rav)
    X_rav_common = scaler_common.transform(X_rav).astype(np.float64)
    X_esd_common = scaler_common.transform(X_esd)

    # 3) Probes (weight matrices)
    print("[2] Training emotion probes...")
    W_rav = _logreg().fit(X_rav_std, y_rav).coef_
    W_esd = _logreg().fit(X_esd_std, y_esd).coef_

    # ESD speaker-independent CV (oracle)
    n_splits = min(10, len(np.unique(speakers_esd)))
    y_esd_cv = cross_val_predict(_logreg(), X_esd_std, y_esd,
                                 cv=GroupKFold(n_splits=n_splits), groups=speakers_esd)
    oracle = {
        "accuracy": float(accuracy_score(y_esd, y_esd_cv)),
        "f1_macro": float(f1_score(y_esd, y_esd_cv, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_esd, y_esd_cv, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_esd, y_esd_cv, average="macro", zero_division=0)),
    }

    # 4) Procrustes alignment of the direction matrices
    print("[3] Procrustes alignment...")
    W_rav_n = W_rav / (np.linalg.norm(W_rav, axis=1, keepdims=True) + 1e-12)
    W_esd_n = W_esd / (np.linalg.norm(W_esd, axis=1, keepdims=True) + 1e-12)
    cosines_before = np.array([float(np.dot(W_rav_n[i], W_esd_n[i])) for i in range(len(SHARED_EMOTIONS))])
    R, _ = orthogonal_procrustes(W_rav_n, W_esd_n)
    W_rav_aligned = W_rav_n @ R
    cosines_after = np.array([float(np.dot(W_rav_aligned[i], W_esd_n[i])) for i in range(len(SHARED_EMOTIONS))])
    disparity = float(np.linalg.norm(W_rav_aligned - W_esd_n, "fro"))
    for e, b, a in zip(SHARED_EMOTIONS, cosines_before, cosines_after):
        print(f"    {e:>10s}: cos {b:+.4f} -> {a:+.4f}")

    # 5) Cross-dataset evaluation
    print("[4] Cross-dataset evaluation...")

    def _eval(Xtr):
        clf = _logreg().fit(Xtr, y_rav)
        yp = clf.predict(X_esd_common)
        return {
            "accuracy": float(accuracy_score(y_esd, yp)),
            "f1_macro": float(f1_score(y_esd, yp, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_esd, yp, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_esd, yp, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_esd, yp).tolist(),
        }

    no_adapt = _eval(X_rav_common)
    procrustes = _eval(X_rav_common @ R)

    print(f"    no-adapt acc={no_adapt['accuracy']:.4f} | procrustes acc={procrustes['accuracy']:.4f} | oracle acc={oracle['accuracy']:.4f}")

    # 6) PCA before/after plot
    print("[5] PCA before/after plot...")
    plot_before_after(W_rav_n, W_esd_n, W_rav_aligned, OUT / "pca_directions_before_after.png")

    # Save report
    report = {
        "shared_emotions": SHARED_EMOTIONS,
        "embedding_dim": int(X_rav.shape[1]),
        "ravdess_n_samples": int(len(meta_rav)),
        "esd_n_samples": int(len(meta_esd)),
        "procrustes_disparity": disparity,
        "cosines_before": {e: float(c) for e, c in zip(SHARED_EMOTIONS, cosines_before)},
        "cosines_after": {e: float(c) for e, c in zip(SHARED_EMOTIONS, cosines_after)},
        "results": {"no_adaptation": no_adapt, "procrustes": procrustes, "oracle_esd_cv": oracle},
    }
    with open(OUT / "procrustes_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {OUT / 'procrustes_results.json'}")


if __name__ == "__main__":
    main()
