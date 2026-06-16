"""
common.viz - the shared t-SNE + k-NN evaluation used across 3_local_evaluation:
standardize -> (speaker-grouped) k-NN CV -> t-SNE -> colored scatter.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


def run_tsne_knn(
    embs: np.ndarray,
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    knn_k: int = 5,
    cv_folds: int = 10,
    seed: int = 42,
    perplexity: int = 30,
    n_iter: int = 1000,
    group_cv: bool = True,
) -> dict:
    """Standardize embeddings, run a k-NN probe (CV accuracy) and a 2-D t-SNE."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    label_names = le.classes_

    X = StandardScaler().fit_transform(embs)

    knn = KNeighborsClassifier(n_neighbors=knn_k)
    if group_cv and groups is not None:
        n_splits = min(cv_folds, len(np.unique(groups)))
        if n_splits < 2:
            cv = StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=seed)
            scores = cross_val_score(knn, X, y, cv=cv, scoring="accuracy")
        else:
            cv = GroupKFold(n_splits=n_splits)
            scores = cross_val_score(knn, X, y, cv=cv, groups=groups, scoring="accuracy")
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(knn, X, y, cv=cv, scoring="accuracy")

    knn_mean, knn_std = float(scores.mean()), float(scores.std())
    print(f"k-NN accuracy: {knn_mean:.3f} ± {knn_std:.3f}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=n_iter,  # 'n_iter' was renamed to 'max_iter' in scikit-learn >=1.5
        verbose=0,
    )
    X_tsne = tsne.fit_transform(X)

    return {
        "X_tsne": X_tsne,
        "X": X,
        "y": y,
        "label_names": label_names,
        "knn_mean": knn_mean,
        "knn_std": knn_std,
        "n_samples": int(len(embs)),
        "n_dims": int(embs.shape[1]),
        "n_classes": int(len(label_names)),
    }


def scatter_by_label(
    X_tsne: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path,
    legend_title: str = "Class",
    figsize=(10, 8),
    dpi: int = 500,
    point_size: int = 14,
    alpha: float = 0.55,
) -> Path:
    """Scatter the 2-D t-SNE embedding, one color per label. Saves to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    for lab in np.unique(labels):
        m = labels == lab
        ax.scatter(
            X_tsne[m, 0], X_tsne[m, 1],
            s=point_size, alpha=alpha, label=str(lab), edgecolors="none",
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(title=legend_title, fontsize=8, title_fontsize=9, loc="best")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
