"""
STEP 2.3 - PCA / cosine heatmap of emotion directions

PCA arrow plot of the RAVDESS vs ESD emotion unit directions, plus
cosine-similarity heatmaps (full and shared-classes only) between them.

Output: 2_geometric_diagnostics/2_pca_procrustes/output/ -> pca_arrow_plot.png,
cosine_heatmap_full.png, cosine_heatmap_shared.png
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

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from common.paths import output_dir
from common.probes import load_probe_dir

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 200,
})

OUT = output_dir(__file__)

# ============================== CONFIGURATION ==============================
# Probe caches produced by STEP 2.0 (train_probes.py):
RAVDESS_PROBE_DIR = "xai_weights/RAVDESS"
ESD_PROBE_DIR = "xai_weights/ESD"
# ==========================================================================


# ============================================================
# Plot 1: PCA arrow plot
# ============================================================
def plot_pca_arrows(W_rav, classes_rav, W_esd, classes_esd, shared_classes, output_path):
    vecs, labels, sources = [], [], []
    for cls in shared_classes:
        idx_r = np.where(classes_rav == cls)[0]
        idx_e = np.where(classes_esd == cls)[0]
        if len(idx_r) == 0 or len(idx_e) == 0:
            continue
        w_r = W_rav[idx_r[0]]; w_r = w_r / np.linalg.norm(w_r)
        w_e = W_esd[idx_e[0]]; w_e = w_e / np.linalg.norm(w_e)
        vecs.append(w_r); labels.append(cls); sources.append("RAVDESS")
        vecs.append(w_e); labels.append(cls); sources.append("ESD")

    vecs = np.array(vecs)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs)

    emotion_colors = {
        "angry": "#d62728", "happy": "#ff7f0e", "neutral": "#7f7f7f",
        "sad": "#1f77b4", "surprised": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (label, source) in enumerate(zip(labels, sources)):
        x, y = coords[i]
        color = emotion_colors.get(label, "#333333")
        linestyle = "-" if source == "RAVDESS" else "--"
        linewidth = 2.5 if source == "RAVDESS" else 2.0
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=linewidth,
                                    linestyle=linestyle, mutation_scale=18))
        offset = 0.02
        ha = "left" if x >= 0 else "right"
        va = "bottom" if y >= 0 else "top"
        suffix = " (R)" if source == "RAVDESS" else " (E)"
        ax.text(x + offset * np.sign(x), y + offset * np.sign(y),
                label.capitalize() + suffix, fontsize=9, color=color, ha=ha, va=va,
                fontweight="bold" if source == "RAVDESS" else "normal",
                fontstyle="normal" if source == "RAVDESS" else "italic")

    margin = np.max(np.abs(coords)) * 1.3
    ax.set_xlim(-margin, margin); ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.5, zorder=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Emotion-separating directions: RAVDESS vs ESD\n(unit vectors projected via PCA)")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-", label="RAVDESS"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="ESD"),
    ]
    for cls in shared_classes:
        c = emotion_colors.get(cls, "#333")
        legend_elements.append(Line2D([0], [0], color=c, lw=2, marker="o", markersize=6,
                                      linestyle="None", label=cls.capitalize()))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Plot 2: Cosine similarity heatmap
# ============================================================
def plot_cosine_heatmap(W_rav, classes_rav, W_esd, classes_esd, output_path,
                        rav_classes_to_show=None, esd_classes_to_show=None):
    if rav_classes_to_show is None:
        rav_classes_to_show = list(classes_rav)
    if esd_classes_to_show is None:
        esd_classes_to_show = list(classes_esd)

    n_rav, n_esd = len(rav_classes_to_show), len(esd_classes_to_show)
    sim_matrix = np.zeros((n_rav, n_esd))
    for i, cls_r in enumerate(rav_classes_to_show):
        idx_r = np.where(classes_rav == cls_r)[0]
        if len(idx_r) == 0:
            continue
        w_r = W_rav[idx_r[0]]
        for j, cls_e in enumerate(esd_classes_to_show):
            idx_e = np.where(classes_esd == cls_e)[0]
            if len(idx_e) == 0:
                continue
            w_e = W_esd[idx_e[0]]
            sim_matrix[i, j] = np.dot(w_r, w_e) / (np.linalg.norm(w_r) * np.linalg.norm(w_e) + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")

    ax.set_xticks(range(n_esd)); ax.set_xticklabels([c.capitalize() for c in esd_classes_to_show],
                                                    rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_rav)); ax.set_yticklabels([c.capitalize() for c in rav_classes_to_show], fontsize=10)
    ax.set_xlabel("ESD probe directions", fontsize=12)
    ax.set_ylabel("RAVDESS probe directions", fontsize=12)
    ax.set_title("Cosine similarity between emotion-separating directions", fontsize=12)

    for i in range(n_rav):
        for j in range(n_esd):
            val = sim_matrix[i, j]
            color = "white" if abs(val) > 0.08 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    for i, cls_r in enumerate(rav_classes_to_show):
        for j, cls_e in enumerate(esd_classes_to_show):
            if cls_r == cls_e:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2.5,
                                           edgecolor="gold", facecolor="none", zorder=10))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================
def main(ravdess_cache_dir=RAVDESS_PROBE_DIR, esd_cache_dir=ESD_PROBE_DIR, out_path=None):
    if out_path is None:
        out_path = str(OUT)
    os.makedirs(out_path, exist_ok=True)
    shared_classes = ["angry", "happy", "neutral", "sad", "surprised"]

    rav = load_probe_dir(ravdess_cache_dir, "emotion", normalize_classes=True)
    esd = load_probe_dir(esd_cache_dir, "emotion", normalize_classes=True)
    print(f"RAVDESS classes: {list(rav['classes'])} | ESD classes: {list(esd['classes'])}")

    plot_pca_arrows(rav["W"], rav["classes"], esd["W"], esd["classes"],
                    shared_classes, os.path.join(out_path, "pca_arrow_plot.png"))
    plot_cosine_heatmap(rav["W"], rav["classes"], esd["W"], esd["classes"],
                        os.path.join(out_path, "cosine_heatmap_full.png"))
    plot_cosine_heatmap(rav["W"], rav["classes"], esd["W"], esd["classes"],
                        os.path.join(out_path, "cosine_heatmap_shared.png"),
                        rav_classes_to_show=shared_classes, esd_classes_to_show=shared_classes)


if __name__ == "__main__":
    main()
