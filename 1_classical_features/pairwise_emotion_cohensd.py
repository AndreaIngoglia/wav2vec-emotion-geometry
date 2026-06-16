"""
STEP 1 - Pairwise Cohen's d between emotions

For each emotion pair and each eGeMAPS feature, computes Cohen's d (pooled
standardized mean difference, per-speaker z-normalized) and plots the features
that most separate the pair.

Output: 1_classical_features/output/<emoA>_vs_<emoB>*.png
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Make the shared package importable when launched from the repository root.
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

from common.paths import output_dir

OUT = output_dir(__file__)

# ============================== CONFIGURATION ==============================
GT_CSV = "data/ESD/ESD_GT.csv"
EGEMAPS_CSV = "data/ESD/ESD_eGeMAPS.csv"        # feature cache from egemaps_feature_extraction.py
# ==========================================================================



def cohens_d_analysis(
    df: pd.DataFrame,
    emotions: list[str],
    speakers: list[str] | None = None,
    genders: list[str] | None = None,
    languages: list[str] | None = None,
    emotion_colors: dict[str, str] | None = None,
    explicit_features: list[str] | None = None,
    normalize_per_speaker: bool = True,
    top_n: int = 10,
    metadata_cols: list[str] | None = None,
    save_dir: str | None = None,
) -> pd.DataFrame:

    if len(emotions) < 2:
        raise ValueError("Need at least 2 emotions for pairwise comparison.")

    # --- defaults ---
    if metadata_cols is None:
        metadata_cols = ['emotion', 'speaker', 'gender', 'filename',
                         'full_path', 'language']
    if emotion_colors is None:
        palette = plt.cm.tab10.colors
        emotion_colors = {e: palette[i % len(palette)] for i, e in enumerate(emotions)}

    # --- filter ---
    mask = df['emotion'].isin(emotions)
    if speakers:
        mask &= df['speaker'].isin(speakers)
    if genders:
        mask &= df['gender'].isin(genders)
    if languages:
        mask &= df['language'].isin(languages)
    filtered = df.loc[mask].copy()

    # --- resolve features ---
    all_features = [c for c in filtered.columns if c not in metadata_cols]
    if explicit_features:
        missing = [f for f in explicit_features if f not in all_features]
        if missing:
            raise ValueError(f"Features not found: {missing}")
        features = explicit_features
    else:
        features = all_features

    # --- per-speaker z-score normalization ---
    if normalize_per_speaker and 'speaker' in filtered.columns:
        spk_means = filtered.groupby('speaker')[features].transform('mean')
        spk_stds = filtered.groupby('speaker')[features].transform('std')
        # Avoid division by zero: keep original value where std == 0
        spk_stds = spk_stds.replace(0, np.nan)
        filtered[features] = (filtered[features] - spk_means) / spk_stds
        print("Applied per-speaker z-score normalization")

    # --- determine speaker groups ---
    # When speakers are specified, compute and plot per speaker separately
    if speakers:
        speaker_groups = [(s, filtered[filtered['speaker'] == s]) for s in speakers]
    else:
        speaker_groups = [(None, filtered)]

    print(f"Samples: {len(filtered)} | Features: {len(features)} | "
          f"Emotions: {emotions} | Speaker groups: {len(speaker_groups)}")

    # --- compute Cohen's d for every speaker x pair x feature ---
    records = []
    for speaker, spk_data in speaker_groups:
        for em_a, em_b in combinations(emotions, 2):
            data_a = spk_data.loc[spk_data['emotion'] == em_a, features]
            data_b = spk_data.loc[spk_data['emotion'] == em_b, features]

            for feat in features:
                va = data_a[feat].dropna()
                vb = data_b[feat].dropna()
                if len(va) < 2 or len(vb) < 2:
                    continue

                ma, mb = np.mean(va), np.mean(vb)
                na, nb = len(va), len(vb)
                dof = na + nb - 2
                pooled = np.sqrt(((na - 1) * np.std(va, ddof=1) ** 2 +
                                  (nb - 1) * np.std(vb, ddof=1) ** 2) / dof)
                d = (ma - mb) / pooled if pooled > 0 else 0.0

                records.append({
                    'speaker': speaker,
                    'emotion_a': em_a,
                    'emotion_b': em_b,
                    'feature': feat,
                    'cohens_d': d,
                    'abs_d': abs(d),
                    'mean_a': ma,
                    'mean_b': mb,
                })

    results = (pd.DataFrame(records)
               .sort_values('abs_d', ascending=False)
               .reset_index(drop=True))

    # --- plot per speaker x pair (top_n features, descending |d|) ---
    for speaker, spk_data in speaker_groups:
        for em_a, em_b in combinations(emotions, 2):
            q = "emotion_a == @em_a and emotion_b == @em_b"
            if speaker is not None:
                q += " and speaker == @speaker"
            pair = results.query(q).head(top_n)

            if pair.empty:
                continue

            n_plots = len(pair)
            ncols = 2
            nrows = (n_plots + 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))

            spk_label = f" — Speaker {speaker}" if speaker is not None else ""
            norm_label = " (z-normed)" if normalize_per_speaker else ""
            fig.suptitle(f"{em_a} vs {em_b}{spk_label}{norm_label}",
                         fontsize=16, fontweight='bold')
            axes = np.array(axes).flatten()

            for idx, (_, row) in enumerate(pair.iterrows()):
                ax = axes[idx]

                # Plot ALL emotions on the same subplot
                for em in emotions:
                    vals = spk_data.loc[spk_data['emotion'] == em, row['feature']].dropna()
                    c = emotion_colors[em]
                    ax.hist(vals, bins=30, alpha=0.5, label=em, color=c, density=True)
                    ax.axvline(np.mean(vals), color=c, ls='--', lw=2)

                ax.set_title(f"{row['feature']}\nd({em_a}/{em_b}) = {row['cohens_d']:.3f}", fontsize=10, fontweight='bold')
                #ax.set_title(f"{row['feature']}", fontsize=10, fontweight='bold')
                ax.set_xlabel('Value (z-score)' if normalize_per_speaker else 'Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(alpha=0.3)

            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            if save_dir:
                fname = f"{em_a}_vs_{em_b}"
                if speaker is not None:
                    fname += f"_speaker_{speaker}"
                fig.savefig(f"{save_dir}/{fname}.png", dpi=150,
                            bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

    return results


gt = pd.read_csv(GT_CSV, sep=';')
eg = pd.read_csv(EGEMAPS_CSV, sep=';')
df = pd.merge(gt, eg, on='filename', how='inner')

emotion_colors = {
    'Neutral':  '#808080',
    'Happy':    '#FFD700',
    'Sad':      '#4169E1',
    'Angry':    '#DC143C',
    'Surprise': '#FF8C00',
}

results = cohens_d_analysis(
    df,
    emotions=['Happy', 'Surprise'],
    emotion_colors=emotion_colors,
    normalize_per_speaker=True,
    explicit_features=['equivalentSoundLevel_dBp'],
    speakers=[4],
    top_n=1,
    save_dir=str(OUT),
)

results.to_csv('cohens_d_results.csv', index=False)