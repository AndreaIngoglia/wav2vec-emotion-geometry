"""
STEP 1 - Eta-squared (one-way ANOVA) feature ranking

Ranks eGeMAPS features by eta^2 = SS_between / SS_total of a one-way ANOVA over
the emotion classes (per-speaker z-normalized) and plots the top features as
overlaid densities.

Output: 1_classical_features/output/ -> eta_squared_results.csv, eta_squared*.png
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

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



def eta_squared_analysis(
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
    """
    Compute eta-squared (η²) from one-way ANOVA across all emotion classes.

    η² = SS_between / SS_total, i.e. the proportion of total variance in a
    feature that is explained by emotion class.  Features are ranked by η²
    (descending) and the top_n are plotted as overlaid density histograms
    with per-emotion means.

    Parameters
    ----------
    df : pd.DataFrame
        Merged ground truth + eGeMAPS dataframe (joined on 'filename').
    emotions : list[str]
        Emotion labels to include (at least 2).
    speakers, genders, languages : list | None
        Filters. When speakers is set, analysis is done per speaker.
    emotion_colors : dict[str, str] | None
        Mapping emotion -> hex color.
    explicit_features : list[str] | None
        If set, restrict analysis to these features only.
    normalize_per_speaker : bool
        Z-score features per speaker before analysis (default True).
    top_n : int
        Number of top features to display.
    metadata_cols : list[str] | None
        Columns to exclude from features.
    save_dir : str | None
        If set, save figures instead of showing them.

    Returns
    -------
    pd.DataFrame
        All features with η², F-statistic, and p-value, sorted by η² desc.
    """
    if len(emotions) < 2:
        raise ValueError("Need at least 2 emotions.")

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
        spk_stds = spk_stds.replace(0, np.nan)
        filtered[features] = (filtered[features] - spk_means) / spk_stds
        print("Applied per-speaker z-score normalization")

    # --- determine speaker groups ---
    if speakers:
        speaker_groups = [(s, filtered[filtered['speaker'] == s]) for s in speakers]
    else:
        speaker_groups = [(None, filtered)]

    print(f"Samples: {len(filtered)} | Features: {len(features)} | "
          f"Emotions: {emotions} | Speaker groups: {len(speaker_groups)}")

    # --- compute η² for every speaker x feature ---
    records = []
    for speaker, spk_data in speaker_groups:
        for feat in features:
            # Build groups: one array of values per emotion
            groups = []
            for em in emotions:
                vals = spk_data.loc[spk_data['emotion'] == em, feat].dropna()
                if len(vals) < 2:
                    break
                groups.append(vals.values)

            if len(groups) != len(emotions):
                continue

            # One-way ANOVA via scipy
            f_stat, p_value = f_oneway(*groups)

            # Compute η² = SS_between / SS_total
            all_vals = np.concatenate(groups)
            grand_mean = np.mean(all_vals)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

            records.append({
                'speaker': speaker,
                'feature': feat,
                'eta_squared': eta_sq,
                'f_stat': f_stat,
                'p_value': p_value,
                'n_total': len(all_vals),
            })

    results = (pd.DataFrame(records)
               .sort_values('eta_squared', ascending=False)
               .reset_index(drop=True))

    # --- plot per speaker (top_n features, descending η²) ---
    for speaker, spk_data in speaker_groups:
        if speaker is not None:
            top = results.query("speaker == @speaker").head(top_n)
        else:
            top = results.head(top_n)

        if top.empty:
            continue

        n_plots = len(top)
        ncols = 2
        nrows = (n_plots + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))

        spk_label = f" — Speaker {speaker}" if speaker is not None else ""
        norm_label = " (z-normed)" if normalize_per_speaker else ""
        em_str = ", ".join(emotions)
        fig.suptitle(f"Top features by η² [{em_str}]{spk_label}{norm_label}",
                     fontsize=14, fontweight='bold')
        axes = np.array(axes).flatten()

        for idx, (_, row) in enumerate(top.iterrows()):
            ax = axes[idx]

            for em in emotions:
                vals = spk_data.loc[spk_data['emotion'] == em, row['feature']].dropna()
                c = emotion_colors[em]
                ax.hist(vals, bins=30, alpha=0.5, label=em, color=c, density=True)
                ax.axvline(np.mean(vals), color=c, ls='--', lw=2)

            ax.set_title(
                f"{idx+1}. {row['feature']}\n"
                f"η² = {row['eta_squared']:.3f}   F = {row['f_stat']:.1f}   "
                f"p = {row['p_value']:.2e}",
                fontsize=9, fontweight='bold')
            ax.set_xlabel('Value (z-score)' if normalize_per_speaker else 'Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        if save_dir:
            fname = "eta_squared"
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

results = eta_squared_analysis(
    df,
    emotions=['Happy', 'Angry', 'Surprise', 'Sad', 'Neutral'],
    emotion_colors=emotion_colors,
    normalize_per_speaker=True,
    top_n=10,
    save_dir=str(OUT),
)
print(results.to_string())
results.to_csv(OUT / 'eta_squared_results.csv', index=False, sep=';')