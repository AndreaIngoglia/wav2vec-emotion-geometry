"""
common.io - load per-speaker embedding .npz caches and dataset ground-truth
CSVs, and join them on filename.

.npz layout (written by common.embeddings): embs (N, D), labels (N,), paths (N,).
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def _iter_speaker_npz(npz_template: str, speakers: Sequence[str]):
    """Yield ``(speaker, np.load(...))`` for every existing per-speaker file."""
    found = False
    for spk in speakers:
        path = npz_template.format(speaker=spk)
        if not os.path.exists(path):
            print(f"[WARN] Missing: {path} (skipping)")
            continue
        found = True
        yield str(spk), np.load(path, allow_pickle=True)
    if not found:
        raise FileNotFoundError(
            "No embedding files were found. Check your paths/template: "
            f"{npz_template!r}"
        )


def load_embeddings_labels(
    npz_template: str, speakers: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load + concatenate embeddings and emotion labels -> (embs, labels, speaker_ids)."""
    all_embs, all_labels, all_spk = [], [], []
    for spk, data in _iter_speaker_npz(npz_template, speakers):
        embs = data["embs"]
        all_embs.append(embs)
        all_labels.append(data["labels"])
        all_spk.append(np.full(len(embs), spk, dtype=object))
    return (
        np.concatenate(all_embs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_spk, axis=0),
    )


def load_embeddings_only(
    npz_template: str, speakers: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Load + concatenate embeddings, ignoring labels -> (embs, speaker_ids)."""
    all_embs, all_spk = [], []
    for spk, data in _iter_speaker_npz(npz_template, speakers):
        embs = data["embs"]
        all_embs.append(embs)
        all_spk.append(np.full(len(embs), spk, dtype=object))
    return np.concatenate(all_embs, axis=0), np.concatenate(all_spk, axis=0)


def load_embeddings_filenames(
    npz_template: str, speakers: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings + the wav filename of each row -> (X_raw, filenames, speaker_ids)."""
    all_embs, all_fnames, all_spk = [], [], []
    for spk, data in _iter_speaker_npz(npz_template, speakers):
        embs = data["embs"]
        if "paths" not in data.files:
            raise KeyError(
                f"Embedding file for speaker {spk} is missing the 'paths' key, "
                "which is needed to join on filename."
            )
        paths = np.asarray(data["paths"]).astype(str)
        fnames = np.array([os.path.basename(p) for p in paths], dtype=object)
        all_embs.append(embs)
        all_fnames.append(fnames)
        all_spk.append(np.full(len(embs), spk, dtype=object))
    return (
        np.concatenate(all_embs, axis=0),
        np.concatenate(all_fnames, axis=0),
        np.concatenate(all_spk, axis=0),
    )


def load_esd_gt(gt_csv: str, sep: str = ";") -> pd.DataFrame:
    """Load + normalize the ESD ground truth (gender, speaker, emotion, filename, language)."""
    df = pd.read_csv(gt_csv, sep=sep)
    required = {"gender", "speaker", "emotion", "filename", "language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ESD GT missing columns: {sorted(missing)}")
    df = df.copy()
    df["filename"] = df["filename"].astype(str)
    df["speaker"] = df["speaker"].astype(str).str.zfill(4)
    df["gender"] = df["gender"].astype(str).str.lower()
    df["language"] = df["language"].astype(str).str.lower()
    df["emotion"] = df["emotion"].astype(str).str.lower()
    return df


def load_ravdess_gt(gt_csv: str, sep: str = ";") -> pd.DataFrame:
    """Load + normalize the RAVDESS ground truth (gender, actor_id, emotion, filename)."""
    df = pd.read_csv(gt_csv, sep=sep)
    required = {"gender", "actor_id", "emotion", "filename"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RAVDESS GT missing columns: {sorted(missing)}")
    df = df.copy()
    df["filename"] = df["filename"].astype(str)
    df["actor_id"] = df["actor_id"].astype(str).str.zfill(4)
    df["gender"] = df["gender"].astype(str).str.lower()
    df["emotion"] = df["emotion"].astype(str).str.lower()
    return df


def load_gt_emotion(gt_csv: str, sep: str = ";", normalize: bool = True) -> pd.DataFrame:
    """Minimal GT loader: just ``filename`` + normalized ``emotion``."""
    from .labels import normalize_label

    df = pd.read_csv(gt_csv, sep=sep)
    df = df.copy()
    df["filename"] = df["filename"].astype(str)
    if normalize:
        df["emotion"] = df["emotion"].astype(str).apply(normalize_label)
    else:
        df["emotion"] = df["emotion"].astype(str)
    return df


def merge_embeddings_with_gt(
    X_raw: np.ndarray,
    filenames: np.ndarray,
    gt: pd.DataFrame,
    meta_cols: List[str] | None = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Inner-join embeddings to ground truth on ``filename`` -> (X_aligned, meta)."""
    emb_df = pd.DataFrame(
        {"idx": np.arange(len(X_raw)), "filename": np.asarray(filenames).astype(str)}
    )
    merged = emb_df.merge(gt, on="filename", how="inner")
    if len(merged) == 0:
        raise RuntimeError(
            "Join produced 0 rows: filenames do not match between embeddings and GT."
        )
    X_aligned = X_raw[merged["idx"].values]
    if meta_cols is None:
        meta = merged.drop(columns=["idx"]).copy()
    else:
        meta = merged[meta_cols].copy()
    return X_aligned, meta
