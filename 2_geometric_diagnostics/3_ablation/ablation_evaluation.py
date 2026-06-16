"""
STEP 2.4 - Joint subspace ablation (RAVDESS / ESD)

Projects the nuisance probe subspaces out of the standardized embeddings and
re-evaluates the emotion probe (baseline vs each individual ablation vs joint),
plus neighborhood-composition stats.

Output: 2_geometric_diagnostics/3_ablation/output/<DATASET>/ -> *_folds_*.csv,
*_joint_ablation_summary.csv, *_joint_ablation_report.json, *_X_abl_joint.npz
"""

import os
import json
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

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
# Embedding caches (STEP 0) and probe caches (STEP 2.0):
RAVDESS_NPZ = "data/RAVDESS/wav2vec2-lg-xlsr-en-speech-emotion-recognition/{speaker}_embeddings.npz"
ESD_NPZ = "data/ESD/wav2vec2-lg-xlsr-en-speech-emotion-recognition/{speaker}_embeddings.npz"
RAVDESS_GT = "data/RAVDESS/RAVDESS_GT.csv"
ESD_GT = "data/ESD/ESD_GT.csv"
RAVDESS_PROBE_DIR = "xai_weights/RAVDESS"
ESD_PROBE_DIR = "xai_weights/ESD"
# ==========================================================================



# ============================================================
# I/O — reused from existing scripts
# ============================================================

def load_embeddings_per_speaker(
    npz_template: str,
    speakers: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_X, all_fnames, all_groups = [], [], []
    for spk in speakers:
        path = npz_template.format(speaker=spk)
        if not os.path.exists(path):
            print(f"[WARN] Missing: {path} (skipping)")
            continue
        data = np.load(path, allow_pickle=True)
        if "embs" not in data.files:
            print(f"[WARN] No 'embs' in: {path} (skipping)")
            continue
        X = data["embs"].astype(np.float32)
        all_X.append(X)
        all_groups.append(np.full((X.shape[0],), str(spk), dtype=object))
        if "paths" in data.files:
            fn = np.array([os.path.basename(str(p)) for p in data["paths"]], dtype=object)
        elif "filenames" in data.files:
            fn = data["filenames"].astype(object)
        else:
            fn = np.array([""] * X.shape[0], dtype=object)
        all_fnames.append(fn)

    if not all_X:
        raise RuntimeError("No embeddings loaded.")
    return np.concatenate(all_X), np.concatenate(all_fnames), np.concatenate(all_groups)


def load_gt_csv(csv_path: str, sep: str = ";") -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=sep)
    if "filename" not in df.columns:
        if "full_path" in df.columns:
            df["filename"] = df["full_path"].astype(str).apply(os.path.basename)
        elif "filepath" in df.columns:
            df["filename"] = df["filepath"].astype(str).apply(os.path.basename)
        else:
            raise ValueError("GT CSV must contain 'filename' or 'full_path'/'filepath'.")
    df["filename"] = df["filename"].astype(str)
    return df


def merge_embeddings_with_gt(
    X: np.ndarray, fnames: np.ndarray, spk_npz: np.ndarray, gt: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    df_e = pd.DataFrame({
        "filename": fnames.astype(str),
        "speaker_npz": spk_npz.astype(str),
    }).reset_index().rename(columns={"index": "_row"})

    merged = df_e.merge(gt, on="filename", how="inner")
    if len(merged) == 0:
        raise RuntimeError("Join produced 0 rows.")
    X_aligned = X[merged["_row"].to_numpy()]
    meta = merged.drop(columns=["_row"]).reset_index(drop=True)
    return X_aligned, meta


# ============================================================
# Probe loading and subspace construction
# ============================================================

def load_probe_W(probe_path: str) -> np.ndarray:
    data = np.load(probe_path, allow_pickle=True)
    if "W" in data.files:
        return data["W"].astype(np.float64)
    if "w" in data.files:
        return data["w"].astype(np.float64)[None, :]
    raise KeyError(f"No 'W' or 'w' in {probe_path}")


def orthonormal_basis_from_W(W: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Returns U (D, r) orthonormal basis for span(W)."""
    U, s, _ = np.linalg.svd(W.T, full_matrices=False)
    r = int(np.sum(s > rtol))
    if r == 0:
        return np.zeros((W.shape[1], 0), dtype=np.float64)
    return U[:, :r]


def build_joint_nuisance_basis(
    probe_dir: str,
    nuisance_names: List[str],
    rtol: float = 1e-10,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load all nuisance probe weight matrices, stack them,
    and compute a single joint orthonormal basis.

    Returns:
        U_joint : (D, r_joint) orthonormal basis spanning all nuisance subspaces
        ranks   : dict mapping each nuisance name to its individual rank
    """
    all_rows = []
    ranks = {}

    for name in nuisance_names:
        probe_path = os.path.join(probe_dir, f"{name}_probe.npz")
        if not os.path.exists(probe_path):
            print(f"[WARN] Missing probe: {probe_path} — skipping.")
            continue
        W = load_probe_W(probe_path)
        U_ind = orthonormal_basis_from_W(W, rtol=rtol)
        ranks[name] = U_ind.shape[1]
        all_rows.append(W)
        print(f"  Loaded {name:10s} | W shape {W.shape} | individual rank = {U_ind.shape[1]}")

    if not all_rows:
        raise RuntimeError("No nuisance probes found.")

    # Stack all weight rows and compute joint basis
    W_joint = np.concatenate(all_rows, axis=0)  # (sum_of_rows, D)
    U_joint = orthonormal_basis_from_W(W_joint, rtol=rtol)

    sum_individual = sum(ranks.values())
    print(f"\n  Joint W shape: {W_joint.shape}")
    print(f"  Sum of individual ranks: {sum_individual}")
    print(f"  Joint basis rank:        {U_joint.shape[1]}")
    if U_joint.shape[1] < sum_individual:
        print(f"  → {sum_individual - U_joint.shape[1]} dimensions shared across nuisance subspaces")

    return U_joint, ranks


def remove_subspace(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """X' = X - (X @ U) @ U.T"""
    if U.shape[1] == 0:
        return X.copy()
    X64 = X.astype(np.float64, copy=False)
    U64 = U.astype(np.float64, copy=False)
    return (X64 - (X64 @ U64) @ U64.T).astype(X.dtype, copy=False)


# ============================================================
# Emotion probe evaluation (full CV reporting)
# ============================================================

def build_cv(y, group_ids, seed, cv_folds):
    if group_ids is not None:
        unique = np.unique(group_ids)
        n_splits = min(cv_folds, len(unique))
        if n_splits < 2:
            return StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=seed), None
        return GroupKFold(n_splits=n_splits), group_ids
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed), None


def emotion_probe_cv_full(
    X: np.ndarray,
    y_labels: np.ndarray,
    group_ids: Optional[np.ndarray],
    seed: int = 42,
    cv_folds: int = 10,
    solver: str = "lbfgs",
    max_iter: int = 5000,
    C: float = 1.0,
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Any]]:
    le = LabelEncoder()
    y = le.fit_transform(y_labels.astype(str))

    logreg = LogisticRegression(
        multi_class="multinomial",
        solver=solver, C=C, max_iter=max_iter, random_state=seed,
    )

    cv, groups_for_cv = build_cv(y, group_ids, seed, cv_folds)

    scoring = {
        "acc": "accuracy",
        "prec_macro": "precision_macro",
        "rec_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    cvres = cross_validate(
        logreg, X, y, cv=cv, groups=groups_for_cv,
        scoring=scoring, return_train_score=False,
    )

    fold_df = pd.DataFrame({
        "fold": np.arange(1, len(cvres["test_acc"]) + 1),
        "acc": cvres["test_acc"],
        "precision_macro": cvres["test_prec_macro"],
        "recall_macro": cvres["test_rec_macro"],
        "f1_macro": cvres["test_f1_macro"],
    })

    y_pred = cross_val_predict(
        logreg, X, y, cv=cv, groups=groups_for_cv, method="predict",
    )

    summary = {
        "acc_mean": float(fold_df["acc"].mean()),
        "acc_std": float(fold_df["acc"].std(ddof=0)),
        "f1_macro_mean": float(fold_df["f1_macro"].mean()),
        "f1_macro_std": float(fold_df["f1_macro"].std(ddof=0)),
        "precision_macro_mean": float(fold_df["precision_macro"].mean()),
        "recall_macro_mean": float(fold_df["recall_macro"].mean()),
        "acc_cv_pred": float(accuracy_score(y, y_pred)),
        "precision_macro_cv_pred": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro_cv_pred": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro_cv_pred": float(f1_score(y, y_pred, average="macro")),
        "n_classes": int(len(le.classes_)),
        "classes": [str(c) for c in le.classes_],
    }

    extras = {
        "confusion_matrix_cv_pred": confusion_matrix(y, y_pred).tolist(),
        "cv_n_splits": int(getattr(cv, "n_splits", cv_folds)),
    }

    return summary, fold_df, extras


# ============================================================
# Neighborhood composition (for completeness on ablated data)
# ============================================================

def neighborhood_composition(
    X: np.ndarray, meta: pd.DataFrame, attributes: List[str], k: int = 15, metric: str = "euclidean",
) -> Dict[str, Dict[str, float]]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]
    out = {}
    for attr in attributes:
        vals = meta[attr].astype(str).to_numpy()
        same = (vals[idx] == vals[:, None]).mean(axis=1) * 100.0
        out[attr] = {"mean_pct": float(np.mean(same)), "std_pct": float(np.std(same))}
    return out


# ============================================================
# Main pipeline: joint ablation
# ============================================================

def run_joint_ablation(
    dataset_name: str,
    npz_template: str,
    speakers: List[str],
    gt_csv: str,
    gt_sep: str,
    probe_dir: str,
    out_dir: str,
    nuisance_names: List[str],
    emotion_col: str = "emotion",
    group_col: Optional[str] = "speaker",
    neighborhood_attrs: Optional[List[str]] = None,
    k: int = 15,
    seed: int = 42,
    cv_folds: int = 10,
    C: float = 1.0,
    max_iter: int = 5000,
    save_ablated_embeddings: bool = True,
) -> Dict[str, Any]:

    print(f"\n{'='*60}")
    print(f"  JOINT SUBSPACE ABLATION — {dataset_name}")
    print(f"  Removing: {nuisance_names}")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    # --- Load and prepare data ---
    print("\n[1] Loading embeddings...")
    X_raw, fnames, spk_npz = load_embeddings_per_speaker(npz_template, speakers)
    print(f"    N={len(X_raw)}, D={X_raw.shape[1]}")

    print("[2] Loading GT + merging...")
    gt = load_gt_csv(gt_csv, sep=gt_sep)
    X_aligned, meta = merge_embeddings_with_gt(X_raw, fnames, spk_npz, gt)
    print(f"    Merged rows: {len(meta)}")

    print("[3] Standardizing...")
    X_std = StandardScaler().fit_transform(X_aligned).astype(np.float32)

    groups_for_cv = (
        meta[group_col].astype(str).to_numpy()
        if (group_col and group_col in meta.columns) else None
    )
    y_emo = meta[emotion_col].astype(str).to_numpy()

    if neighborhood_attrs is None:
        neighborhood_attrs = [c for c in ["emotion", "speaker", "gender", "language", "actor_id"] if c in meta.columns]

    # --- Baseline emotion probe ---
    print("\n[4] Baseline emotion probe...")
    base_summ, base_folds, base_extras = emotion_probe_cv_full(
        X_std, y_emo, groups_for_cv, seed=seed, cv_folds=cv_folds, C=C, max_iter=max_iter,
    )
    print(f"    Acc={base_summ['acc_cv_pred']:.4f}  F1={base_summ['f1_macro_cv_pred']:.4f}")

    base_folds.to_csv(os.path.join(out_dir, f"{dataset_name}_folds_baseline.csv"), index=False)

    # --- Build joint nuisance basis ---
    print("\n[5] Building joint nuisance basis...")
    U_joint, individual_ranks = build_joint_nuisance_basis(probe_dir, nuisance_names)
    joint_rank = U_joint.shape[1]

    # --- Individual ablations (for comparison in the same table) ---
    print("\n[6] Running individual ablations for comparison...")
    individual_results = {}
    for name in nuisance_names:
        probe_path = os.path.join(probe_dir, f"{name}_probe.npz")
        if not os.path.exists(probe_path):
            continue
        W = load_probe_W(probe_path)
        U_ind = orthonormal_basis_from_W(W)
        X_abl_ind = remove_subspace(X_std, U_ind)

        summ, folds, extras = emotion_probe_cv_full(
            X_abl_ind, y_emo, groups_for_cv, seed=seed, cv_folds=cv_folds, C=C, max_iter=max_iter,
        )
        print(f"    Remove {name:10s} (rank {U_ind.shape[1]:>2d}): "
              f"Acc={summ['acc_cv_pred']:.4f}  F1={summ['f1_macro_cv_pred']:.4f}")

        folds.to_csv(os.path.join(out_dir, f"{dataset_name}_folds_abl_{name}.csv"), index=False)
        individual_results[name] = {"summary": summ, "rank": U_ind.shape[1], "extras": extras}

    # --- Joint ablation ---
    print(f"\n[7] Joint ablation (rank {joint_rank})...")
    X_abl_joint = remove_subspace(X_std, U_joint)

    joint_summ, joint_folds, joint_extras = emotion_probe_cv_full(
        X_abl_joint, y_emo, groups_for_cv, seed=seed, cv_folds=cv_folds, C=C, max_iter=max_iter,
    )
    print(f"    Acc={joint_summ['acc_cv_pred']:.4f}  F1={joint_summ['f1_macro_cv_pred']:.4f}")

    joint_folds.to_csv(os.path.join(out_dir, f"{dataset_name}_folds_abl_joint.csv"), index=False)

    # --- Neighborhood composition on joint-ablated embeddings ---
    print("\n[8] Neighborhood composition (joint-ablated)...")
    nn_baseline = neighborhood_composition(X_std, meta, neighborhood_attrs, k=k, metric="euclidean")
    nn_joint = neighborhood_composition(X_abl_joint, meta, neighborhood_attrs, k=k, metric="euclidean")

    print(f"    {'Attribute':12s} | {'Baseline':>10s} | {'Joint-ablated':>14s}")
    print(f"    {'-'*12}-+-{'-'*10}-+-{'-'*14}")
    for attr in neighborhood_attrs:
        b = nn_baseline[attr]["mean_pct"]
        j = nn_joint[attr]["mean_pct"]
        print(f"    {attr:12s} | {b:9.2f}% | {j:13.2f}%")

    # --- Save ablated embeddings ---
    if save_ablated_embeddings:
        npz_path = os.path.join(out_dir, f"{dataset_name}_X_abl_joint.npz")
        np.savez_compressed(
            npz_path,
            X=X_abl_joint.astype(np.float32),
            filename=meta["filename"].astype(str).to_numpy(),
            removed_subspaces=np.array(nuisance_names, dtype=object),
            joint_rank=np.array([joint_rank]),
        )
        print(f"\n[SAVED] {npz_path}")

    # --- Build summary table ---
    rows = [{
        "dataset": dataset_name,
        "condition": "baseline",
        "removed": "none",
        "removed_rank": 0,
        "acc_cv_pred": base_summ["acc_cv_pred"],
        "f1_macro_cv_pred": base_summ["f1_macro_cv_pred"],
        "precision_macro_cv_pred": base_summ["precision_macro_cv_pred"],
        "recall_macro_cv_pred": base_summ["recall_macro_cv_pred"],
        "acc_mean": base_summ["acc_mean"],
        "acc_std": base_summ["acc_std"],
        "f1_macro_mean": base_summ["f1_macro_mean"],
        "f1_macro_std": base_summ["f1_macro_std"],
    }]

    for name, res in individual_results.items():
        s = res["summary"]
        rows.append({
            "dataset": dataset_name,
            "condition": f"remove_{name}",
            "removed": name,
            "removed_rank": res["rank"],
            "acc_cv_pred": s["acc_cv_pred"],
            "f1_macro_cv_pred": s["f1_macro_cv_pred"],
            "precision_macro_cv_pred": s["precision_macro_cv_pred"],
            "recall_macro_cv_pred": s["recall_macro_cv_pred"],
            "acc_mean": s["acc_mean"],
            "acc_std": s["acc_std"],
            "f1_macro_mean": s["f1_macro_mean"],
            "f1_macro_std": s["f1_macro_std"],
        })

    rows.append({
        "dataset": dataset_name,
        "condition": "remove_ALL_nuisance",
        "removed": "+".join(nuisance_names),
        "removed_rank": joint_rank,
        "acc_cv_pred": joint_summ["acc_cv_pred"],
        "f1_macro_cv_pred": joint_summ["f1_macro_cv_pred"],
        "precision_macro_cv_pred": joint_summ["precision_macro_cv_pred"],
        "recall_macro_cv_pred": joint_summ["recall_macro_cv_pred"],
        "acc_mean": joint_summ["acc_mean"],
        "acc_std": joint_summ["acc_std"],
        "f1_macro_mean": joint_summ["f1_macro_mean"],
        "f1_macro_std": joint_summ["f1_macro_std"],
    })

    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"{dataset_name}_joint_ablation_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    # --- Print final summary table ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {dataset_name}")
    print(f"{'='*70}")
    print(f"  {'Condition':<28s} | {'Rank':>4s} | {'Acc':>7s} | {'F1':>7s}")
    print(f"  {'-'*28}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}")
    for _, row in df_summary.iterrows():
        print(f"  {row['condition']:<28s} | {row['removed_rank']:>4d} | "
              f"{row['acc_cv_pred']:>6.4f} | {row['f1_macro_cv_pred']:>6.4f}")

    # --- JSON dump ---
    report = {
        "dataset": dataset_name,
        "nuisance_names": nuisance_names,
        "individual_ranks": individual_ranks,
        "joint_rank": joint_rank,
        "baseline": base_summ,
        "individual_ablations": {
            name: res["summary"] for name, res in individual_results.items()
        },
        "joint_ablation": joint_summ,
        "neighborhood_baseline": nn_baseline,
        "neighborhood_joint_ablated": nn_joint,
        "confusion_matrix_baseline": base_extras["confusion_matrix_cv_pred"],
        "confusion_matrix_joint": joint_extras["confusion_matrix_cv_pred"],
    }

    json_path = os.path.join(out_dir, f"{dataset_name}_joint_ablation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] {json_path}")

    return report


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    # --- RAVDESS ---
    run_joint_ablation(
        dataset_name="RAVDESS",
        npz_template=RAVDESS_NPZ,
        speakers=[str(a).zfill(4) for a in range(1, 25)],
        gt_csv=RAVDESS_GT,
        gt_sep=";",
        probe_dir=RAVDESS_PROBE_DIR,
        out_dir=str(OUT / "RAVDESS"),
        nuisance_names=["gender", "actor_id"],
        emotion_col="emotion",
        group_col="actor_id",
        neighborhood_attrs=["emotion", "actor_id", "gender"],
        k=15, seed=42, cv_folds=10, C=1.0, max_iter=5000,
        save_ablated_embeddings=True,
    )

    # --- ESD ---
    run_joint_ablation(
        dataset_name="ESD",
        npz_template=ESD_NPZ,
        speakers=[str(s).zfill(4) for s in range(1, 21)],
        gt_csv=ESD_GT,
        gt_sep=";",
        probe_dir=ESD_PROBE_DIR,
        out_dir=str(OUT / "ESD"),
        nuisance_names=["language", "speaker", "gender"],
        emotion_col="emotion",
        group_col="speaker",
        neighborhood_attrs=["emotion", "speaker", "gender", "language"],
        k=15, seed=42, cv_folds=10, C=1.0, max_iter=5000,
        save_ablated_embeddings=True,
    )


# --- ESD English only ---
    # Filter: speakers 0011-0020 are English
    run_joint_ablation(
        dataset_name="ESD_english",
        npz_template=ESD_NPZ,
        speakers=[str(s).zfill(4) for s in range(11, 21)],
        gt_csv=ESD_GT,
        gt_sep=";",
        probe_dir=ESD_PROBE_DIR,  # Uses the FULL ESD probes (trained on all 20 speakers)
        out_dir=str(OUT / "ESD_english"),
        nuisance_names=["speaker", "gender"],  # No language probe — single language
        emotion_col="emotion",
        group_col="speaker",
        neighborhood_attrs=["emotion", "speaker", "gender"],
        k=15, seed=42, cv_folds=10, C=1.0, max_iter=5000,
        save_ablated_embeddings=True,
    )

    # --- ESD Chinese only ---
    # Filter: speakers 0001-0010 are Chinese
    run_joint_ablation(
        dataset_name="ESD_chinese",
        npz_template=ESD_NPZ,
        speakers=[str(s).zfill(4) for s in range(1, 11)],
        gt_csv=ESD_GT,
        gt_sep=";",
        probe_dir=ESD_PROBE_DIR,  # Uses the FULL ESD probes
        out_dir=str(OUT / "ESD_chinese"),
        nuisance_names=["speaker", "gender"],  # No language probe — single language
        emotion_col="emotion",
        group_col="speaker",
        neighborhood_attrs=["emotion", "speaker", "gender"],
        k=15, seed=42, cv_folds=10, C=1.0, max_iter=5000,
        save_ablated_embeddings=True,
    )