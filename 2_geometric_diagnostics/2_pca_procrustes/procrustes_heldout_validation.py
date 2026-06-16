"""
STEP 2.3 - Procrustes held-out validation (the claim test)

Speaker-independent test that the RAVDESS <-> ESD mismatch of the
emotion-separating directions is a rotation. Fit the Procrustes rotation R* from
an ESD FIT subset, then classify HELD-OUT ESD speakers (mapped into the RAVDESS
frame) with the original RAVDESS probe, against the same probe without R*
(~chance baseline) and an ESD-CV oracle (loose ceiling). Repeated over random
splits, reported as mean +/- std.

Scope: R* is constrained only inside the <=10-dim probe-weight subspace, so this
shows the *directional* mismatch is a rotation, not that the two embedding spaces
differ by one global rotation.

Output: 2_geometric_diagnostics/2_pca_procrustes/output/ ->
heldout_validation_summary.json, heldout_validation.png, heldout_confusion.png
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
os.chdir(_ROOT)  # resolve every relative path below from the project root
 
import json
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
 
from common.io import (
    load_embeddings_filenames, load_esd_gt, load_ravdess_gt, merge_embeddings_with_gt,
)
from common.labels import normalize_label
from common.paths import output_dir
 
# ============================== CONFIGURATION ==============================
# Emotions shared by RAVDESS and ESD.
SHARED_EMOTIONS = sorted(["angry", "happy", "neutral", "sad", "surprised"])
 
# Held-out validation.
N_HELDOUT_SPEAKERS = 4    # ESD has 20 speakers -> 16 fit / 4 held-out
N_REPEATS = 5             # number of random held-out splits (report mean +/- std)
SEED = 42
 
# Logistic-regression probe.
PROBE_C = 1.0
PROBE_MAX_ITER = 5000
 
# Sanity threshold for the rotation-direction check (mean per-class cosine after R).
# Must reproduce the ~0.93-0.98 of Table 14; if it comes out near 0, R is inverted.
ROT_SANITY_MIN_COSINE = 0.80
 
# Embedding caches (STEP 0) and ground truth.
RAV_NPZ = "data/RAVDESS/wav2vec2-lg-xlsr-en-speech-emotion-recognition/{speaker}_embeddings.npz"
ESD_NPZ = "data/ESD/wav2vec2-lg-xlsr-en-speech-emotion-recognition/{speaker}_embeddings.npz"
RAV_GT_CSV = "data/RAVDESS/RAVDESS_GT.csv"
ESD_GT_CSV = "data/ESD/ESD_GT.csv"
RAV_ACTORS = [str(s).zfill(4) for s in range(1, 25)]
ESD_SPEAKERS = [str(s).zfill(4) for s in range(1, 21)]
 
OUT = output_dir(__file__)
# ==========================================================================
 
 
def _logreg():
    # [FIX minor] multi_class="multinomial" is deprecated: lbfgs is multinomial
    # by default. Dropping it removes the FutureWarning without changing results.
    return LogisticRegression(solver="lbfgs", C=PROBE_C,
                              max_iter=PROBE_MAX_ITER, random_state=SEED)
 
 
def _normrows(W):
    return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
 
 
def _mean_diag_cosine(A, B):
    """Mean per-row cosine between two row-normalized direction matrices."""
    A, B = _normrows(A), _normrows(B)
    return float(np.mean(np.sum(A * B, axis=1)))
 
 
def _load(npz_template, speakers, gt, meta_cols):
    """Load embeddings, join GT, normalize emotion, keep shared emotions."""
    X_raw, fnames, _ = load_embeddings_filenames(npz_template, speakers)
    X, meta = merge_embeddings_with_gt(X_raw, fnames, gt, meta_cols=meta_cols)
    meta = meta.copy()
    meta["emotion"] = meta["emotion"].map(normalize_label)
    keep = meta["emotion"].isin(SHARED_EMOTIONS).values
    return X[keep], meta[keep].reset_index(drop=True)
 
 
def _eval(y_true, y_pred):
    return (float(accuracy_score(y_true, y_pred)),
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)))
 
 
def main():
    os.makedirs(OUT, exist_ok=True)
    le = LabelEncoder().fit(SHARED_EMOTIONS)
    n_classes = len(SHARED_EMOTIONS)
    chance = 1.0 / n_classes
 
    # ---- STEP 1: load embeddings ----
    print("[1] Loading RAVDESS + ESD embeddings...")
    X_rav, meta_rav = _load(RAV_NPZ, RAV_ACTORS, load_ravdess_gt(RAV_GT_CSV),
                            ["filename", "actor_id", "emotion"])
    X_esd, meta_esd = _load(ESD_NPZ, ESD_SPEAKERS, load_esd_gt(ESD_GT_CSV),
                            ["filename", "speaker", "emotion"])
    y_rav = le.transform(meta_rav["emotion"].values)
    y_esd = le.transform(meta_esd["emotion"].values)
    speakers_esd = meta_esd["speaker"].values.astype(str)
    print(f"    RAVDESS: {X_rav.shape} | ESD: {X_esd.shape} | classes: {list(le.classes_)}")
 
    # ---- STEP 2: the ORIGINAL RAVDESS probe (trained once, never touched) ----
    # [FIX 1] ONE reference space for everything: the RAVDESS-standardized space,
    # i.e. the space the frozen probe lives in. ESD embeddings are pushed through
    # THE SAME scaler_rav, so R* maps ESD->RAVDESS within one metric space and the
    # frozen probe always receives coherent inputs. scaler_rav is fitted on
    # RAVDESS only, which is independent of the held-out ESD speakers -> no leakage.
    print("[2] Training the original RAVDESS probe (frozen afterwards)...")
    scaler_rav = StandardScaler().fit(X_rav)
    probe_rav = _logreg().fit(scaler_rav.transform(X_rav), y_rav)
    W_rav = probe_rav.coef_              # (C, D), rows ordered by probe_rav.classes_
    b_rav = probe_rav.intercept_         # (C,)
    rav_classes = probe_rav.classes_     # encoded labels in row order
    # [FIX 3] make sure the frozen probe really saw all shared classes
    assert set(rav_classes) == set(range(n_classes)), \
        f"RAVDESS probe is missing classes: {set(range(n_classes)) - set(rav_classes)}"
 
    # Pre-standardize ESD ONCE in the RAVDESS frame (no per-split ESD scaler).
    X_esd_rav = scaler_rav.transform(X_esd)
 
    unique_speakers = sorted(set(speakers_esd))
    if N_HELDOUT_SPEAKERS >= len(unique_speakers):
        raise ValueError("N_HELDOUT_SPEAKERS must be smaller than the number of ESD speakers.")
    rng = np.random.default_rng(SEED)
 
    # ---- STEPS 3-4: repeated speaker-held-out evaluation ----
    print(f"[3] Held-out validation over {N_REPEATS} random splits "
          f"({N_HELDOUT_SPEAKERS} held-out speakers each)...")
    rows = []          # per-repeat metrics
    sanity_cosines = []  # per-split rotation sanity check
    first_split = None  # keep one split's predictions for confusion matrices
    for rep in range(N_REPEATS):
        held = set(rng.choice(unique_speakers, size=N_HELDOUT_SPEAKERS, replace=False).tolist())
        mask_held = np.array([s in held for s in speakers_esd])
        mask_fit = ~mask_held
 
        # Everything is already in the RAVDESS frame (FIX 1): just slice.
        X_fit = X_esd_rav[mask_fit]
        X_held = X_esd_rav[mask_held]
        y_fit = y_esd[mask_fit]
        y_held = y_esd[mask_held]
 
        # ESD probe on FIT speakers -> its directions -> Procrustes R*
        probe_esd = _logreg().fit(X_fit, y_fit)
        # [FIX 3] the fit split must contain all shared classes before reindexing
        assert set(probe_esd.classes_) == set(rav_classes), \
            f"FIT split {sorted(held)} is missing an emotion class"
        # align the ESD-fit rows to the RAVDESS class order before Procrustes
        esd_row = {c: i for i, c in enumerate(probe_esd.classes_)}
        W_esd_fit = probe_esd.coef_[[esd_row[c] for c in rav_classes]]
        # orthogonal_procrustes(A, B) returns R with A @ R ~ B, so W_rav @ R ~ W_esd
        R, _ = orthogonal_procrustes(_normrows(W_rav), _normrows(W_esd_fit))
 
        # [FIX 2] EMPIRICAL direction check: after rotation the RAVDESS directions
        # must align with the ESD ones, reproducing Table 14 (~0.93-0.98). If this
        # is near 0, the rotation sense is wrong and the whole test is a false
        # negative -- so we fail loudly instead of silently reporting bad numbers.
        cos_after = _mean_diag_cosine(_normrows(W_rav) @ R, _normrows(W_esd_fit))
        sanity_cosines.append(cos_after)
        assert cos_after >= ROT_SANITY_MIN_COSINE, (
            f"Rotation sanity check failed (mean cosine {cos_after:.3f} < "
            f"{ROT_SANITY_MIN_COSINE}). R likely applied with the wrong sense."
        )
 
        # (3) ROTATED: map held-out ESD into the RAVDESS frame, classify with the
        #     untouched RAVDESS probe.
        #     W_rav @ R ~ W_esd as ROW directions => for COLUMN embeddings the
        #     ESD->RAVDESS map is z_rav = R @ z_esd, i.e. (X_held @ R.T) for rows.
        #     (The sense is not trusted by reasoning alone: FIX 2 verifies it.)
        logits_rot = (X_held @ R.T) @ W_rav.T + b_rav
        y_pred_rot = rav_classes[np.argmax(logits_rot, axis=1)]
        # (4) BASELINE: same RAVDESS probe, no rotation.
        logits_base = X_held @ W_rav.T + b_rav
        y_pred_base = rav_classes[np.argmax(logits_base, axis=1)]
        # ORACLE: the ESD probe itself on the held-out speakers (LOOSE upper bound).
        y_pred_oracle = probe_esd.predict(X_held)
 
        acc_rot, f1_rot = _eval(y_held, y_pred_rot)
        acc_base, f1_base = _eval(y_held, y_pred_base)
        acc_ora, f1_ora = _eval(y_held, y_pred_oracle)
        rows.append((acc_base, f1_base, acc_rot, f1_rot, acc_ora, f1_ora))
        print(f"    split {rep+1}: held-out={sorted(held)}  "
              f"cos(after R)={cos_after:.3f} | "
              f"baseline acc={acc_base:.3f} | procrustes acc={acc_rot:.3f} | "
              f"oracle acc={acc_ora:.3f}")
 
        if first_split is None:
            first_split = dict(held=sorted(held), y_held=y_held,
                               base=y_pred_base, rot=y_pred_rot)
 
    arr = np.array(rows)  # (N_REPEATS, 6)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    keys = ["baseline_acc", "baseline_f1", "procrustes_acc", "procrustes_f1", "oracle_acc", "oracle_f1"]
    summary_means = {k: float(mean[i]) for i, k in enumerate(keys)}
    summary_stds = {k + "_std": float(std[i]) for i, k in enumerate(keys)}
 
    # ---- Report ----
    print("\n" + "=" * 70)
    print("  HELD-OUT VALIDATION  (mean +/- std over splits)")
    print("=" * 70)
    print(f"  {'Condition':<26s} | {'Accuracy':>16s} | {'F1 (macro)':>16s}")
    print(f"  {'-'*26}-+-{'-'*16}-+-{'-'*16}")
    print(f"  {'Baseline (no rotation)':<26s} | {mean[0]:>7.3f} +/- {std[0]:<5.3f} | {mean[1]:>7.3f} +/- {std[1]:<5.3f}")
    print(f"  {'Procrustes (rotated)':<26s} | {mean[2]:>7.3f} +/- {std[2]:<5.3f} | {mean[3]:>7.3f} +/- {std[3]:<5.3f}")
    print(f"  {'Oracle (ESD probe)':<26s} | {mean[4]:>7.3f} +/- {std[4]:<5.3f} | {mean[5]:>7.3f} +/- {std[5]:<5.3f}")
    print(f"  {'Chance':<26s} | {chance:>16.3f} | {'-':>16s}")
    print("-" * 70)
    print(f"  Rotation sanity (mean cos after R): "
          f"{np.mean(sanity_cosines):.3f} +/- {np.std(sanity_cosines):.3f}  "
          f"(should reproduce Table 14)")
    print("=" * 70)
    gap = mean[4] - mean[0]
    closed = (mean[2] - mean[0]) / gap * 100 if gap > 1e-9 else float("nan")
    print(f"  Procrustes improvement over baseline: +{mean[2]-mean[0]:.3f} "
          f"({closed:.1f}% of the gap to the (loose) oracle closed)")
 
    # ---- Bar plot ----
    labels = ["Baseline\n(no rotation)", "Procrustes\n(rotated)", "Oracle\n(ESD probe)"]
    accs = [mean[0], mean[2], mean[4]]
    errs = [std[0], std[2], std[4]]
    colors = ["#B0BEC5", "#43A047", "#1E88E5"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, accs, yerr=errs, capsize=6, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(chance, color="red", linestyle="--", linewidth=1.2, label=f"Chance ({chance:.2f})")
    for i, (a, e) in enumerate(zip(accs, errs)):
        ax.text(i, a + e + 0.01, f"{a:.2f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Held-out accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Procrustes held-out validation (ESD, {N_HELDOUT_SPEAKERS} held-out speakers x {N_REPEATS} splits)\n"
                 "Rotation learnt on FIT speakers, applied to UNSEEN speakers", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "heldout_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT / 'heldout_validation.png'}")
 
    # ---- Confusion matrices for one split (baseline vs rotated) ----
    labels_idx = list(range(n_classes))
    names = [c.capitalize() for c in le.classes_]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in [(axes[0], "base", "Baseline (no rotation)"),
                           (axes[1], "rot", "Procrustes (rotated)")]:
        cm = confusion_matrix(first_split["y_held"], first_split[key], labels=labels_idx)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(labels_idx); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(labels_idx); ax.set_yticklabels(names)
        thr = cm.max() / 2 if cm.size else 0
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thr else "black", fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=11)
    fig.suptitle(f"Held-out ESD speakers {first_split['held']} classified by the original RAVDESS probe",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "heldout_confusion.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / 'heldout_confusion.png'}")
 
    # ---- Save summary ----
    report = {
        "shared_emotions": list(le.classes_),
        "n_classes": n_classes,
        "chance_level": chance,
        "n_heldout_speakers": N_HELDOUT_SPEAKERS,
        "n_repeats": N_REPEATS,
        "ravdess_n_samples": int(len(meta_rav)),
        "esd_n_samples": int(len(meta_esd)),
        "embedding_dim": int(X_rav.shape[1]),
        "reference_space": "RAVDESS-standardized (single scaler for all)",  # [FIX 1]
        "rotation_sanity_mean_cosine": float(np.mean(sanity_cosines)),       # [FIX 2]
        "rotation_sanity_std_cosine": float(np.std(sanity_cosines)),
        **summary_means,
        **summary_stds,
        # [FIX 4 + 5] honest scope in the machine-readable report too.
        "scope_note": (
            "Demonstrates that the mismatch of the EMOTION-SEPARATING DIRECTIONS "
            "is a rotation, NOT that the two embedding spaces differ by a single "
            "global rotation: R* is constrained only in the <=10-dim weight "
            "subspace read by the RAVDESS probe."
        ),
        "oracle_note": (
            "Oracle uses all 2048 dims while the rotated probe is confined to the "
            "weight subspace, so it is a LOOSE ceiling; '% gap closed' need not "
            "reach 100% even under a perfect rotation."
        ),
        "interpretation": (
            "If 'procrustes_acc' is well above 'baseline_acc' (and approaches "
            "'oracle_acc'), the RAVDESS<->ESD directional mismatch on UNSEEN "
            "speakers is recovered by a single rotation learnt on other speakers."
        ),
    }
    with open(OUT / "heldout_validation_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {OUT / 'heldout_validation_summary.json'}")
 
 
if __name__ == "__main__":
    main()
 