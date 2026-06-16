"""
STEP 2.0 - Linear probes (ESD / RAVDESS)

Trains cross-validated logistic-regression probes on the standardized embeddings
to recover the linear directions for emotion and the nuisance attributes
(language, gender, speaker/actor). These cached directions feed every geometric
diagnostic.

Output: xai_weights/<DATASET>/<attribute>_probe.npz (weights W, classes, b) +
<attribute>_probe.json (CV metrics) -> shared cache read by the diagnostics.
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

import warnings

from sklearn.preprocessing import StandardScaler

from common.io import (
    load_embeddings_filenames, load_esd_gt, load_ravdess_gt, merge_embeddings_with_gt,
)
from common.probes import run_probe_and_extract_weights

warnings.filterwarnings("ignore")

# ============================= CONFIGURATION =============================
# Which dataset to train probes for: "esd", "ravdess", "all".
DATASET = "all"
SEED = 42
CV_FOLDS = 10
C = 1.0
MAX_ITER = 5000
# ========================================================================

_EMB = "data/{ds}/wav2vec2-lg-xlsr-en-speech-emotion-recognition/{{speaker}}_embeddings.npz"

DATASETS = {
    "esd": dict(
        npz_template=_EMB.format(ds="ESD"),
        gt_csv="data/ESD/ESD_GT.csv",
        cache_dir="xai_weights/ESD",
        speakers=[str(s).zfill(4) for s in range(1, 21)],
        gt_loader=load_esd_gt,
        meta_cols=["filename", "speaker", "gender", "language", "emotion"],
        # (attribute, group_column_or_None, group_cv)
        probes=[
            ("emotion", "speaker", True),
            ("language", "speaker", True),
            ("gender", "speaker", True),
            ("speaker", None, False),
        ],
    ),
    "ravdess": dict(
        npz_template=_EMB.format(ds="RAVDESS"),
        gt_csv="data/RAVDESS/RAVDESS_GT.csv",
        cache_dir="xai_weights/RAVDESS",
        speakers=[str(s).zfill(4) for s in range(1, 25)],
        gt_loader=load_ravdess_gt,
        meta_cols=["filename", "actor_id", "gender", "emotion"],
        probes=[
            ("emotion", "actor_id", True),
            ("gender", "actor_id", True),
            ("actor_id", None, False),
        ],
    ),
}


def run(dataset):
    cfg = DATASETS[dataset]
    print(f"{dataset.upper()} - LINEAR PROBES -> LATENT DIRECTIONS/SUBSPACES")

    # STEP 1: load embeddings + filenames, join with ground truth
    X_raw, filenames, _ = load_embeddings_filenames(cfg["npz_template"], cfg["speakers"])
    gt = cfg["gt_loader"](cfg["gt_csv"])
    X_aligned, meta = merge_embeddings_with_gt(X_raw, filenames, gt, meta_cols=cfg["meta_cols"])
    print(f"Merged rows: {len(meta)}  (N={len(X_raw)}, D={X_raw.shape[1]})")

    # STEP 2: standardize once (global scaler)
    X = StandardScaler().fit_transform(X_aligned)

    # STEP 3: train + cache one probe per attribute
    out = {}
    for name, group_col, group_cv in cfg["probes"]:
        groups = meta[group_col].values if group_col else None
        out[name] = run_probe_and_extract_weights(
            X=X, y_raw=meta[name].values, class_name=name, cache_dir=cfg["cache_dir"],
            groups=groups, group_cv=group_cv,
            seed=SEED, cv_folds=CV_FOLDS, C=C, max_iter=MAX_ITER,
        )
    print(f"\nDone. Cached directions are in: {cfg['cache_dir']}/")
    return out


def run_esd():
    return run("esd")


def run_ravdess():
    return run("ravdess")


def main():
    for name in (list(DATASETS) if DATASET == "all" else [DATASET]):
        run(name)


if __name__ == "__main__":
    main()
