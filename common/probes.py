"""
common.probes - logistic-regression probe training with cross-validated metrics
and on-disk caching, plus loaders for cached weights.

Each probe is cached as <class>_probe.npz (W, b, classes, class_name) +
<class>_probe.json (CV metrics) under xai_weights/<DATASET>/ as an intermediate
artifact.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    GroupKFold, StratifiedKFold, cross_val_predict, cross_val_score,
)
from sklearn.preprocessing import LabelEncoder


def make_cv(groups: Optional[np.ndarray], seed: int, cv_folds: int, group_cv: bool):
    """Pick a CV splitter: GroupKFold when grouping is requested and possible,
    StratifiedKFold otherwise. Returns ``(cv, groups_for_cv)``."""
    if group_cv and groups is not None:
        n_splits = min(cv_folds, len(np.unique(groups)))
        if n_splits < 2:
            return StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=seed), None
        return GroupKFold(n_splits=n_splits), groups
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed), None


def run_probe_and_extract_weights(
    X: np.ndarray,
    y_raw: np.ndarray,
    class_name: str,
    cache_dir: str,
    groups: Optional[np.ndarray] = None,
    group_cv: bool = True,
    seed: int = 42,
    cv_folds: int = 10,
    C: float = 1.0,
    max_iter: int = 5000,
) -> Dict[str, Any]:
    """Train a logistic-regression probe with CV metrics, fit on the full data to
    extract the separating directions (weights), and cache everything. Loads the
    cache instead of retraining if it already exists."""
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, f"{class_name}_probe.npz")
    json_path = os.path.join(cache_dir, f"{class_name}_probe.json")

    if os.path.exists(npz_path) and os.path.exists(json_path):
        print(f"[CACHE HIT] {class_name} -> {npz_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        data = np.load(npz_path, allow_pickle=True)
        return {"cached": True, "meta": meta, "W": data["W"], "classes": data["classes"]}

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_

    cv, groups_for_cv = make_cv(groups, seed, cv_folds, group_cv)

    logreg = LogisticRegression(
        multi_class="multinomial" if len(classes) > 2 else "auto",
        solver="lbfgs", C=C, max_iter=max_iter, random_state=seed,
    )

    print(f"\n=== PROBE: {class_name} ===")
    print(f"Classes ({len(classes)}): {list(classes)}")
    print(f"CV: {type(cv).__name__} (n_splits={cv.get_n_splits()})")

    scores_acc = cross_val_score(logreg, X, y, cv=cv, groups=groups_for_cv, scoring="accuracy")
    acc_mean, acc_std = float(scores_acc.mean()), float(scores_acc.std())
    print(f"Accuracy (CV mean): {acc_mean:.4f} ± {acc_std:.4f}")

    y_pred = cross_val_predict(logreg, X, y, cv=cv, groups=groups_for_cv, method="predict")

    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y, y_pred, average="macro", zero_division=0))
    f1m = float(f1_score(y, y_pred, average="macro"))

    print(f"Accuracy (CV-pred): {acc:.4f} | F1(macro): {f1m:.4f}")

    report = classification_report(y, y_pred, target_names=classes, digits=4)
    cm = confusion_matrix(y, y_pred)

    logreg.fit(X, y)
    W = logreg.coef_.astype(np.float32)
    b = logreg.intercept_.astype(np.float32)

    np.savez_compressed(npz_path, W=W, b=b, classes=classes.astype(object), class_name=class_name)

    meta = {
        "class_name": class_name,
        "n_samples": int(X.shape[0]),
        "n_dims": int(X.shape[1]),
        "n_classes": int(len(classes)),
        "cv_type": type(cv).__name__,
        "group_cv": bool(group_cv),
        "acc_mean": acc_mean, "acc_std": acc_std, "acc_cv_pred": acc,
        "precision_macro": prec, "recall_macro": rec, "f1_macro": f1m,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "C": float(C), "max_iter": int(max_iter), "seed": int(seed),
        "cv_folds_requested": int(cv_folds),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVED] {npz_path}")
    return {"cached": False, "meta": meta, "W": W, "b": b, "classes": classes}


def ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing probe file: {path}")


def load_probe_npz(path: str, normalize_classes: bool = False) -> Dict[str, Any]:
    """Load a probe ``.npz`` -> dict with ``W`` (float64), ``b`` (if present), ``classes``."""
    data = np.load(path, allow_pickle=True)
    if "W" not in data.files:
        raise KeyError(f"Probe file missing 'W': {path}")
    out: Dict[str, Any] = {"W": data["W"].astype(np.float64)}
    out["b"] = data["b"] if "b" in data.files else None
    if "classes" in data.files:
        classes = data["classes"].astype(str)
        if normalize_classes:
            from .labels import normalize_labels_array
            classes = normalize_labels_array(classes)
        out["classes"] = classes
    else:
        out["classes"] = None
    return out


def load_probe_dir(cache_dir: str, probe_name: str = "emotion",
                   normalize_classes: bool = False) -> Dict[str, Any]:
    """Load ``<cache_dir>/<probe_name>_probe.npz`` (+ its JSON metadata if any)."""
    npz_path = os.path.join(cache_dir, f"{probe_name}_probe.npz")
    json_path = os.path.join(cache_dir, f"{probe_name}_probe.json")
    ensure_exists(npz_path)
    out = load_probe_npz(npz_path, normalize_classes=normalize_classes)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            out["meta"] = json.load(f)
    else:
        out["meta"] = None
    return out
