"""
STEP 5 - Cross-lingual evaluation: base vs retrained head

Evaluates emotion recognition on three out-of-domain non-English datasets
(AESDD-Greek, CaFE-French, EMODB-German), comparing the base HuggingFace
classifier against the frozen backbone + retrained LNMLPHead. Both are masked to
each dataset's shared emotions; reports accuracy/precision/recall/F1 (overall and
per class) and confusion matrices.

Output: 5_evaluation/output/<DATASET>/ -> <DATASET>_results.json,
<DATASET>_confusion_comparison.png
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
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from common.embeddings import load_audio
from common.paths import output_dir
from common.retrained_model import load_retrained_model, embed_meanpool

# ============================= CONFIGURATION =============================
# Which dataset to evaluate: "aesdd", "cafe", "emodb", "all".
DATASET = "all"
# Retrained head to load (saved_models/<condition>/): "all", "english", "chinese".
RETRAINED_DIR = "4_model_retraining/saved_models/all"
# ========================================================================
TARGET_SR = 16_000

EMOTION_REMAP = {
    "anger": "angry", "happiness": "happy", "joy": "happy",
    "sadness": "sad", "surprise": "surprised",
}

# Per-dataset config: GT, audio root, audio extensions, shared emotions, language.
DATASETS = {
    "aesdd": dict(gt="data/AESDD/AESDD_GT.csv", root="data/AESDD/AESDD_Dataset", exts=(".wav",),
                  shared=["angry", "happy", "sad"], language="Greek"),
    "cafe": dict(gt="data/CaFE/CaFE_GT.csv", root="data/CaFE/CaFe_Dataset", exts=(".aiff", ".aif", ".wav"),
                 shared=["angry", "happy", "neutral", "sad", "surprised"], language="French"),
    "emodb": dict(gt="data/EMODB/EMODB_GT.csv", root="data/EMODB/EMODB_Dataset", exts=(".wav",),
                  shared=["angry", "happy", "neutral", "sad"], language="German"),
}

OUT = output_dir(__file__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _norm_emotion(value: str) -> str:
    v = str(value).strip().lower()
    return EMOTION_REMAP.get(v, v)


def _build_path_map(root: str, exts) -> dict:
    base = Path(root)
    exts = tuple(e.lower() for e in exts)
    path_map = {}
    if not base.exists():
        return path_map
    for f in base.rglob("*"):
        if f.suffix.lower() in exts:
            path_map.setdefault(f.name, str(f))
    return path_map


def load_dataset(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["gt"], sep=";")
    df["emotion"] = df["emotion"].apply(_norm_emotion)
    df = df[df["emotion"].isin(cfg["shared"])].copy()

    path_map = _build_path_map(cfg["root"], cfg["exts"])
    if not path_map:
        raise FileNotFoundError(
            f"No audio found under {cfg['root']}/. Place the dataset audio there first."
        )
    df["local_path"] = df["filename"].astype(str).str.strip().apply(
        lambda fn: path_map.get(Path(fn).name, "")
    )
    df = df[df["local_path"].apply(lambda p: p != "" and os.path.exists(p))].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No samples resolved to existing audio files.")
    return df


# ---------------------------------------------------------------------------
# Inference (masked to the dataset's shared emotions)
# ---------------------------------------------------------------------------
def _hf_remap(base_model) -> dict:
    remap = {}
    for hf_id, hf_label in base_model.config.id2label.items():
        lbl = str(hf_label).lower().strip()
        if lbl in ("calm", "neutral"):
            remap[int(hf_id)] = "neutral"
        elif lbl in ("surprised", "surprise"):
            remap[int(hf_id)] = "surprised"
        else:
            remap[int(hf_id)] = lbl
    return remap


def run_inference(models, paths, shared, device):
    """Return (y_pred_base, y_pred_retrained) for a list of audio paths."""
    import torch

    extractor = models["extractor"]
    base_model = models["base_model"]
    backbone = models["backbone"]
    head = models["head"]
    label2id = models["label2id"]
    id2label = models["id2label"]

    hf_remap = _hf_remap(base_model)
    shared_hf_ids = [i for i, lab in hf_remap.items() if lab in shared]
    shared_re_ids = [label2id[e] for e in shared if e in label2id]

    y_base, y_retr = [], []
    with torch.no_grad():
        for path in paths:
            try:
                wav = load_audio(path)
                inputs = extractor(wav, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Base model
                logits_b = base_model(**inputs).logits
                mb = torch.full_like(logits_b, float("-inf"))
                for idx in shared_hf_ids:
                    mb[0, idx] = logits_b[0, idx]
                y_base.append(hf_remap.get(int(torch.argmax(mb, dim=-1)), "unknown"))

                # Retrained head
                emb = embed_meanpool(backbone, inputs)
                logits_r = head(emb)
                mr = torch.full_like(logits_r, float("-inf"))
                for idx in shared_re_ids:
                    mr[0, idx] = logits_r[0, idx]
                y_retr.append(id2label.get(int(torch.argmax(mr, dim=-1)), "unknown"))
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {path} -> {e}")
                y_base.append("error")
                y_retr.append("error")
    return y_base, y_retr


# ---------------------------------------------------------------------------
# Metrics + plots
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, labels, model_name):
    yt, yp = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, labels=labels, average="macro", zero_division=0)
    rec = recall_score(yt, yp, labels=labels, average="macro", zero_division=0)
    f1m = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)

    per_p = precision_score(yt, yp, labels=labels, average=None, zero_division=0)
    per_r = recall_score(yt, yp, labels=labels, average=None, zero_division=0)
    per_f = f1_score(yt, yp, labels=labels, average=None, zero_division=0)
    per_acc = {e: float(accuracy_score((yt == e).astype(int), (yp == e).astype(int))) for e in labels}

    all_labels = sorted(set(list(yt) + list(yp)))
    print(f"\n  {model_name}: acc={acc:.4f}  f1={f1m:.4f}")
    return {
        "model": model_name,
        "accuracy": float(acc), "precision_macro": float(prec),
        "recall_macro": float(rec), "f1_macro": float(f1m),
        "per_class_accuracy": per_acc,
        "per_class_precision": {e: float(per_p[i]) for i, e in enumerate(labels)},
        "per_class_recall": {e: float(per_r[i]) for i, e in enumerate(labels)},
        "per_class_f1": {e: float(per_f[i]) for i, e in enumerate(labels)},
        "cm_full_labels": all_labels,
        "cm_full": confusion_matrix(yt, yp, labels=all_labels).tolist(),
        "cm_shared": confusion_matrix(yt, yp, labels=labels).tolist(),
        "prediction_distribution": dict(Counter(yp)),
        "classification_report": classification_report(yt, yp, labels=labels, digits=4, zero_division=0),
    }


def _plot_cm(ax, cm, labels, title):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.capitalize() for l in labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.capitalize() for l in labels])
    thr = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)


def plot_confusions(dataset, cfg, m_base, m_retr, out_dir):
    labels = cfg["shared"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_cm(axes[0], np.array(m_base["cm_shared"]), labels,
             f"Base (RAVDESS)\nacc={m_base['accuracy']:.3f} F1={m_base['f1_macro']:.3f}")
    _plot_cm(axes[1], np.array(m_retr["cm_shared"]), labels,
             f"Retrained (ESD)\nacc={m_retr['accuracy']:.3f} F1={m_retr['f1_macro']:.3f}")
    fig.suptitle(f"{dataset.upper()} ({cfg['language']}) - shared emotions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = Path(out_dir) / f"{dataset}_confusion_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(dataset: str, retrained_dir: str = RETRAINED_DIR, device: str = None):
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DATASETS[dataset]
    out_dir = OUT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  EVALUATING: {dataset.upper()} ({cfg['language']})\n{'='*60}")
    df = load_dataset(cfg)
    print(f"  Samples: {len(df)} | shared emotions: {cfg['shared']}")

    models = load_retrained_model(retrained_dir, device=device)

    paths = df["local_path"].tolist()
    y_true = df["emotion"].tolist()
    y_base, y_retr = run_inference(models, paths, cfg["shared"], device)

    m_base = compute_metrics(y_true, y_base, cfg["shared"], "Base RAVDESS model")
    m_retr = compute_metrics(y_true, y_retr, cfg["shared"], "Retrained head (ESD)")

    plot_confusions(dataset, cfg, m_base, m_retr, out_dir)

    report = {
        "dataset": dataset.upper(),
        "language": cfg["language"],
        "shared_emotions": cfg["shared"],
        "n_samples": int(len(df)),
        "retrained_dir": retrained_dir,
        "base_model": {k: v for k, v in m_base.items() if not k.startswith("cm_")},
        "retrained_model": {k: v for k, v in m_retr.items() if not k.startswith("cm_")},
    }
    out_json = out_dir / f"{dataset}_results.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: {out_json}")
    return report


def run_aesdd(**kw):
    return run("aesdd", **kw)


def run_cafe(**kw):
    return run("cafe", **kw)


def run_emodb(**kw):
    return run("emodb", **kw)


def main():
    targets = list(DATASETS) if DATASET == "all" else [DATASET]
    for name in targets:
        run(name, retrained_dir=RETRAINED_DIR)


if __name__ == "__main__":
    main()
