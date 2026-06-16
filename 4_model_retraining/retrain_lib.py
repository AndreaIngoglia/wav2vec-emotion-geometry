"""
4_model_retraining.retrain_lib - shared backend for the head-retraining notebooks
(head_retrain_language.ipynb and save_best_models.ipynb).

Holds everything the two notebooks used to copy-paste: ESD ground-truth loading +
language inference, speaker-grouped CV splits, audio I/O, the frozen-backbone
wav2vec2 forward, evaluation, and the per-fold training loop. The classifier head
is reused from common.retrained_model.LNMLPHead (single source of truth);
build_head() only re-applies the training-time Xavier init.

The notebooks stay thin: set a TrainConfig, load the GT, then call train_one().
Imports of torch/transformers happen at module load (the notebooks need them).
"""

from __future__ import annotations

import os
import sys
import json
import random
import subprocess
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
)

# --- Make `import common` work whether run locally or from a Colab copy of the
#     repo: walk up from this file until the folder containing `common/` is found.
_ROOT = next(
    (p for p in pathlib.Path(__file__).resolve().parents
     if (p / "common" / "__init__.py").exists()),
    None,
)
if _ROOT is not None and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.retrained_model import LNMLPHead, DEFAULT_MODEL, TARGET_SR  # noqa: E402
from common.labels import normalize_label  # noqa: E402

SHARED_EMOTIONS = sorted(["angry", "happy", "neutral", "sad", "surprised"])


# ============================== CONFIG ==============================

@dataclass
class TrainConfig:
    """Hyper-parameters shared by both notebooks (override per experiment)."""
    batch_size: int = 32
    chunk_rows: int = 200
    num_epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_frac: float = 0.10
    max_grad_norm: float = 1.0
    label_smooth: float = 0.05
    patience: int = 4
    proj_dim: int = 256
    hidden_dim: int = 128
    dropout: float = 0.1
    balance_classes: bool = True
    target_sr: int = TARGET_SR
    seed: int = 42


def set_seed(seed: int = 42) -> torch.device:
    """Seed python/numpy/torch and return the available device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================== DATA ==============================

def prepare_local_cache(zip_path, cache_dir, audio_root) -> pathlib.Path:
    """Populate a local ESD wav cache (Colab): unzip if needed, else use the
    Drive folder directly. Returns the directory that holds the wavs."""
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = list(cache_dir.rglob("*.wav"))
    if len(existing) > 1000:
        print(f"ESD cache: {len(existing)} files (already populated)")
        return cache_dir
    if pathlib.Path(zip_path).exists():
        print("Extracting ESD from ZIP...")
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(cache_dir)],
                       capture_output=True, text=True, check=True)
        print(f"Extracted {len(list(cache_dir.rglob('*.wav')))} files")
        return cache_dir
    print("No ZIP found, reading audio from Drive folder")
    return pathlib.Path(audio_root)


def build_path_map(audio_dir) -> Dict[str, str]:
    """Map every wav filename under ``audio_dir`` to its full path."""
    return {f.name: str(f) for f in pathlib.Path(audio_dir).rglob("*.wav")}


def _infer_language(spk: str) -> str:
    """ESD convention: speakers 1-10 = English, 11-20 = Chinese."""
    try:
        return "english" if int(spk) <= 10 else "chinese"
    except (TypeError, ValueError):
        return "unknown"


def load_esd_gt(gt_csv, path_map: Dict[str, str],
                shared_emotions: Sequence[str] = SHARED_EMOTIONS) -> pd.DataFrame:
    """Load the ESD ground truth and return a clean, audio-resolved frame.

    Normalizes emotion spelling, resolves ``speaker`` and ``language`` (inferred
    from the speaker id when absent), maps each row to a local wav path, keeps only
    shared emotions with an existing file, and adds the integer label ``y``.
    """
    df = pd.read_csv(gt_csv, sep=";")
    df["emotion"] = df["emotion"].map(normalize_label)

    if "speaker" in df.columns:
        df["speaker"] = df["speaker"].astype(str).str.strip()
    elif "speaker_id" in df.columns:
        df["speaker"] = df["speaker_id"].astype(str).str.strip()

    if "language" in df.columns:
        df["language"] = df["language"].astype(str).str.strip().str.lower()
    else:
        print("WARNING: no 'language' column - inferring from speaker id")
        df["language"] = df["speaker"].map(_infer_language)

    if "filename" in df.columns:
        df["local_path"] = df["filename"].astype(str).str.strip().map(path_map)
    elif "full_path" in df.columns:
        df["local_path"] = df["full_path"].apply(
            lambda p: path_map.get(os.path.basename(str(p).strip()), ""))

    df = df[df["emotion"].isin(list(shared_emotions))].copy()
    df = df[df["local_path"].notna() & (df["local_path"] != "")].copy()
    df = df[df["local_path"].apply(lambda p: os.path.exists(str(p)))].copy()
    df = df.reset_index(drop=True)

    label2id = {lab: i for i, lab in enumerate(sorted(shared_emotions))}
    df["y"] = df["emotion"].map(label2id).astype(int)
    return df


# ============================== SPEAKER SPLITS ==============================

def split_speakers_kfold(speakers: Sequence[str], k: int,
                         seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    """Round-robin speaker-grouped k-fold: each speaker is held out once."""
    speakers = list(speakers)
    random.Random(seed).shuffle(speakers)
    folds = [[] for _ in range(k)]
    for i, spk in enumerate(speakers):
        folds[i % k].append(spk)
    splits = []
    for i in range(k):
        val_spk = sorted(folds[i])
        train_spk = [s for s in speakers if s not in set(val_spk)]
        splits.append((train_spk, val_spk))
    return splits


def split_speakers_kfold_stratified(df: pd.DataFrame, k: int,
                                    seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    """k-fold that puts an equal number of English and Chinese speakers in each
    validation fold (round-robin within each language)."""
    rng = random.Random(seed)
    en = sorted(df[df["language"] == "english"]["speaker"].unique().tolist())
    zh = sorted(df[df["language"] == "chinese"]["speaker"].unique().tolist())
    rng.shuffle(en)
    rng.shuffle(zh)

    en_folds, zh_folds = [[] for _ in range(k)], [[] for _ in range(k)]
    for i, spk in enumerate(en):
        en_folds[i % k].append(spk)
    for i, spk in enumerate(zh):
        zh_folds[i % k].append(spk)

    splits = []
    for i in range(k):
        val_spk = sorted(en_folds[i] + zh_folds[i])
        train_spk = [s for s in en + zh if s not in set(val_spk)]
        splits.append((train_spk, val_spk))
    return splits


# ============================== AUDIO ==============================

def load_audio_mono_resample(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read a wav as mono float32 and resample (soxr_hq) to ``target_sr``."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.size == 0:
        raise RuntimeError(f"Empty audio: {path}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return wav


def chunk_rows(rows: Sequence, n: int):
    """Yield successive ``n``-sized chunks of ``rows``."""
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def load_audio_batch_sequential(paths: Sequence[str], target_sr: int = TARGET_SR):
    """Load a list of paths to tensors, skipping unreadable files. Returns
    (tensors, indices_of_paths_that_loaded)."""
    results, valid = [], []
    for i, p in enumerate(paths):
        try:
            results.append(torch.from_numpy(load_audio_mono_resample(p, target_sr)).float())
            valid.append(i)
        except Exception:
            pass
    return results, valid


# ============================== MODEL ==============================

def build_head(num_labels: int, cfg: TrainConfig, in_features: int) -> LNMLPHead:
    """Construct common.retrained_model.LNMLPHead and apply the training-time
    Xavier init (common's class ships without init since it is normally loaded
    from saved weights)."""
    head = LNMLPHead(in_features, num_labels, cfg.proj_dim, cfg.hidden_dim, cfg.dropout)
    nn.init.xavier_uniform_(head.proj.weight)
    nn.init.zeros_(head.proj.bias)
    for m in head.mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return head


class HeadOnlyModel(nn.Module):
    """Frozen wav2vec2 backbone -> mean-pool of last hidden state -> trainable head."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, **inputs):
        with torch.no_grad():
            hidden = self.backbone(**inputs).last_hidden_state
            embeddings = hidden.mean(dim=1)
        return self.head(embeddings)


def build_model(num_labels: int, cfg: TrainConfig, device,
                model_id: str = DEFAULT_MODEL) -> Tuple[HeadOnlyModel, int]:
    """Build a HeadOnlyModel (frozen backbone + fresh head). Returns (model, embed_dim)."""
    base = AutoModelForAudioClassification.from_pretrained(model_id)
    embed_dim = base.config.hidden_size
    head = build_head(num_labels, cfg, embed_dim)
    model = HeadOnlyModel(base.wav2vec2, head).to(device)
    return model, embed_dim


def make_extractor(model_id: str = DEFAULT_MODEL):
    """Load the wav2vec2 feature extractor."""
    return AutoFeatureExtractor.from_pretrained(model_id)


def forward_batch(model, extractor, waveforms, labels, device, target_sr: int = TARGET_SR):
    """Feature-extract a batch of waveforms and run the model. Returns (logits, y)."""
    inputs = extractor([w.numpy() for w in waveforms], sampling_rate=target_sr,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return model(**inputs), y


# ============================== EVAL & TRAIN ==============================

@torch.inference_mode()
def evaluate(model, rows, extractor, criterion, cfg: TrainConfig, device,
             num_labels: int) -> dict:
    """Stream ``rows`` (path, y) through the model and return accuracy / macro
    P-R-F1 / loss plus the raw y_true,y_pred (for confusion matrices)."""
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    all_true, all_pred = [], []
    buf_w, buf_y = [], []

    def flush():
        nonlocal total, correct, total_loss
        if not buf_w:
            return
        logits, y_t = forward_batch(model, extractor, buf_w, buf_y, device, cfg.target_sr)
        total_loss += criterion(logits, y_t).item() * len(buf_w)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y_t).sum().item()
        total += len(buf_w)
        all_true.extend(y_t.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
        buf_w.clear()
        buf_y.clear()

    for chunk in chunk_rows(rows, cfg.chunk_rows):
        paths = [p for p, _ in chunk]
        labels = [y for _, y in chunk]
        wavs, valid = load_audio_batch_sequential(paths, cfg.target_sr)
        for w, y in zip(wavs, (labels[i] for i in valid)):
            buf_w.append(w)
            buf_y.append(int(y))
            if len(buf_w) >= cfg.batch_size:
                flush()
    flush()

    if total == 0:
        return {"val_acc": 0.0, "val_loss": 0.0, "val_f1": 0.0, "val_prec": 0.0,
                "val_rec": 0.0, "y_true": [], "y_pred": [], "val_n": 0}
    return {
        "val_acc": correct / total,
        "val_loss": total_loss / total,
        "val_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "val_prec": precision_score(all_true, all_pred, average="macro", zero_division=0),
        "val_rec": recall_score(all_true, all_pred, average="macro", zero_division=0),
        "y_true": all_true, "y_pred": all_pred, "val_n": total,
    }


def _balanced_epoch_rows(train_rows, balance: bool):
    """Return the row order for one epoch: class-interleaved (oversampling the
    minority classes) when ``balance`` is set, else a plain shuffle."""
    if not balance:
        rows = list(train_rows)
        random.shuffle(rows)
        return rows
    by_class: Dict[int, list] = {}
    for p, y in train_rows:
        by_class.setdefault(int(y), []).append((p, int(y)))
    for c in by_class:
        random.shuffle(by_class[c])
    keys = sorted(by_class)
    max_len = max(len(by_class[c]) for c in keys)
    rows = []
    for i in range(max_len):
        for c in keys:
            if i < len(by_class[c]):
                rows.append(by_class[c][i])
    random.shuffle(rows)
    return rows


def train_one(esd_df: pd.DataFrame, train_spk, val_spk, label_names, cfg: TrainConfig,
              device, model_id: str = DEFAULT_MODEL, verbose: bool = True) -> dict:
    """Train one speaker-grouped fold and track the best-by-val-accuracy epoch.

    Returns a dict with best_val_acc / best_epoch, the best epoch's metrics
    (val_f1/prec/rec, per_class_recall, confusion_matrix), the full per-epoch
    ``history``, the best head ``state`` (CPU state_dict, for saving), and the
    train/val sample counts.
    """
    num_labels = len(label_names)
    extractor = make_extractor(model_id)

    train_df = esd_df[esd_df["speaker"].isin(train_spk)]
    val_df = esd_df[esd_df["speaker"].isin(val_spk)]
    train_rows = list(zip(train_df["local_path"].tolist(), train_df["y"].tolist()))
    val_rows = list(zip(val_df["local_path"].tolist(), val_df["y"].tolist()))

    model, embed_dim = build_model(num_labels, cfg, device, model_id)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)

    steps_per_epoch = max(1, len(train_rows) // cfg.batch_size)
    total_steps = max(1, steps_per_epoch * cfg.num_epochs)
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_acc, best_epoch, best_state, best_metrics = -1.0, 0, None, {}
    history, no_improve = [], 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss, n_seen, correct = 0.0, 0, 0
        buf_w, buf_y = [], []

        def step():
            nonlocal running_loss, n_seen, correct
            optimizer.zero_grad(set_to_none=True)
            logits, y_t = forward_batch(model, extractor, buf_w, buf_y, device, cfg.target_sr)
            loss = criterion(logits, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            bs = len(buf_w)
            running_loss += loss.item() * bs
            correct += (torch.argmax(logits, dim=-1) == y_t).sum().item()
            n_seen += bs
            buf_w.clear()
            buf_y.clear()

        for chunk in chunk_rows(_balanced_epoch_rows(train_rows, cfg.balance_classes),
                                cfg.chunk_rows):
            paths = [p for p, _ in chunk]
            labels_chunk = [y for _, y in chunk]
            wavs, valid = load_audio_batch_sequential(paths, cfg.target_sr)
            for w, y in zip(wavs, (labels_chunk[i] for i in valid)):
                buf_w.append(w)
                buf_y.append(int(y))
                if len(buf_w) >= cfg.batch_size:
                    step()
        if buf_w:
            step()
        if n_seen == 0:
            continue

        train_acc, train_loss = correct / n_seen, running_loss / n_seen
        val_m = evaluate(model, val_rows, extractor, criterion, cfg, device, num_labels)

        if val_m["y_true"]:
            pcr = recall_score(val_m["y_true"], val_m["y_pred"],
                               labels=list(range(num_labels)), average=None, zero_division=0)
            cm = confusion_matrix(val_m["y_true"], val_m["y_pred"], labels=list(range(num_labels)))
        else:
            pcr, cm = np.zeros(num_labels), np.zeros((num_labels, num_labels))
        per_class_recall = {label_names[i]: float(pcr[i]) for i in range(num_labels)}

        if verbose:
            print(f"  Epoch {epoch}/{cfg.num_epochs}: train_acc={train_acc:.4f} "
                  f"loss={train_loss:.4f} | val_acc={val_m['val_acc']:.4f} "
                  f"f1={val_m['val_f1']:.4f} prec={val_m['val_prec']:.4f} "
                  f"rec={val_m['val_rec']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")
            print(f"    Recall: { {k: f'{v:.3f}' for k, v in per_class_recall.items()} }")

        history.append({
            "epoch": epoch, "train_acc": float(train_acc), "train_loss": float(train_loss),
            "val_acc": float(val_m["val_acc"]), "val_loss": float(val_m["val_loss"]),
            "val_f1": float(val_m["val_f1"]), "val_prec": float(val_m["val_prec"]),
            "val_rec": float(val_m["val_rec"]), "per_class_recall": per_class_recall,
            "confusion_matrix": cm.tolist(), "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if val_m["val_acc"] > best_acc:
            best_acc, best_epoch, no_improve = float(val_m["val_acc"]), epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}
            best_metrics = {
                "val_acc": float(val_m["val_acc"]), "val_f1": float(val_m["val_f1"]),
                "val_prec": float(val_m["val_prec"]), "val_rec": float(val_m["val_rec"]),
                "epoch": epoch, "confusion_matrix": cm.tolist(),
                "per_class_recall": per_class_recall,
            }
            if verbose:
                print(f"    >>> New best! val_acc={best_acc:.4f}")
        else:
            no_improve += 1
            if verbose:
                print(f"    No improvement ({no_improve}/{cfg.patience})")
            if no_improve >= cfg.patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_val_acc": best_acc, "best_epoch": best_epoch, "best_metrics": best_metrics,
        "history": history, "best_state": best_state, "embed_dim": embed_dim,
        "n_train": len(train_rows), "n_val": len(val_rows),
    }
