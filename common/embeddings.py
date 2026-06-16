"""
common.embeddings - wav2vec2 feature extraction: turn audio files into pooled
embeddings and save them as a per-dataset .npz cache.

Default model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition with the
facebook/wav2vec2-large-xlsr-53 feature extractor. Embeddings are the masked
mean+std pooling of the average of the last avg_last_n hidden states (2048-d).
Heavy deps (torch/transformers) are imported lazily so importing this is cheap.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

DEFAULT_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
DEFAULT_PROCESSOR = "facebook/wav2vec2-large-xlsr-53"
TARGET_SR = 16000


def load_audio(path: str, target_sr: int = TARGET_SR):
    """Load an audio file as mono float32 numpy at ``target_sr``.

    Tries torchaudio first (used for the .wav datasets); falls back to
    soundfile + librosa, which also reads formats torchaudio may choke on
    (e.g. the .aiff files in CaFE).
    """
    import numpy as np

    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0) if wav.size(0) > 1 else wav.squeeze(0)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.numpy().astype(np.float32)
    except Exception:
        import soundfile as sf
        import librosa

        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return np.asarray(wav, dtype=np.float32)


def pool_features(x, mask):
    """Masked mean+std pooling over time. ``x``: [B,T,D], ``mask``: [B,T] -> [B,2D]."""
    import torch

    mask = mask.unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp(min=1.0)
    mean = (x * mask).sum(dim=1) / denom
    var = ((x - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / denom
    std = torch.sqrt(var.clamp(min=1e-8))
    return torch.cat([mean, std], dim=-1)


def downsample_attention_mask(attention_mask, T: int):
    """Map an input attention mask [B, L] to a feature-frame mask [B, T]."""
    import torch

    B, L = attention_mask.shape
    device = attention_mask.device
    idx = torch.linspace(0, L, steps=T + 1, device=device).long()
    frame_mask = torch.zeros((B, T), device=device, dtype=torch.float32)
    for t in range(T):
        seg = attention_mask[:, idx[t]: idx[t + 1]]
        frame_mask[:, t] = (seg.sum(dim=1) > 0).float()
    return frame_mask


def extract_embeddings_from_df(
    df: pd.DataFrame,
    output_path: str,
    path_col: str = "full_path",
    label_col: str = "emotion",
    speaker: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    processor_name: str = DEFAULT_PROCESSOR,
    batch_size: int = 8,
    device: Optional[str] = None,
    max_seconds: Optional[float] = None,
    avg_last_n: int = 4,
    extra_npz: Optional[dict] = None,
):
    """Compute pooled embeddings for every row of ``df`` and save them to
    ``output_path`` as a compressed ``.npz`` (keys: ``embs``, ``labels``,
    ``paths`` + any ``extra_npz``).

    ``df`` must already be filtered to the desired subset (e.g. one speaker) and
    contain ``path_col`` and ``label_col``.
    """
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if path_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"df must contain '{path_col}' and '{label_col}' columns")

    tag = f"speaker {speaker}" if speaker is not None else f"{len(df)} files"
    print(f"Computing embeddings for {tag}  (N={len(df)})")
    if len(df) == 0:
        print(f"[WARN] No rows for {tag}. Skipping.")
        return None

    print(f"Loading feature extractor from: {processor_name}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(processor_name)
    print(f"Loading model from: {model_name}")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"Model hidden size: {model.config.hidden_size}")

    waves: List[np.ndarray] = []
    labels: List = []
    paths: List = []
    for _, row in df.iterrows():
        wav = load_audio(row[path_col])
        if max_seconds is not None:
            wav = wav[: int(TARGET_SR * max_seconds)]
        waves.append(wav)
        labels.append(row[label_col])
        paths.append(row[path_col])

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(waves), batch_size):
            batch = waves[i: i + batch_size]
            inputs = processor(batch, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            input_values = inputs["input_values"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            out = model(
                input_values, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True,
            )

            if avg_last_n > 0:
                x = torch.stack(out.hidden_states[-avg_last_n:], dim=0).mean(dim=0)
            else:
                x = out.last_hidden_state

            B, T, _ = x.shape
            if attention_mask is None:
                mask = torch.ones((B, T), device=device)
            else:
                mask = downsample_attention_mask(attention_mask, T)

            emb = pool_features(x, mask).cpu().numpy()
            all_embs.append(emb)
            print(f"  {min(i + batch_size, len(waves))}/{len(waves)}")

    embs = np.vstack(all_embs)
    labels_arr = np.array(labels, dtype=object)
    paths_arr = np.array(paths, dtype=object)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = dict(
        embs=embs, labels=labels_arr, paths=paths_arr,
        model=model_name, processor=processor_name, avg_last_n=avg_last_n,
    )
    if speaker is not None:
        payload["speaker"] = speaker
    if extra_npz:
        payload.update(extra_npz)

    np.savez_compressed(output_path, **payload)
    print("Saved:", os.path.abspath(output_path), "shape:", embs.shape)
    return {"embs": embs, "labels": labels_arr, "paths": paths_arr}
