"""
STEP 0 - wav2vec2 embedding extraction (all datasets)

Turns raw audio into pooled wav2vec2 embeddings (frozen backbone, masked
mean+std pooling of the last AVG_LAST_N hidden states -> 2048-d) and caches them
per dataset. Audio is located by filename via a recursive scan of the dataset
folder.

Output: data/<DATASET>/wav2vec2-.../*.npz -> embedding caches read by every
later step.
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

from pathlib import Path

import pandas as pd

from common.embeddings import extract_embeddings_from_df

# ============================== CONFIGURATION ==============================
# Which dataset(s) to extract: "esd", "ravdess", "aesdd", "cafe", "emodb", "all".
DATASET = "all"

# wav2vec2 model used to produce the embeddings.
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
PROCESSOR_NAME = "facebook/wav2vec2-large-xlsr-53"
AVG_LAST_N = 1   # number of final hidden states averaged before pooling

# Per-dataset input/output paths (all relative to the project root).
#   gt_csv             ground-truth CSV
#   audio_root         folder scanned RECURSIVELY for audio (matched to the GT filename)
#   audio_exts         audio file extensions to look for
#   emb_dir            where the embedding .npz cache is written
#   batch_size         inference batch size
#   normalize_emotion  remap foreign spellings (AESDD/CaFE/EMODB) to canonical labels
PATHS = {
    "esd": dict(
        gt_csv="data/ESD/ESD_GT.csv",
        audio_root="data/ESD/ESD_Dataset",
        audio_exts=(".wav",),
        emb_dir="data/ESD/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        batch_size=4, normalize_emotion=False,
    ),
    "ravdess": dict(
        gt_csv="data/RAVDESS/RAVDESS_GT.csv",
        audio_root="data/RAVDESS/RAVDESS_Dataset",
        audio_exts=(".wav",),
        emb_dir="data/RAVDESS/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        batch_size=4, normalize_emotion=False,
    ),
    "aesdd": dict(
        gt_csv="data/AESDD/AESDD_GT.csv",
        audio_root="data/AESDD/AESDD_Dataset",
        audio_exts=(".wav",),
        emb_dir="data/AESDD/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        batch_size=8, normalize_emotion=True,
    ),
    "cafe": dict(
        gt_csv="data/CaFE/CaFE_GT.csv",
        audio_root="data/CaFE/CaFe_Dataset",
        audio_exts=(".aiff", ".aif", ".wav"),
        emb_dir="data/CaFE/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        batch_size=8, normalize_emotion=True,
    ),
    "emodb": dict(
        gt_csv="data/EMODB/EMODB_GT.csv",
        audio_root="data/EMODB/EMODB_Dataset",
        audio_exts=(".wav",),
        emb_dir="data/EMODB/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        batch_size=8, normalize_emotion=True,
    ),
}

# ESD speaker ids and RAVDESS actor ids to iterate over.
ESD_SPEAKERS = [str(s).zfill(4) for s in range(1, 21)]
RAVDESS_ACTORS = list(range(1, 25))
# RAVDESS: keep only audio-only (modality "03"), speech ("01") rows.
RAVDESS_ONLY_MODALITY_ID = "03"
RAVDESS_ONLY_VOCAL_ID = "01"

# Map foreign emotion spellings onto the project's canonical labels.
EMOTION_REMAP = {
    "anger": "angry", "happiness": "happy", "joy": "happy",
    "sadness": "sad", "surprise": "surprised",
}
# ==========================================================================


def _norm_emotion(value):
    v = str(value).strip().lower()
    return EMOTION_REMAP.get(v, v)


def _build_path_map(audio_root, exts):
    """Map basename -> absolute path for every audio file under audio_root."""
    root = Path(audio_root)
    exts = tuple(e.lower() for e in exts)
    path_map = {}
    if root.exists():
        for f in root.rglob("*"):
            if f.suffix.lower() in exts:
                path_map.setdefault(f.name, str(f))
    return path_map


def _resolve_by_name(df, cfg):
    """Add a __path column by matching the GT filename to a real audio file."""
    if "filename" not in df.columns:
        raise ValueError(f"GT must contain a 'filename' column; found {list(df.columns)}")
    path_map = _build_path_map(cfg["audio_root"], cfg["audio_exts"])
    if not path_map:
        raise FileNotFoundError(
            f"No audio found under {cfg['audio_root']}/. Place the dataset audio there first."
        )
    df = df.copy()
    df["__path"] = df["filename"].astype(str).str.strip().apply(
        lambda fn: path_map.get(Path(fn).name, "")
    )
    missing = int((df["__path"] == "").sum())
    if missing:
        print(f"[WARN] {missing} rows had no matching audio file (skipped).")
    return df[df["__path"] != ""].reset_index(drop=True)


def _extract(df, cfg, output_path, speaker=None):
    extract_embeddings_from_df(
        df, output_path=output_path, path_col="__path", label_col="emotion", speaker=speaker,
        model_name=MODEL_NAME, processor_name=PROCESSOR_NAME,
        batch_size=cfg["batch_size"], avg_last_n=AVG_LAST_N,
        extra_npz=df.attrs.get("extra_npz"),
    )


# ---------------------------------------------------------------------------
# Per-speaker datasets (ESD, RAVDESS)
# ---------------------------------------------------------------------------
def run_esd():
    """ESD: one .npz per speaker; audio resolved by filename."""
    cfg = PATHS["esd"]
    # STEP 1: load GT, resolve audio paths by name, tag speaker
    df = pd.read_csv(cfg["gt_csv"], sep=";")
    df["speaker_str"] = df["speaker"].astype(str).str.zfill(4)
    df = _resolve_by_name(df, cfg)
    # STEP 2: extract one speaker at a time
    for spk in ESD_SPEAKERS:
        sub = df[df["speaker_str"] == spk].reset_index(drop=True)
        _extract(sub, cfg, output_path=f"{cfg['emb_dir']}/{spk}_embeddings.npz", speaker=spk)


def run_ravdess():
    """RAVDESS: one .npz per actor; audio resolved by filename."""
    cfg = PATHS["ravdess"]
    # STEP 1: load GT, resolve audio paths by name
    df = pd.read_csv(cfg["gt_csv"], sep=";")
    if not {"filename", "emotion", "actor_id"}.issubset(df.columns):
        raise ValueError(f"RAVDESS GT must contain filename/emotion/actor_id; found {list(df.columns)}")
    df = _resolve_by_name(df, cfg)
    # STEP 2: extract one actor at a time (after modality/vocal filtering)
    for actor in RAVDESS_ACTORS:
        actor_2d = f"{actor:02d}"
        sub = df[df["actor_id"].astype(str).str.zfill(2) == actor_2d].copy()
        if RAVDESS_ONLY_MODALITY_ID is not None and "modality_id" in sub.columns:
            sub = sub[sub["modality_id"].astype(str).str.zfill(2) == RAVDESS_ONLY_MODALITY_ID]
        if RAVDESS_ONLY_VOCAL_ID is not None and "vocal_channel_id" in sub.columns:
            sub = sub[sub["vocal_channel_id"].astype(str).str.zfill(2) == RAVDESS_ONLY_VOCAL_ID]
        sub = sub.reset_index(drop=True)
        sub.attrs["extra_npz"] = {"actor_id": actor_2d}
        _extract(sub, cfg, output_path=f"{cfg['emb_dir']}/{actor:04d}_embeddings.npz",
                 speaker=f"{actor:04d}")


# ---------------------------------------------------------------------------
# Whole-dataset extraction (AESDD, CaFE, EMODB)
# ---------------------------------------------------------------------------
def _run_whole(name):
    """Extract one all_embeddings.npz for a whole dataset."""
    cfg = PATHS[name]
    # STEP 1: load GT + normalize emotion labels
    df = pd.read_csv(cfg["gt_csv"], sep=";")
    if cfg.get("normalize_emotion"):
        df["emotion"] = df["emotion"].apply(_norm_emotion)
    # STEP 2: resolve audio paths by name
    df = _resolve_by_name(df, cfg)
    # STEP 3: extract the whole dataset into a single cache file
    _extract(df, cfg, output_path=f"{cfg['emb_dir']}/all_embeddings.npz", speaker=None)


def run_aesdd():
    _run_whole("aesdd")


def run_cafe():
    _run_whole("cafe")


def run_emodb():
    _run_whole("emodb")


ORCHESTRATORS = {
    #"esd": run_esd, 
    #"ravdess": run_ravdess,
    "aesdd": run_aesdd, 
    #"cafe": run_cafe, 
    #"emodb": run_emodb,
}


def main():
    targets = list(ORCHESTRATORS) if DATASET == "all" else [DATASET]
    for name in targets:
        print(f"\n{'='*60}\n  EXTRACTING: {name.upper()}\n{'='*60}")
        ORCHESTRATORS[name]()


if __name__ == "__main__":
    main()
