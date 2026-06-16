"""
STEP 2.1 - Subspace angles / overlap (ESD / RAVDESS)

From the cached probe weights, measures how entangled the emotion directions are
with each nuisance: row-wise angles, principal angles between the subspaces, and
scalar overlap measures (smallest angle, mean cos^2, Frobenius affinity).

Output: 2_geometric_diagnostics/1_subspace_entanglement/output/
<DATASET>_angles_overlap_summary.json
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

import numpy as np

from common.geometry import (
    angle_degrees_from_cos, cosine_rowwise, principal_angles_degrees,
    row_center, subspace_overlap_measures,
)
from common.probes import ensure_exists, load_probe_npz
from common.paths import output_dir

OUT = output_dir(__file__)

# ============================= CONFIGURATION =============================
# Which dataset to analyze: "esd", "ravdess", "all".
DATASET = "all"

# Probe caches produced by STEP 2.0 (train_probes.py):
ESD_PROBE_DIR = "xai_weights/ESD"
RAVDESS_PROBE_DIR = "xai_weights/RAVDESS"
# ========================================================================



def _classes_list(probe):
    return list(probe["classes"]) if probe.get("classes") is not None else None


def analyze_esd(probe_dir=ESD_PROBE_DIR, out_json=None, top_k=5, center_multiclass=True):
    if out_json is None:
        out_json = str(OUT / "ESD_angles_overlap_summary.json")

    p_em = os.path.join(probe_dir, "emotion_probe.npz")
    p_lang = os.path.join(probe_dir, "language_probe.npz")
    p_gen = os.path.join(probe_dir, "gender_probe.npz")
    p_spk = os.path.join(probe_dir, "speaker_probe.npz")
    for p in (p_em, p_lang, p_gen, p_spk):
        ensure_exists(p)

    emotion = load_probe_npz(p_em)
    W_em, cls_em = emotion["W"], _classes_list(emotion)
    W_lang = load_probe_npz(p_lang)["W"]
    W_gen = load_probe_npz(p_gen)["W"]
    W_spk = load_probe_npz(p_spk)["W"]

    w_lang = W_lang.reshape(-1) if W_lang.shape[0] == 1 else W_lang.mean(axis=0)
    w_gen = W_gen.reshape(-1) if W_gen.shape[0] == 1 else W_gen.mean(axis=0)
    W_em_span = row_center(W_em) if center_multiclass else W_em
    W_spk_span = row_center(W_spk) if center_multiclass else W_spk

    cos_em_lang = cosine_rowwise(W_em, w_lang)
    cos_em_gen = cosine_rowwise(W_em, w_gen)
    ang_em_lang = angle_degrees_from_cos(cos_em_lang)
    ang_em_gen = angle_degrees_from_cos(cos_em_gen)
    idx_lang = np.argsort(-np.abs(cos_em_lang))[:top_k]
    idx_gen = np.argsort(-np.abs(cos_em_gen))[:top_k]

    def _label(i):
        return str(i) if cls_em is None else cls_em[i]

    top_lang = [{"emotion": _label(i), "cosine": float(cos_em_lang[i]), "angle_deg": float(ang_em_lang[i])} for i in idx_lang]
    top_gen = [{"emotion": _label(i), "cosine": float(cos_em_gen[i]), "angle_deg": float(ang_em_gen[i])} for i in idx_gen]

    ang_sub_em_lang = principal_angles_degrees(W_em_span, w_lang.reshape(1, -1))
    ang_sub_em_gen = principal_angles_degrees(W_em_span, w_gen.reshape(1, -1))
    ang_sub_em_spk = principal_angles_degrees(W_em_span, W_spk_span)
    overlap_em_lang = subspace_overlap_measures(W_em_span, w_lang.reshape(1, -1))
    overlap_em_gen = subspace_overlap_measures(W_em_span, w_gen.reshape(1, -1))
    overlap_em_spk = subspace_overlap_measures(W_em_span, W_spk_span)

    print(f"\n=== ANGLES / OVERLAP — ESD ===  (dim={W_em.shape[1]}, classes={len(cls_em) if cls_em else W_em.shape[0]})")
    if overlap_em_spk["smallest_angle_deg"] is not None:
        print(f"Emotion vs Speaker: smallest angle={overlap_em_spk['smallest_angle_deg']:.2f}deg affinity={overlap_em_spk['affinity']:.4f}")

    summary = {
        "probe_dir": probe_dir, "embedding_dim": int(W_em.shape[1]),
        "center_multiclass_spans": bool(center_multiclass),
        "emotion_classes": cls_em, "top_k": int(top_k),
        "rowwise": {"emotion_vs_language": top_lang, "emotion_vs_gender": top_gen},
        "subspace": {
            "principal_angles_emotion_vs_language_deg": [float(x) for x in ang_sub_em_lang.tolist()],
            "principal_angles_emotion_vs_gender_deg": [float(x) for x in ang_sub_em_gen.tolist()],
            "principal_angles_emotion_vs_speaker_deg": [float(x) for x in ang_sub_em_spk.tolist()],
            "smallest_angles_emotion_vs_speaker_deg": [float(x) for x in ang_sub_em_spk[:min(5, len(ang_sub_em_spk))]],
            "overlap_emotion_vs_language": overlap_em_lang,
            "overlap_emotion_vs_gender": overlap_em_gen,
            "overlap_emotion_vs_speaker": overlap_em_spk,
        },
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {out_json}")
    return summary


def analyze_ravdess(probe_dir=RAVDESS_PROBE_DIR, out_json=None, top_k=8, center_multiclass=True):
    if out_json is None:
        out_json = str(OUT / "RAVDESS_angles_overlap_summary.json")

    p_em = os.path.join(probe_dir, "emotion_probe.npz")
    p_gen = os.path.join(probe_dir, "gender_probe.npz")
    p_act = os.path.join(probe_dir, "actor_id_probe.npz")
    if not os.path.exists(p_act):
        p_act = os.path.join(probe_dir, "speaker_probe.npz")
    for p in (p_em, p_gen, p_act):
        ensure_exists(p)

    emotion = load_probe_npz(p_em)
    W_em, cls_em = emotion["W"], _classes_list(emotion)
    W_gen = load_probe_npz(p_gen)["W"]
    W_act = load_probe_npz(p_act)["W"]

    w_gen = W_gen.reshape(-1) if W_gen.shape[0] == 1 else W_gen.mean(axis=0)
    W_em_span = row_center(W_em) if center_multiclass else W_em
    W_act_span = row_center(W_act) if center_multiclass else W_act

    def _label(i):
        return str(i) if cls_em is None else cls_em[i]

    cos_em_gen = cosine_rowwise(W_em, w_gen)
    ang_em_gen = angle_degrees_from_cos(cos_em_gen)
    idx_gen = np.argsort(-np.abs(cos_em_gen))[:top_k]
    top_gen = [{"emotion": _label(i), "cosine": float(cos_em_gen[i]), "angle_deg": float(ang_em_gen[i])} for i in idx_gen]
    all_gen = [{"emotion": _label(i), "cosine": float(cos_em_gen[i]), "angle_deg": float(ang_em_gen[i])} for i in range(len(cos_em_gen))]

    ang_sub_em_gen = principal_angles_degrees(W_em_span, w_gen.reshape(1, -1))
    overlap_em_gen = subspace_overlap_measures(W_em_span, w_gen.reshape(1, -1))
    ang_sub_em_act = principal_angles_degrees(W_em_span, W_act_span)
    overlap_em_act = subspace_overlap_measures(W_em_span, W_act_span)
    ang_sub_gen_act = principal_angles_degrees(w_gen.reshape(1, -1), W_act_span)
    overlap_gen_act = subspace_overlap_measures(w_gen.reshape(1, -1), W_act_span)

    print(f"\n=== ANGLES / OVERLAP — RAVDESS ===  (dim={W_em.shape[1]}, classes={cls_em if cls_em else W_em.shape[0]})")
    if overlap_em_act["smallest_angle_deg"] is not None:
        print(f"Emotion vs Actor: smallest angle={overlap_em_act['smallest_angle_deg']:.2f}deg affinity={overlap_em_act['affinity']:.4f}")

    summary = {
        "probe_dir": probe_dir, "embedding_dim": int(W_em.shape[1]),
        "center_multiclass_spans": bool(center_multiclass),
        "emotion_classes": cls_em, "top_k": int(top_k),
        "rowwise": {"emotion_vs_gender": top_gen, "all_emotion_vs_gender": all_gen},
        "subspace": {
            "principal_angles_emotion_vs_gender_deg": [float(x) for x in ang_sub_em_gen.tolist()],
            "principal_angles_emotion_vs_actor_deg": [float(x) for x in ang_sub_em_act.tolist()],
            "smallest_angles_emotion_vs_actor_deg": [float(x) for x in ang_sub_em_act[:min(5, len(ang_sub_em_act))]],
            "principal_angles_gender_vs_actor_deg": [float(x) for x in ang_sub_gen_act.tolist()],
            "overlap_emotion_vs_gender": overlap_em_gen,
            "overlap_emotion_vs_actor": overlap_em_act,
            "overlap_gender_vs_actor": overlap_gen_act,
        },
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {out_json}")
    return summary


ORCHESTRATORS = {"esd": analyze_esd, "ravdess": analyze_ravdess}


def main():
    for name in (list(ORCHESTRATORS) if DATASET == "all" else [DATASET]):
        ORCHESTRATORS[name]()


if __name__ == "__main__":
    main()
