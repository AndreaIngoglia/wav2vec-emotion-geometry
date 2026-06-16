"""
STEP 3 - t-SNE + k-NN embedding evaluation

For a chosen label, standardizes the embeddings, runs a (speaker-grouped) k-NN
probe to measure separability and saves a 2-D t-SNE scatter. Pick the analysis
with the ANALYSIS variable (esd_per_speaker, esd_emotion, ravdess_emotion,
esd_gender, esd_nationality, esd_nationality_gender).

Output: 3_local_evaluation/output/*.png -> t-SNE scatter(s) annotated with the
k-NN accuracy.
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

import numpy as np

from common.io import load_embeddings_labels, load_embeddings_only
from common.paths import output_dir
from common.viz import run_tsne_knn, scatter_by_label

# ============================= CONFIGURATION =============================
# Which analysis to run, or "all":
#   esd_per_speaker | esd_emotion | ravdess_emotion |
#   esd_gender | esd_nationality | esd_nationality_gender | all
ANALYSIS = "esd_per_speaker"
# ========================================================================

ESD_NPZ = "data/ESD/wav2vec2-lg-xlsr-en-speech-emotion-recognition/{speaker}_embeddings.npz"
RAV_NPZ = "data/RAVDESS/wav2vec2-lg-xlsr-en-speech-emotion-recognition/{speaker}_embeddings.npz"
ESD_SPEAKERS = [str(s).zfill(4) for s in range(1, 21)]
RAV_ACTORS = [str(s).zfill(4) for s in range(1, 25)]

FEMALE_SPEAKERS = ["0001", "0002", "0003", "0007", "0009", "0015", "0016", "0017", "0018"]
MALE_SPEAKERS = ["0004", "0005", "0006", "0008", "0010", "0011", "0012", "0013", "0014", "0019", "0020"]
CHINESE_SPEAKERS = [str(s).zfill(4) for s in range(1, 11)]
AMERICAN_SPEAKERS = [str(s).zfill(4) for s in range(11, 21)]

OUT = output_dir(__file__)


def _gender_labels(spk_ids):
    m = {s: "Femmina" for s in FEMALE_SPEAKERS}
    m.update({s: "Maschio" for s in MALE_SPEAKERS})
    return np.array([m.get(s, "Unknown") for s in spk_ids], dtype=object)


def _nationality_labels(spk_ids):
    m = {s: "Chinese" for s in CHINESE_SPEAKERS}
    m.update({s: "American" for s in AMERICAN_SPEAKERS})
    return np.array([m.get(s, "Unknown") for s in spk_ids], dtype=object)


def _nat_gender_labels(spk_ids):
    female, male = set(FEMALE_SPEAKERS), set(MALE_SPEAKERS)
    chinese, english = set(CHINESE_SPEAKERS), set(AMERICAN_SPEAKERS)
    out = []
    for s in spk_ids:
        g = "Female" if s in female else ("Male" if s in male else None)
        n = "Chinese" if s in chinese else ("English" if s in english else None)
        out.append(f"{n}_{g}" if (g and n) else "Unknown")
    return np.array(out, dtype=object)


ANALYSES = {
    "esd_per_speaker": dict(kind="per_speaker", npz=ESD_NPZ, speakers=ESD_SPEAKERS,
                            knn_k=15, legend="Emotion"),
    "esd_emotion": dict(kind="unified", npz=ESD_NPZ, speakers=ESD_SPEAKERS, label="emotion",
                        knn_k=15, legend="Emotion", out="ALL_speakers_tsne_knn.png",
                        title="Unified t-SNE (All Speakers)"),
    "ravdess_emotion": dict(kind="unified", npz=RAV_NPZ, speakers=RAV_ACTORS, label="emotion",
                            knn_k=15, legend="Emotion", out="ALL_actors_tsne_knn.png",
                            title="Unified t-SNE (RAVDESS: All Actors)"),
    "esd_gender": dict(kind="unified", npz=ESD_NPZ, speakers=ESD_SPEAKERS, label=_gender_labels,
                       knn_k=15, legend="Genere", out="ALL_speakers_tsne_knn_gender.png",
                       title="Unified t-SNE (All Speakers) - Genere"),
    "esd_nationality": dict(kind="unified", npz=ESD_NPZ, speakers=ESD_SPEAKERS, label=_nationality_labels,
                            knn_k=15, legend="Nationality", out="ALL_speakers_tsne_knn_nationality.png",
                            title="Unified t-SNE (All Speakers) - Nationality"),
    "esd_nationality_gender": dict(kind="unified", npz=ESD_NPZ, speakers=ESD_SPEAKERS, label=_nat_gender_labels,
                                   knn_k=15, legend="Class", out="ALL_speakers_tsne_knn_nat_gender.png",
                                   title="Unified t-SNE - Nationality x Gender"),
}


def run(name):
    cfg = ANALYSES[name]

    # STEP 1 (per-speaker): one plot per ESD speaker
    if cfg["kind"] == "per_speaker":
        for spk in cfg["speakers"]:
            embs, labels, _ = load_embeddings_labels(cfg["npz"], [spk])
            res = run_tsne_knn(embs, labels, groups=None, knn_k=cfg["knn_k"], group_cv=False)
            scatter_by_label(
                res["X_tsne"], labels,
                title=(f"t-SNE - Speaker {spk}\n"
                       f"k-NN(k={cfg['knn_k']}) = {res['knn_mean']*100:.2f}% ± {res['knn_std']*100:.2f}%"),
                out_path=OUT / f"{spk}_tsne_knn.png",
                legend_title=cfg["legend"], figsize=(8, 6), point_size=20, alpha=0.6,
            )
        return

    # STEP 1 (unified): build the embedding matrix + the labels to color by
    if cfg["label"] == "emotion":
        embs, labels, groups = load_embeddings_labels(cfg["npz"], cfg["speakers"])
    else:
        embs, groups = load_embeddings_only(cfg["npz"], cfg["speakers"])
        labels = cfg["label"](groups)
        keep = labels != "Unknown"
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"[WARN] Dropping {n_drop} samples without a label.")
        embs, groups, labels = embs[keep], groups[keep], labels[keep]

    # STEP 2: k-NN probe + t-SNE + scatter
    res = run_tsne_knn(embs, labels, groups=groups, knn_k=cfg["knn_k"], group_cv=True)
    title = (f"{cfg['title']}\n"
             f"k-NN(k={cfg['knn_k']}) = {res['knn_mean']*100:.2f}% ± {res['knn_std']*100:.2f}%")
    scatter_by_label(res["X_tsne"], labels, title=title,
                     out_path=OUT / cfg["out"], legend_title=cfg["legend"])


def main():
    for name in (list(ANALYSES) if ANALYSIS == "all" else [ANALYSIS]):
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        run(name)


if __name__ == "__main__":
    main()
