# A GEOMETRIC DIAGNOSIS OF CROSS-CORPUS FAILURE IN SPEECH EMOTION RECOGNITION

A geometric analysis of emotion encoding in a frozen wav2vec 2.0 speech-emotion model. Using linear probing, subspace geometry, acoustic features, Procrustes alignment, ablation, and a cross-lingual evaluation of a retrained head, it locates where cross-corpus transfer fails and shows the failure is a rotational misalignment of the classifier, not a loss of emotional information in the encoder.

Backbone: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (frozen).
Embeddings = masked **mean+std pooling** of the average of the last 4 hidden
states → a **2048-d** vector per utterance.

## Pipeline

| step | script(s) | reads | writes | answers |
|---|---|---|---|---|
| 0 | `0_embeddings/extract_embeddings.py` | audio + GT | `data/<DS>/wav2vec2-.../*.npz` | audio -> 2048-d embeddings |
| 1 | `1_classical_features/*` | audio / eGeMAPS / GT | `ESD_eGeMAPS.csv`, plots | which acoustic features separate emotions |
| 2.0 | `0_linear_probes/train_probes.py` | embeddings + GT | `xai_weights/<DS>/*_probe.npz` | linear directions for emotion & nuisances |
| 2.1 | `1_subspace_entanglement/subspace_angle.py` | probes | angles JSON | is emotion tangled with nuisances |
| 2.3 | `2_pca_procrustes/*` | probes / embeddings | PNG + JSON | align directions via Procrustes; rotation test |
| 2.4 | `3_ablation/ablation_evaluation.py` | embeddings + probes | csv + JSON + npz | does emotion survive removing nuisances |
| 3 | `3_local_evaluation/tsne_knn_eval.py` | embeddings | t-SNE PNGs | separability by emotion / gender / nationality |
| 4 | `4_model_retraining/*.ipynb` | audio | `saved_models/<cond>/` | retrain an MLP head on ESD |
| 5 | `5_evaluation/*` | predictions / audio + head | metrics JSON + PNG | base vs retrained, in-domain & cross-lingual |

## Datasets

| dataset | role | language | emotions used |
|---|---|---|---|
| ESD | main / source | English + Mandarin | angry, happy, neutral, sad, surprised |
| RAVDESS | secondary / source | English (24 actors) | angry, happy, neutral, sad, surprised |
| AESDD | cross-lingual test | Greek | angry, happy, sad |
| CaFE | cross-lingual test | French | angry, happy, neutral, sad, surprised |
| EMODB | cross-lingual test | German | angry, happy, neutral, sad |

Audio is not committed: place each dataset's files under `data/<DATASET>/`
(ESD/RAVDESS resolve audio from their GT path column, the others by `filename`).

**Retrained head** (`4_model_retraining/save_best_models.ipynb`): frozen backbone +
`LayerNorm -> Linear(1024->256) -> LayerNorm -> GELU/Dropout -> Linear(256->128)
-> GELU/Dropout -> Linear(128->5)`, trained on mean-pooled ESD embeddings per
language condition (english / chinese / all). Step 5 loads it via
`common.retrained_model` and compares against the base model.

## Setup

```bash
pip install -r requirements.txt
pip install -e .            # optional: clean `import common` from anywhere
```
