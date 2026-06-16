"""
STEP 5 - ESD baseline evaluation (HuggingFace predictions)

Scores the base HuggingFace classifier on ESD: joins its predictions with the
ground truth on filename and computes accuracy/precision/recall/F1, a
classification report and a confusion matrix.

Output: 5_evaluation/output/esd_confusion_matrix.csv (+ metrics to stdout)
"""

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Make the shared package importable when launched from the repository root.
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

from common.paths import output_dir

OUT = output_dir(__file__)


# ============================== CONFIGURATION ==============================
ESD_GT = "data/ESD/ESD_GT.csv"                 # ground truth
ESD_PRED = "data/ESD/ESD_predictions.csv"      # base-model predictions to score
# ==========================================================================

esd_gt = pd.read_csv(ESD_GT, sep=";")[["filename", "language", "gender", "speaker", "emotion"]]

# mappping label GT to label PRED
emotion_mapping = {
	"Neutral": "neutral",
	"Happy": "happy",
	"Angry": "angry",
	"Sad": "sad",
	"Surprise": "surprised"
}

esd_gt["emotion"] = esd_gt["emotion"].map(emotion_mapping)

esd_pred = pd.read_csv(ESD_PRED, sep=";")[["filename", "predicted_emotion"]]

print(esd_pred.groupby(["predicted_emotion"]).count())

df = esd_gt.merge(esd_pred, on="filename")


df2 = esd_pred.merge(esd_gt, on="filename")
df_unmatched = df2[df2["emotion"].isna() | df2["predicted_emotion"].isna()]

print(f"Unmatched rows: {len(df_unmatched)}")
df_unmatched.head(20)


# --- Metrics (rows with both GT + prediction) ---
df_eval = df.dropna(subset=["emotion", "predicted_emotion"]).copy()
if df_eval.empty:
    raise RuntimeError(
        "No rows had both ground-truth emotion and predicted_emotion. "
        "Check that GT 'filename' matches prediction filenames."
    )

y_true = df_eval["emotion"].astype(str).tolist()
y_pred = df_eval["predicted_emotion"].astype(str).tolist()

# Keep a consistent label order (recommended)
labels = ["neutral", "happy", "angry", "sad", "surprised"]

# If you prefer automatic labels:
# labels = sorted(set(y_true) | set(y_pred))

cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{l}" for l in labels],
    columns=[f"pred_{l}" for l in labels],
)

CONFUSION_CSV = str(OUT / "esd_confusion_matrix.csv")
cm_df.to_csv(CONFUSION_CSV, sep=";", index=False)
print(f"Saved confusion matrix to: {CONFUSION_CSV}")

print("\n=== Summary Metrics (matched rows only) ===")
print(f"Matched rows: {len(df_eval)} / total merged: {len(df)}")
print(f"Accuracy     : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision(m) : {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
print(f"Recall(m)    : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
print(f"F1(macro)    : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))