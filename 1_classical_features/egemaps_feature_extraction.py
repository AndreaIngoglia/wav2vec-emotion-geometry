"""
STEP 1 - eGeMAPS feature extraction (ESD)

Extracts the 88 classical eGeMAPSv02 acoustic functionals (openSMILE) from every
ESD utterance.

Output: data/ESD/ESD_eGeMAPS.csv -> one row per file (feature cache for the
other classical-feature scripts).
"""

import os

import opensmile
import pandas as pd
import pathlib

# Anchor relative paths to the project root (folder containing 'common').
_ROOT = next(
    _p for _p in pathlib.Path(__file__).resolve().parents
    if (_p / "common" / "__init__.py").exists()
)
os.chdir(_ROOT)

# ============================== CONFIGURATION ==============================

WAV_ROOT = "data/ESD"
OUTPUT_CSV = "data/ESD/ESD_eGeMAPS.csv"
# ==========================================================================


# Initialize openSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

rows = []

# Iterate over WAV files
counter = 1
for root, _, files in os.walk(WAV_ROOT):
    for fname in files:

        # Just to keep track of the status of the computation
        print(f"{counter}")
        counter += 1

        if not fname.lower().endswith(".wav"):
            continue

        path = os.path.join(root, fname)

        try:
            # Extract eGeMAPS
            feats = smile.process_file(path)

            # feats is a DataFrame with a single row
            row = feats.iloc[0].to_dict()
            row["filename"] = fname

            rows.append(row)

        except Exception as e:
            print(f"Failed on {fname}: {e}")

# Save to CSV
df = pd.DataFrame(rows)

# Put filename first
cols = ["filename"] + [c for c in df.columns if c != "filename"]
df = df[cols]
df = df.round(3)

df.to_csv(OUTPUT_CSV, index=False, sep=";")
