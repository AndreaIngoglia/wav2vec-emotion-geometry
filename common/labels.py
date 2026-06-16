"""
common.labels - emotion-label normalization (lower-case + canonical spelling,
e.g. "surprise" -> "surprised"), shared by the cross-dataset alignment and PCA
scripts.
"""

from __future__ import annotations

import numpy as np

# Canonical spelling used across the project.
LABEL_MAP = {
    "surprise": "surprised",
}


def normalize_label(label: str) -> str:
    """Lower-case a single label and map known aliases to a canonical form."""
    low = str(label).lower()
    return LABEL_MAP.get(low, low)


def normalize_labels_array(labels) -> np.ndarray:
    """Vectorized :func:`normalize_label` returning an object ndarray."""
    return np.array([normalize_label(l) for l in labels], dtype=object)
