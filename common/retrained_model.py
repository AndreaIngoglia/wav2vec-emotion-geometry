"""
common.retrained_model - the LNMLPHead emotion classifier retrained on ESD, plus
a loader that pairs it with the frozen wav2vec2 backbone for inference.

Heads are trained by 4_model_retraining/save_best_models.ipynb and saved as
4_model_retraining/saved_models/<condition>/{config.json, head_weights.pt}
(condition: english / chinese / all). Model = frozen backbone -> mean-pool of the
last hidden state -> LNMLPHead. Imports torch/transformers at load time.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
TARGET_SR = 16_000


class LNMLPHead(nn.Module):
    """LayerNorm -> Linear proj -> LayerNorm -> GELU/Dropout MLP -> logits.

    Must match the architecture trained in ``save_best_models.ipynb`` so the
    saved ``head_weights.pt`` loads cleanly.
    """

    def __init__(self, in_features, num_labels, proj_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.input_ln = nn.LayerNorm(in_features)
        self.proj = nn.Linear(in_features, proj_dim)
        self.ln = nn.LayerNorm(proj_dim)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        x = self.input_ln(x)
        x = self.proj(x)
        x = self.ln(x)
        return self.mlp(x)


def load_retrained_model(saved_dir, model_id: str = DEFAULT_MODEL, device: str = "cpu") -> dict:
    """Load the base HF model + the retrained head from ``saved_dir``.

    Returns a dict with the feature ``extractor``, the ``base_model`` (used as-is
    for the baseline), the frozen ``backbone``, the retrained ``head``, the saved
    ``config`` and the ``label2id`` / ``id2label`` mappings.
    """
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    saved_dir = Path(saved_dir)
    with open(saved_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    base_model = AutoModelForAudioClassification.from_pretrained(model_id).eval().to(device)

    backbone = base_model.wav2vec2
    for p in backbone.parameters():
        p.requires_grad = False

    head = LNMLPHead(
        in_features=config["embed_dim"],
        num_labels=config["num_labels"],
        proj_dim=config["proj_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    )
    state = torch.load(str(saved_dir / "head_weights.pt"), map_location="cpu")
    head.load_state_dict(state)
    head.eval().to(device)

    label2id = config["label2id"]
    return {
        "extractor": extractor,
        "base_model": base_model,
        "backbone": backbone,
        "head": head,
        "config": config,
        "label2id": label2id,
        "id2label": {v: k for k, v in label2id.items()},
    }


@torch.no_grad()
def embed_meanpool(backbone, inputs) -> "torch.Tensor":
    """Mean-pool the backbone's last hidden state over time -> (B, D)."""
    return backbone(**inputs).last_hidden_state.mean(dim=1)
