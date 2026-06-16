"""
common - shared library for the master-thesis Speech Emotion Recognition pipeline.

Modules:
    paths            project-root discovery + per-script output/ helper
    io               embedding-cache + ground-truth loaders and their join
    embeddings       wav2vec2 feature extraction (audio -> pooled embeddings)
    retrained_model  the retrained LNMLPHead classifier + loader
    probes           logistic-regression probe training/caching + loading
    geometry         subspace geometry (principal angles, overlap measures)
    labels           emotion-label normalization ("surprise" -> "surprised")
    viz              t-SNE + k-NN evaluation and plotting

Scripts run from the repo root and write into an output/ folder beside each
script (common.paths.output_dir). Embedding (.npz) and probe caches are
intermediate artifacts kept in their canonical locations, not per-script outputs.
"""

__all__ = [
    "paths",
    "io",
    "embeddings",
    "retrained_model",
    "probes",
    "geometry",
    "labels",
    "viz",
]

__version__ = "2.0.0"
