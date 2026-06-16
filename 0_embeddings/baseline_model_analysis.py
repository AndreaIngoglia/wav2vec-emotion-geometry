"""
STEP 0 (aux) - Pretrained checkpoint introspection

Inspects the baseline wav2vec2 emotion-recognition checkpoint: config,
architecture, classification-head weights and state-dict summary.

Output: 0_embeddings/output/ -> model_report.md, model_architecture.txt,
head_weights.txt, state_dict_summary.txt, model_diagram.mmd
"""

import os
import json
from datetime import datetime
import torch
from transformers import AutoConfig

# Optional: safetensors
try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None


# Make the shared package importable when launched from the repository root.
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

# ============================== CONFIGURATION ==============================
# Local folder of the pretrained checkpoint to inspect (edit this path).
MODEL_REPO_DIR = "wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# Reports are written to 0_embeddings/output/ (next to this script).
OUTPUT_DIR = str(output_dir(__file__))
# ==========================================================================


# ------------------------- Helpers -------------------------

def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _find_first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _load_state_dict(weights_path: str):
    ext = os.path.splitext(weights_path)[1].lower()
    if ext == ".bin":
        sd = torch.load(weights_path, map_location="cpu")
        # Sometimes wrapped
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        return sd
    if ext == ".safetensors":
        if safetensors_load_file is None:
            raise RuntimeError(
                "model.safetensors provided but safetensors is not installed/available."
            )
        return safetensors_load_file(weights_path)
    raise ValueError(f"Unsupported weights file: {weights_path}")


def _tensor_shape(x):
    return tuple(x.shape) if torch.is_tensor(x) else None


def _format_kv(k, v):
    return f"- **{k}**: `{v}`"


def _guess_head_keys(sd: dict):
    # Common names in HF audio classification/sequence classification heads
    markers = ("classifier", "projector", "score", "lm_head", "output_layer")
    return sorted([k for k in sd.keys() if any(m in k.lower() for m in markers)])


def _make_mermaid(cfg: AutoConfig, preproc):
    n_layers = getattr(cfg, "num_hidden_layers", None)
    hs = getattr(cfg, "hidden_size", None)
    heads = getattr(cfg, "num_attention_heads", None)
    conv = getattr(cfg, "num_feat_extract_layers", None)
    pooling = getattr(cfg, "pooling_mode", None)
    num_labels = getattr(cfg, "num_labels", None)
    sr = None
    if preproc:
        sr = preproc.get("sampling_rate", None)

    lines = []
    lines.append("flowchart TD")
    lines.append(f'  A["Raw audio waveform{f" ({sr} Hz)" if sr else ""}"] --> B["Feature Encoder (Conv stack)"]')
    if conv is not None:
        lines.append(f'  B --> Bm["Conv layers: {conv}"]')
    lines.append('  B --> C["Transformer Encoder (Wav2Vec2)"]')
    if n_layers is not None or hs is not None or heads is not None:
        meta = []
        if n_layers is not None: meta.append(f"Layers={n_layers}")
        if hs is not None: meta.append(f"Hidden={hs}")
        if heads is not None: meta.append(f"Heads={heads}")
        lines.append(f'  C --> Cm["{" | ".join(meta)}"]')
        lines.append("  Cm --> D[Frame-level representations]")
    else:
        lines.append("  C --> D[Frame-level representations]")

    if pooling:
        lines.append(f'  D --> E["Temporal pooling: {pooling}"]')
    else:
        lines.append('  D --> E["Temporal pooling"]')

    if num_labels is not None:
        lines.append(f'  E --> F["Classification head → logits ({num_labels} classes)"]')
    else:
        lines.append('  E --> F["Classification head → logits"]')

    return "\n".join(lines) + "\n"


def _try_load_training_args(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        obj = torch.load(path, map_location="cpu")
        # training_args.bin is usually a TrainingArguments object; stringify it
        return str(obj)
    except Exception as e:
        return f"[Could not load training_args.bin: {repr(e)}]"


# ------------------------- Public API -------------------------

def analyze_hf_repo_checkpoint(path_destinazione: str, path_repo_modello: str):
    """
    Analyze a locally cloned Hugging Face repo folder containing:
    - config.json
    - (optional) preprocessor_config.json
    - (optional) training_args.bin
    - pytorch_model.bin and/or model.safetensors

    Writes multiple output files into path_destinazione.
    """
    _safe_mkdir(path_destinazione)

    repo = os.path.abspath(path_repo_modello)

    config_path = os.path.join(repo, "config.json")
    preproc_path = os.path.join(repo, "preprocessor_config.json")
    trainargs_path = os.path.join(repo, "training_args.bin")

    weights_path = _find_first_existing(
        os.path.join(repo, "model.safetensors"),
        os.path.join(repo, "pytorch_model.bin"),
    )

    if weights_path is None:
        raise FileNotFoundError("No weights found (model.safetensors or pytorch_model.bin) in repo folder.")

    analyze_checkpoint_files(
        path_destinazione=path_destinazione,
        config_json_path=config_path if os.path.exists(config_path) else None,
        preprocessor_json_path=preproc_path if os.path.exists(preproc_path) else None,
        training_args_bin_path=trainargs_path if os.path.exists(trainargs_path) else None,
        weights_path=weights_path,
        repo_root=repo,
    )


def analyze_checkpoint_files(
    path_destinazione: str,
    config_json_path: str,
    preprocessor_json_path: str,
    training_args_bin_path: str,
    weights_path: str,
    repo_root: str = None,
):
    """
    Lower-level function: pass explicit file paths.
    All params are strings (or None).
    Writes reports/diagram to path_destinazione.
    """
    _safe_mkdir(path_destinazione)

    # ---- Load JSON configs if present
    raw_cfg = _read_json(config_json_path) if (config_json_path and os.path.exists(config_json_path)) else None
    preproc = _read_json(preprocessor_json_path) if (preprocessor_json_path and os.path.exists(preprocessor_json_path)) else None
    training_args_text = _try_load_training_args(training_args_bin_path) if training_args_bin_path else None

    cfg = None
    if config_json_path and os.path.exists(config_json_path):
        # AutoConfig can load from folder; prefer repo_root if available
        load_from = repo_root if (repo_root and os.path.exists(repo_root)) else os.path.dirname(os.path.abspath(config_json_path))
        cfg = AutoConfig.from_pretrained(load_from, local_files_only=True)

    # ---- Load weights
    sd = _load_state_dict(weights_path)

    # ---- Stats from weights
    total_params = 0
    dtypes = set()
    biggest = []
    for k, v in sd.items():
        if torch.is_tensor(v):
            n = v.numel()
            total_params += n
            dtypes.add(str(v.dtype))
            biggest.append((n, k, tuple(v.shape)))
    biggest.sort(reverse=True)

    head_keys = _guess_head_keys(sd)

    # ---- Try instantiating model and printing architecture (best-effort)
    model_print = None
    model_load_info = None
    if cfg is not None:
        try:
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Model
            archs = getattr(cfg, "architectures", []) or []
            arch0 = archs[0] if archs else ""

            if "Wav2Vec2ForSequenceClassification" in arch0:
                model_obj = Wav2Vec2ForSequenceClassification(cfg)
            else:
                model_obj = Wav2Vec2Model(cfg)

            missing, unexpected = model_obj.load_state_dict(sd, strict=False)
            model_load_info = f"Instantiated: {model_obj.__class__.__name__}\nMissing keys: {len(missing)}\nUnexpected keys: {len(unexpected)}\n"
            model_print = str(model_obj)
        except Exception as e:
            model_load_info = f"[Could not instantiate Transformers model: {repr(e)}]"
            model_print = None

    # ---- Write outputs
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_report = os.path.join(path_destinazione, "model_report.md")
    out_arch = os.path.join(path_destinazione, "model_architecture.txt")
    out_head = os.path.join(path_destinazione, "head_weights.txt")
    out_sd = os.path.join(path_destinazione, "state_dict_summary.txt")
    out_mmd = os.path.join(path_destinazione, "model_diagram.mmd")

    # 1) Markdown report
    lines = []
    lines.append(f"# Model report\n")
    lines.append(_format_kv("Generated", now))
    lines.append(_format_kv("Weights file", os.path.abspath(weights_path)))
    if repo_root:
        lines.append(_format_kv("Repo root", os.path.abspath(repo_root)))
    lines.append("")
    lines.append("## Weight statistics")
    lines.append(_format_kv("Total parameters (by tensor elements)", f"{total_params:,}"))
    lines.append(_format_kv("Dtypes", ", ".join(sorted(dtypes)) if dtypes else "unknown"))
    lines.append("")
    lines.append("### Largest tensors (top 20)")
    for n, k, shp in biggest[:20]:
        lines.append(f"- `{k}` — shape `{shp}` — params `{n:,}`")
    lines.append("")

    if cfg is not None:
        lines.append("## Configuration (from config.json)")
        lines.append(_format_kv("architectures", getattr(cfg, "architectures", None)))
        lines.append(_format_kv("model_type", getattr(cfg, "model_type", None)))
        lines.append(_format_kv("num_hidden_layers", getattr(cfg, "num_hidden_layers", None)))
        lines.append(_format_kv("hidden_size", getattr(cfg, "hidden_size", None)))
        lines.append(_format_kv("intermediate_size", getattr(cfg, "intermediate_size", None)))
        lines.append(_format_kv("num_attention_heads", getattr(cfg, "num_attention_heads", None)))
        lines.append(_format_kv("num_feat_extract_layers", getattr(cfg, "num_feat_extract_layers", None)))
        lines.append(_format_kv("pooling_mode", getattr(cfg, "pooling_mode", None)))
        lines.append(_format_kv("num_labels", getattr(cfg, "num_labels", None)))
        lines.append(_format_kv("final_dropout", getattr(cfg, "final_dropout", None)))
        lines.append("")

        id2label = getattr(cfg, "id2label", None)
        if id2label:
            lines.append("### Labels (id2label)")
            # cfg.id2label keys are often strings
            for k in sorted(id2label.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                lines.append(f"- `{k}`: `{id2label[k]}`")
            lines.append("")

    if preproc:
        lines.append("## Preprocessor (from preprocessor_config.json)")
        for k in ["sampling_rate", "do_normalize", "return_attention_mask", "feature_size", "padding_value"]:
            if k in preproc:
                lines.append(_format_kv(k, preproc[k]))
        lines.append("")

    if training_args_text:
        lines.append("## Training arguments (from training_args.bin)")
        lines.append("```")
        lines.append(training_args_text)
        lines.append("```")
        lines.append("")

    if head_keys:
        lines.append("## Classification head-related tensors (from weights)")
        lines.append("These keys strongly indicate the classifier/projector layers present in the checkpoint.")
        for k in head_keys[:200]:
            lines.append(f"- `{k}` — shape `{_tensor_shape(sd[k])}`")
        if len(head_keys) > 200:
            lines.append(f"- ... and {len(head_keys)-200} more")
        lines.append("")

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 2) Architecture print
    with open(out_arch, "w", encoding="utf-8") as f:
        if model_load_info:
            f.write(model_load_info + "\n\n")
        if model_print:
            f.write(model_print + "\n")
        else:
            f.write("No model object printed (config missing or instantiation failed).\n")

    # 3) Head keys file
    with open(out_head, "w", encoding="utf-8") as f:
        if not head_keys:
            f.write("No head-like keys found.\n")
        else:
            for k in head_keys:
                f.write(f"{k}\t{_tensor_shape(sd[k])}\n")

    # 4) State dict summary
    with open(out_sd, "w", encoding="utf-8") as f:
        f.write(f"Weights: {os.path.abspath(weights_path)}\n")
        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Dtypes: {', '.join(sorted(dtypes)) if dtypes else 'unknown'}\n\n")
        f.write("Top 100 tensors by size:\n")
        for n, k, shp in biggest[:100]:
            f.write(f"{n:,}\t{k}\t{shp}\n")

    # 5) Mermaid diagram
    if cfg is not None:
        diagram = _make_mermaid(cfg, preproc)
    else:
        diagram = "flowchart TD\n  A[config.json not found] --> B[Cannot build diagram reliably]\n"
    with open(out_mmd, "w", encoding="utf-8") as f:
        f.write(diagram)

    print("Wrote outputs to:", os.path.abspath(path_destinazione))
    print(" -", os.path.basename(out_report))
    print(" -", os.path.basename(out_arch))
    print(" -", os.path.basename(out_head))
    print(" -", os.path.basename(out_sd))
    print(" -", os.path.basename(out_mmd))


analyze_checkpoint_files(
    path_destinazione=OUTPUT_DIR,
    config_json_path=f"{MODEL_REPO_DIR}/config.json",
    preprocessor_json_path=f"{MODEL_REPO_DIR}/preprocessor_config.json",
    training_args_bin_path=f"{MODEL_REPO_DIR}/training_args.bin",
    weights_path=f"{MODEL_REPO_DIR}/pytorch_model.bin",
    repo_root=MODEL_REPO_DIR,
)
