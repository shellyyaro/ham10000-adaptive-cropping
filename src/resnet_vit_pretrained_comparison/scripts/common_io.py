# src/experiments/common_io.py
# or:
# resnet_vit_pretrained_comparison/scripts/common_io.py
# depending on the path your project imports from

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix


WINDOWS_LEGACY_MAX_PATH = 260


# ============================================================
# Basic path helpers
# ============================================================

def _shorten_versioned_stem_for_path(
    parent: Path,
    stem: str,
    version_suffix: str,
    file_suffix: str,
    max_path_len: int = WINDOWS_LEGACY_MAX_PATH,
) -> str:
    """
    Keep versioned filenames below the Windows legacy path limit.

    A base artifact can be shortened to fit exactly, but adding _V1/_V2 can
    push it over the limit on reruns. In that case, trim the stem a little
    more while preserving the version suffix.
    """
    reserve = len(version_suffix) + len(file_suffix)
    # +1 accounts for the path separator between parent and filename.
    max_stem_len = max_path_len - len(str(parent)) - 1 - reserve
    max_stem_len = max(3, max_stem_len)
    return stem[:max_stem_len].rstrip("._-") or "file"


def get_free_path(
    base_path: str | Path,
    max_path_len: int = WINDOWS_LEGACY_MAX_PATH,
) -> Path:
    """
    If the target path already exists, create a versioned path:
    file.csv -> file_V1.csv -> file_V2.csv -> ...

    Note:
    This still protects against overwriting files if you rerun into the same
    directory. However, normal val/test outputs now include the split name in
    the filename, so you should not get ambiguous predictions.csv vs
    predictions_V1.csv just because both val and test were saved.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        return base_path

    stem, suffix, parent = base_path.stem, base_path.suffix, base_path.parent
    i = 1
    while True:
        version_suffix = f"_V{i}"
        versioned_stem = _shorten_versioned_stem_for_path(
            parent=parent,
            stem=stem,
            version_suffix=version_suffix,
            file_suffix=suffix,
            max_path_len=max_path_len,
        )
        cand = parent / f"{versioned_stem}{version_suffix}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def sanitize_tag(text: Any) -> str:
    """
    Make a path/file-name friendly string.
    """
    if text is None:
        return "none"

    s = str(text).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "unnamed"


def is_canonical_artifact_filename(filename: str) -> bool:
    """
    Return True for short, loader-facing artifact names that should not be
    shortened just because the experiment directory is already long.
    """
    fname = filename.lower()
    return re.match(
        r"^(val|test|train)_(predictions|classification_report|confusion_matrix|history)\.(csv|png)$",
        fname,
    ) is not None


def clamp_filename_for_path(
    parent_dir: str | Path,
    filename: str,
    max_path_len: int = 240,
) -> str:
    """
    Shorten filename if full path would exceed max_path_len.
    Preserves suffix and appends a stable hash when shortening is needed.
    """
    parent_dir = Path(parent_dir)
    filename = sanitize_tag(filename)

    full_path = parent_dir / filename
    if len(str(full_path)) <= max_path_len:
        return filename

    if (
        is_canonical_artifact_filename(filename)
        # Windows' legacy 260-character limit includes the terminating NUL in
        # some APIs, so a Python path string of length 260 can still fail.
        and len(str(full_path)) < WINDOWS_LEGACY_MAX_PATH
    ):
        return filename

    suffix = Path(filename).suffix
    stem = Path(filename).stem

    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:10]

    reserve = len(suffix) + len(digest) + 2
    allowed_stem_len = max(8, max_path_len - len(str(parent_dir)) - reserve - 5)

    short_stem = stem[:allowed_stem_len].rstrip("._-")
    if not short_stem:
        short_stem = "file"

    return f"{short_stem}_{digest}{suffix}"


def safe_out_path(
    out_dir: str | Path,
    filename: str,
    max_path_len: int = 240,
) -> Path:
    """
    Safe output path:
    - creates parent dir
    - sanitizes filename
    - shortens it if needed
    - returns versioned path if file already exists
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = clamp_filename_for_path(out_dir, filename, max_path_len=max_path_len)
    return get_free_path(out_dir / safe_name, max_path_len=max_path_len)


# ============================================================
# Run naming helpers
# ============================================================

def build_run_name_compact(
    model_key: str,
    cfg: Dict[str, Any],
    no_meta: bool = False,
    run_tag: str | None = None,
    arch: str | None = None,
) -> str:
    """
    Build a readable compact run name.

    Priority:
    1) explicit run_tag
    2) fallback from model_key / arch / meta flag
    """
    if run_tag:
        return sanitize_tag(run_tag).lower()

    parts = [sanitize_tag(model_key).lower()]

    if arch:
        parts.append(sanitize_tag(arch).lower())
    elif cfg.get("arch") is not None:
        parts.append(sanitize_tag(cfg.get("arch")).lower())

    parts.append("nometa" if no_meta else "meta")

    if "pretrained" in cfg:
        parts.append("pretrained" if bool(cfg.get("pretrained")) else "scratch")

    return "_".join([p for p in parts if p])


def make_run_dir(out_root: str | Path, run_name: str | None = None) -> Path:
    """
    Use out_root itself as the run directory.

    Important:
    We intentionally do NOT create an extra nested folder from run_name,
    because in many experiment scripts out_root already points to the
    configuration-specific directory. Creating another subfolder can easily
    cause Windows path-length errors.

    run_name is kept only for backward-compatible function signature.
    """
    run_dir = Path(out_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ============================================================
# Config writing
# ============================================================

def _json_safe(obj: Any) -> Any:
    """
    Convert objects to something JSON/config friendly.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_json_safe(v) for v in obj)
    return obj


def _legacy_text_config_writer_not_used(
    out_dir: str | Path,
    cfg: Dict[str, Any],
    extras: Dict[str, Any] | None = None,
    filename: str = "legacy_config.log",
) -> Path:
    """
    Write a human-readable config file similar to existing experiment logs.

    Updated:
    - Includes crop-only / random crop / multi-view crop / eval TTA fields.
    - Retained only for reading old development history; submission runners do
      not import or call this legacy text writer.
        * crop input
        * random crop policy
        * multi-view crop training
        * avg logits loss
        * avg logits + consistency
        * TTA logits average evaluation
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = safe_out_path(out_dir, filename)

    extras = extras or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    experiment_meta = {
        "timestamp": now,
        "run_tag": cfg.get("run_tag", "N/A"),
        "early_stopping_metric": extras.get("early_stopping_metric", "N/A"),
        "target_class": extras.get("target_class", "N/A"),
    }

    model_cfg_keys = [
        "model_type",
        "arch",
        "pretrained",
        "use_meta",
        "meta_hidden_dim",
        "clf_hidden_dim",
        "dropout",
        "img_emb_dim",

        # EfficientNet dual image / mask-gating
        "masked_branch_hidden_dim",
        "masked_branch_dropout",
        "masked_feat_scale_init",
        "gate_strength",
        "gate_init_std",
        "gate_strength_mode",
        "sample_gate_hidden_dim",
        "use_masked_img_stats",
        "use_mask_metadata",
        "gate_use_orig_feat",
        "gate_use_feat_diff",
        "share_backbone_weights",
        "force_backbone_bn_eval",

        # stem / pooling
        "stem_kernel_size",
        "stem_stride",
        "stem_use_maxpool",
        "pool_type",
        "gem_p",
        "gem_trainable",

        # meta
        "use_meta_bn",

        # multiscale stem
        "use_multiscale_stem",
        "stem_branch_kernels",
        "stem_branch_dilations",

        # multiscale block
        "use_multiscale_block",
        "multiscale_block_position",
        "multiscale_kernels",
        "multiscale_dilations",

        # mid-level fusion
        "use_midlevel_fusion",
        "midlevel_source",
        "midlevel_pool_type",
        "midlevel_emb_dim",
    ]

    training_keys = [
        "lr",
        "backbone_lr",
        "head_lr",
        "weight_decay",
        "batch_size",
        "num_epochs",
        "patience",
        "freeze_backbone_epochs",
        "img_size",
        "drop_last_train",
    ]

    aug_keys = [
        "rotation_deg",
        "hflip",
        "vflip",
        "hflip_p",
        "vflip_p",

        "use_color_jitter",
        "color_jitter_brightness",
        "color_jitter_contrast",
        "color_jitter_saturation",
        "color_jitter_hue",

        # Per-view color jitter flags
        "per_view_color_jitter_brightness",
        "per_view_color_jitter_contrast",
        "per_view_color_jitter_saturation",
        "per_view_color_jitter_hue",

        "use_random_crop",
        "random_crop_scale",
        "use_gaussian_blur",
        "gaussian_blur_p",
        "gaussian_blur_sigma",
        "gaussian_blur_kernel",
    ]

    imbalance_keys = [
        "loss_name",
        "class_weighting",
        "label_smoothing",
        "logit_adjustment",
        "tau",
        "sampler",
        "focal_gamma",
        "focal_alpha_mode",
    ]

    mask_keys = [
        "use_fused_masks",
        "mask_mode",
        "mask_npz_train",
        "mask_npz_val",
        "mask_npz_test",
        "background_mode",
        "blur_kernel_size",
        "blur_sigma",
        "softmultiply_alpha",
        "mask_power",
        "missing_mask_mode",
        "sanity_check_all_ones_masks",

        # crop branch
        "use_crop",
        "crop_threshold",
        "crop_margin_ratio",
        "crop_min_area",
        "crop_fallback_mode",

        # crop resize / adaptive aspect handling
        "crop_resize_mode",
        "adaptive_crop_shape",

        # optional mask-guided attention
        "use_mask_attention",
        "mask_attention_alpha",
    ]

    # ========================================================
    # New block for the crop/multiview/eval experiments
    # ========================================================
    crop_eval_keys = [
        # General input routing
        "input_source",
        "use_crop_as_input",

        # Single random crop policy
        "random_crop_policy",
        "crop_policy_choices",

        # Multi-view crop training
        "multi_view_crop",
        "multi_view_crop_policy_choices",
        "multi_view_loss_mode",

        # Evaluation mode
        "per_view_color_jitter_brightness",
        "per_view_color_jitter_contrast",
        "per_view_color_jitter_saturation",
        "per_view_color_jitter_hue",
        "eval_crop_mode",
        "eval_crop_policy_choices",

        # resize strategy for crop views
        "crop_resize_mode",

        # adaptive crop geometry
        "adaptive_crop_shape",

        # optional mask-guided attention
        "use_mask_attention",
        "mask_attention_alpha",
    ]

    consistency_keys = [
        # old lesion-level consistency options
        "use_lesion_consistency",
        "consistency_feature_space",
        "consistency_pair_sampler",
        "return_features",
        "return_lesion_id",

        # current multi-view prediction consistency options
        "main_loss_weight",
        "consistency_weight",
        "consistency_loss_type",
        "view_ce_weight",
        "worst_view_ce_weight",
        "consensus_kl_weight",
        "consensus_kl_temperature",
        "wrong_crop_count_weight",
        "wrong_crop_count_temperature",

        # current multi-view feature consistency options
        "feature_consistency_weight",
        "feature_consistency_loss_type",
    ]

    other_keys = [
        "use_masked_image",
        "use_hog_lbp",
        "agcwd_preset",
        "debug_stop_after_model_build",
        "debug_mask_pipeline",
    ]

    setup_block = {
        "seed": extras.get("seed", cfg.get("seed", "N/A")),
        "split_seed": extras.get("split_seed", cfg.get("split_seed", "N/A")),
        "no_meta": extras.get("no_meta", cfg.get("no_meta", "N/A")),
        "device": extras.get("device", "N/A"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,

        # Useful experiment-level extras passed by runner
        "input_source": extras.get("input_source", cfg.get("input_source", "N/A")),
        "use_crop_as_input": extras.get(
            "use_crop_as_input",
            cfg.get("use_crop_as_input", "N/A"),
        ),
        "strict_mask_loading": extras.get("strict_mask_loading", "N/A"),
        "crop_resize_mode": extras.get(
            "crop_resize_mode",
            cfg.get("crop_resize_mode", "N/A"),
        ),
        "adaptive_crop_shape": extras.get(
            "adaptive_crop_shape",
            cfg.get("adaptive_crop_shape", "N/A"),
        ),
        "use_mask_attention": extras.get(
            "use_mask_attention",
            cfg.get("use_mask_attention", "N/A"),
        ),
        "mask_attention_alpha": extras.get(
            "mask_attention_alpha",
            cfg.get("mask_attention_alpha", "N/A"),
        ),
    }

    env_block = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    all_known_keys = (
        model_cfg_keys
        + training_keys
        + aug_keys
        + imbalance_keys
        + mask_keys
        + crop_eval_keys
        + consistency_keys
        + other_keys
    )

    def take(keys):
        return {
            k: _json_safe(cfg.get(k, "N/A"))
            for k in keys
            if k in cfg or k in all_known_keys
        }

    # Also write selected extras as a dedicated block.
    runner_extras_keys = [
        "seed",
        "split_seed",
        "no_meta",
        "target_class",
        "early_stopping_metric",
        "freeze_backbone_epochs",
        "input_source",
        "use_crop_as_input",
        "strict_mask_loading",
        "crop_threshold",
        "crop_margin_ratio",
        "crop_min_area",
        "crop_fallback_mode",
        "crop_resize_mode",
        "adaptive_crop_shape",
        "random_crop_policy_train_only",
        "crop_policy_choices_train_only",
        "multi_view_crop",
        "multi_view_crop_policy_choices",
        "multi_view_loss_mode",
        "main_loss_weight",
        "view_ce_weight",
        "worst_view_ce_weight",
        "consistency_weight",
        "consistency_loss_type",
        "consensus_kl_weight",
        "consensus_kl_temperature",
        "wrong_crop_count_weight",
        "wrong_crop_count_temperature",
        "feature_consistency_weight",
        "feature_consistency_loss_type",
        "per_view_color_jitter_brightness",
        "per_view_color_jitter_contrast",
        "per_view_color_jitter_saturation",
        "per_view_color_jitter_hue",
        "eval_crop_mode",
        "eval_crop_policy_choices",
    ]

    runner_extras_block = {
        k: _json_safe(extras.get(k, "N/A"))
        for k in runner_extras_keys
        if k in extras
    }

    sections = [
        ("EXPERIMENT METADATA", experiment_meta),
        ("MODEL CONFIGURATION", take(model_cfg_keys)),
        ("TRAINING PARAMETERS", take(training_keys)),
        ("AUGMENTATION", take(aug_keys)),
        ("IMBALANCE HANDLING", take(imbalance_keys)),
        ("MASK / SECOND BRANCH CONFIGURATION", take(mask_keys)),
        ("CROP / MULTI-VIEW / EVAL CONFIGURATION", take(crop_eval_keys)),
        ("CONSISTENCY CONFIGURATION", take(consistency_keys)),
        ("EXPERIMENT SETUP", setup_block),
        ("RUNNER EXTRAS", runner_extras_block),
        ("ENVIRONMENT", env_block),
        ("OTHER PARAMETERS", take(other_keys)),
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")

        for title, block in sections:
            f.write(f"[{title}]\n")
            if block:
                for k, v in block.items():
                    f.write(f"{k}: {_json_safe(v)}\n")
            else:
                f.write("N/A\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("FULL CONFIG JSON-LIKE DUMP\n")
        f.write("=" * 80 + "\n")
        for k in sorted(cfg.keys()):
            f.write(f"{k}: {_json_safe(cfg[k])}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"Config file generated: {now}\n")
        f.write("=" * 80 + "\n")

    return path


def write_config_json(
    out_dir: str | Path,
    cfg: Dict[str, Any],
    extras: Dict[str, Any] | None = None,
    filename: str = "config.json",
) -> Path:
    """
    Write a machine-readable JSON config file with the full configuration.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = safe_out_path(out_dir, filename)
    extras = extras or {}

    payload = {
        "written_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": _json_safe(cfg),
        "extras": _json_safe(extras),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path


def write_best_params_json(
    out_dir: str | Path,
    best_params: Dict[str, Any],
    extras: Dict[str, Any] | None = None,
    filename: str = "best_params_used.json",
) -> Path:
    """
    Save the exact best_params dict that was passed into training/model construction.
    This is often the most faithful source of truth for later reconstruction.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = safe_out_path(out_dir, filename)
    extras = extras or {}

    payload = {
        "written_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_params": _json_safe(best_params),
        "extras": _json_safe(extras),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path


# ============================================================
# Plot saving
# ============================================================

def save_training_plots(
    history: Dict[str, list],
    out_dir: str | Path,
    run_name: str = "train",
) -> Dict[str, str]:
    """
    Save simple line plots for loss / accuracy / balanced accuracy / macro f1 if available.

    Output filenames include run_name, for example:
        train_loss.png
        train_acc.png
        train_balanced_acc.png
        train_macro_f1.png
        train_history.csv

    This avoids ambiguous names in experiment folders.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    if not history:
        return paths

    df = pd.DataFrame(history)
    if df.empty:
        return paths

    run_prefix = sanitize_tag(run_name or "train").lower()

    x = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)

    plot_specs = [
        ("loss", ["train_loss", "val_loss"]),
        ("main_loss", ["train_main_loss", "val_main_loss"]),
        ("consistency_loss", ["train_consistency_loss", "val_consistency_loss"]),
        ("view_ce_loss", ["train_view_ce_loss", "val_view_ce_loss"]),
        ("worst_view_ce_loss", ["train_worst_view_ce_loss", "val_worst_view_ce_loss"]),
        ("consensus_kl_loss", ["train_consensus_kl_loss", "val_consensus_kl_loss"]),
        ("wrong_crop_count_loss", ["train_wrong_crop_count_loss", "val_wrong_crop_count_loss"]),
        ("feature_consistency_loss", ["train_feature_consistency_loss", "val_feature_consistency_loss"]),
        ("num_views", ["train_num_views", "val_num_views"]),
        ("balanced_crop_correctness", ["train_balanced_crop_correctness", "val_balanced_crop_correctness"]),
        ("acc", ["train_acc", "val_acc"]),
        ("balanced_acc", ["train_balanced_acc", "val_balanced_acc"]),
        ("macro_f1", ["train_macro_f1", "val_macro_f1"]),
    ]

    for metric_name, cols in plot_specs:
        present_cols = [c for c in cols if c in df.columns]
        if not present_cols:
            continue

        plt.figure(figsize=(8, 5))
        for c in present_cols:
            plt.plot(x, df[c], label=c)

        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{run_prefix} {metric_name}")
        plt.legend()
        plt.tight_layout()

        out_path = safe_out_path(out_dir, f"{run_prefix}_{metric_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        paths[f"{run_prefix}_{metric_name}_plot_path"] = str(out_path)

    history_csv = safe_out_path(out_dir, f"{run_prefix}_history.csv")
    df.to_csv(history_csv, index=False)
    paths[f"{run_prefix}_history_csv_path"] = str(history_csv)

    return paths


# ============================================================
# Eval artifacts
# ============================================================

def save_eval_artifacts(
    model_key: str,
    run_name: str,
    out_dir: str | Path,
    y_true_idx: list[int] | np.ndarray,
    y_pred_idx: list[int] | np.ndarray,
    label2idx: Dict[str, int],
    split_df: pd.DataFrame,
    target_class: str = "mel",
    title_tag: str = "",
) -> Dict[str, str]:
    """
    Save evaluation artifacts with clear split-aware names.

    Example when run_name='val':
        val_predictions.csv
        val_classification_report.csv
        val_confusion_matrix.csv
        val_confusion_matrix.png

    Example when run_name='test':
        test_predictions.csv
        test_classification_report.csv
        test_confusion_matrix.csv
        test_confusion_matrix.png

    This avoids ambiguous files such as:
        predictions.csv
        predictions_V1.csv
    where it is unclear which split each file belongs to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_prefix = sanitize_tag(run_name or "eval").lower()

    idx2label = {v: k for k, v in label2idx.items()}
    y_true_idx = np.asarray(y_true_idx)
    y_pred_idx = np.asarray(y_pred_idx)

    pred_df = split_df.reset_index(drop=True).copy()
    pred_df["split"] = split_prefix
    pred_df["y_true_idx"] = y_true_idx
    pred_df["y_pred_idx"] = y_pred_idx
    pred_df["true_dx"] = [idx2label[i] for i in y_true_idx]
    pred_df["pred_dx"] = [idx2label[i] for i in y_pred_idx]
    pred_df["is_correct"] = (pred_df["y_true_idx"] == pred_df["y_pred_idx"]).astype(int)

    preds_path = safe_out_path(out_dir, f"{split_prefix}_predictions.csv")
    pred_df.to_csv(preds_path, index=False)

    labels_sorted = sorted(label2idx.items(), key=lambda x: x[1])
    class_names = [name for name, _ in labels_sorted]
    class_indices = [idx for _, idx in labels_sorted]

    report = classification_report(
        y_true_idx,
        y_pred_idx,
        labels=class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T
    report_path = safe_out_path(out_dir, f"{split_prefix}_classification_report.csv")
    report_df.to_csv(report_path)

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=class_indices)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    cm_csv_path = safe_out_path(out_dir, f"{split_prefix}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")

    title_parts = [split_prefix, "confusion matrix"]
    if title_tag:
        title_parts.append(str(title_tag))

    plt.title(" ".join(title_parts))
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    cm_png_path = safe_out_path(out_dir, f"{split_prefix}_confusion_matrix.png")
    plt.savefig(cm_png_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        # New clear split-aware keys
        f"{split_prefix}_preds_path": str(preds_path),
        f"{split_prefix}_classification_report_path": str(report_path),
        f"{split_prefix}_cm_csv_path": str(cm_csv_path),
        f"{split_prefix}_cm_png_path": str(cm_png_path),

        # Backward-compatible generic keys
        # Existing runners expect these keys.
        "preds_path": str(preds_path),
        "classification_report_path": str(report_path),
        "cm_csv_path": str(cm_csv_path),
        "cm_png_path": str(cm_png_path),
    }
