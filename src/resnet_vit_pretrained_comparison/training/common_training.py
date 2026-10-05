from __future__ import annotations

import random
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)


# ============================================================
# 1) Reproducibility
# ============================================================
def set_seed(seed: int = 0):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2) Batch utilities
# ============================================================
def unpack_batch(batch, device, non_blocking: bool = True):
    """
    Unpack batch from DataLoader, handling multiple formats.

    Supported tuple formats:
        - 6-tuple:
            (images, meta_vecs, labels, masks,
             has_majority_flag, mean_teacher_confidence)

        - 4-tuple:
            (images, meta_vecs, labels, masks)

        - 3-tuple:
            (images, meta_vecs, labels)

        - 2-tuple:
            (images, labels)

    Supported dict format:
        keys may include:
            images, meta_vecs, labels, masks,
            has_majority_flag, mean_teacher_confidence

    Returns:
        images, meta_vecs, labels, masks, extras

    extras is always a dict and may contain:
        - has_majority_flag
        - mean_teacher_confidence
    """
    extras = {}

    # --------------------------------------------------------
    # dict batch support
    # --------------------------------------------------------
    if isinstance(batch, dict):
        images = batch.get("images")
        meta_vecs = batch.get("meta_vecs")
        labels = batch.get("labels")
        masks = batch.get("masks")

        has_majority_flag = batch.get("has_majority_flag")
        mean_teacher_confidence = batch.get("mean_teacher_confidence")

        if images is None or labels is None:
            raise ValueError(
                f"Dict batch must contain at least 'images' and 'labels'. "
                f"Got keys={list(batch.keys())}"
            )

        if has_majority_flag is not None:
            extras["has_majority_flag"] = has_majority_flag.to(
                device, non_blocking=non_blocking
            )
        if mean_teacher_confidence is not None:
            extras["mean_teacher_confidence"] = mean_teacher_confidence.to(
                device, non_blocking=non_blocking
            )

        return (
            images.to(device, non_blocking=non_blocking),
            meta_vecs.to(device, non_blocking=non_blocking) if meta_vecs is not None else None,
            labels.to(device, non_blocking=non_blocking),
            masks.to(device, non_blocking=non_blocking) if masks is not None else None,
            extras,
        )

    # --------------------------------------------------------
    # tuple/list batch support
    # --------------------------------------------------------
    if isinstance(batch, (list, tuple)) and len(batch) == 6:
        images, meta_vecs, labels, masks, has_majority_flag, mean_teacher_confidence = batch

        extras["has_majority_flag"] = (
            has_majority_flag.to(device, non_blocking=non_blocking)
            if has_majority_flag is not None
            else None
        )
        extras["mean_teacher_confidence"] = (
            mean_teacher_confidence.to(device, non_blocking=non_blocking)
            if mean_teacher_confidence is not None
            else None
        )

        return (
            images.to(device, non_blocking=non_blocking),
            meta_vecs.to(device, non_blocking=non_blocking) if meta_vecs is not None else None,
            labels.to(device, non_blocking=non_blocking),
            masks.to(device, non_blocking=non_blocking) if masks is not None else None,
            extras,
        )

    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        images, meta_vecs, labels, masks = batch
        return (
            images.to(device, non_blocking=non_blocking),
            meta_vecs.to(device, non_blocking=non_blocking) if meta_vecs is not None else None,
            labels.to(device, non_blocking=non_blocking),
            masks.to(device, non_blocking=non_blocking) if masks is not None else None,
            extras,
        )

    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        images, meta_vecs, labels = batch
        return (
            images.to(device, non_blocking=non_blocking),
            meta_vecs.to(device, non_blocking=non_blocking) if meta_vecs is not None else None,
            labels.to(device, non_blocking=non_blocking),
            None,
            extras,
        )

    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        images, labels = batch
        return (
            images.to(device, non_blocking=non_blocking),
            None,
            labels.to(device, non_blocking=non_blocking),
            None,
            extras,
        )

    raise ValueError(
        f"Unexpected batch format: type={type(batch)}, "
        f"len={len(batch) if hasattr(batch, '__len__') else 'N/A'}"
    )


def forward_flexible(
    model,
    images,
    meta_vecs,
    *,
    masks=None,
    use_meta=True,
    use_mask=False,
    mask_mode="multiply",
):
    """
    Run a forward pass, optionally concatenating the mask as the 4th channel.

    UPDATED behavior:
        - Spatial mask modes (multiply, blur_bg, soft_multiply, replace_bg) are now
          handled entirely inside the Dataset __getitem__, BEFORE the transform.
          forward_flexible no longer needs to process those modes.

        - "concat" mode is still handled here, because the mask tensor must be
          concatenated at the tensor level AFTER the standard RGB transform has run.
          The Dataset returns the mask as a separate (1, H, W) tensor in slot [3],
          and this function appends it as the 4th channel: (B, 3, H, W) -> (B, 4, H, W).

    Args:
        model      : nn.Module
        images     : (B, 3, H, W)
        meta_vecs  : (B, meta_dim) or None
        masks      : (B, 1, H, W) - used only for "concat" mode
        use_meta   : whether to pass meta_vecs to model
        use_mask   : whether to apply concat
        mask_mode  : only "concat" is handled here

    Returns:
        logits: (B, num_classes)
    """
    if use_mask and masks is not None and str(mask_mode).lower() == "concat":
        images = torch.cat([images, masks], dim=1)  # (B, 4, H, W)

    if use_meta and meta_vecs is not None:
        return model(images, meta_vecs)

    try:
        return model(images)
    except TypeError:
        return model(images, None)


# ============================================================
# 3) Class statistics & sampling
# ============================================================
def compute_class_counts(
    df: pd.DataFrame,
    label2idx: Dict[str, int],
    label_col: str = "dx",
) -> np.ndarray:
    """Compute per-class sample counts from DataFrame."""
    y = df[label_col].map(label2idx).astype(int).values
    counts = np.bincount(y, minlength=len(label2idx)).astype(np.float64)
    return counts


def make_weighted_sampler(
    df: pd.DataFrame,
    label2idx: Dict[str, int],
    label_col: str = "dx",
    eps: float = 1e-12,
) -> WeightedRandomSampler:
    """Create WeightedRandomSampler for class-balanced sampling."""
    y = df[label_col].map(label2idx).astype(int).values
    counts = np.bincount(y, minlength=len(label2idx)).astype(np.float64)
    per_class_weight = 1.0 / np.maximum(counts, eps)
    sample_weights = per_class_weight[y]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ============================================================
# 4) Imbalanced metrics
# ============================================================
def compute_imbalanced_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label2idx: Dict[str, int],
    target_class: str = "mel",
) -> Dict[str, float]:
    """
    Compute metrics suitable for imbalanced classification.

    Returns dict with keys:
      - balanced_acc
      - macro_f1
      - {target_class}_recall
      - {target_class}_precision
    """
    label_list_sorted = sorted(label2idx.items(), key=lambda kv: kv[1])
    labels = [idx for _, idx in label_list_sorted]

    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    recall_per_class = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )
    precision_per_class = precision_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )

    target_idx = label2idx.get(target_class)
    if target_idx is not None:
        target_pos = labels.index(target_idx)
        target_recall = float(recall_per_class[target_pos])
        target_precision = float(precision_per_class[target_pos])
    else:
        target_recall = 0.0
        target_precision = 0.0

    return {
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        f"{target_class}_recall": target_recall,
        f"{target_class}_precision": target_precision,
    }


def compute_balanced_crop_correctness(
    y_true: np.ndarray,
    crop_correct_fraction: np.ndarray,
    label2idx: Dict[str, int],
) -> float:
    """
    Balanced crop-level correctness for multi-view crop experiments.

    For each lesion/sample, crop_correct_fraction is the fraction of crop views
    that predicted the true lesion label correctly. The final metric averages
    this fraction inside each class, then averages across classes so majority
    classes do not dominate the stopping criterion.
    """
    y_true = np.asarray(y_true).reshape(-1)
    crop_correct_fraction = np.asarray(crop_correct_fraction, dtype=np.float64).reshape(-1)

    if y_true.size == 0 or crop_correct_fraction.size == 0:
        return 0.0
    if y_true.size != crop_correct_fraction.size:
        raise ValueError(
            "y_true and crop_correct_fraction must have the same length. "
            f"Got {y_true.size} and {crop_correct_fraction.size}."
        )

    class_scores = []
    for _, class_idx in sorted(label2idx.items(), key=lambda kv: kv[1]):
        mask = y_true == int(class_idx)
        if np.any(mask):
            class_scores.append(float(np.mean(crop_correct_fraction[mask])))

    if not class_scores:
        return 0.0

    return float(np.mean(class_scores))


# ============================================================
# 5) Early stopping helpers
# ============================================================
def metric_direction(metric_key: str) -> str:
    metric_key = (metric_key or "").lower().strip()
    if metric_key in {"loss", "val_loss"}:
        return "min"
    return "max"


def get_val_metric_from_history(
    history: Dict[str, list],
    metric_key: str,
    target_class: str = "mel",
) -> float:
    k = (metric_key or "").lower().strip()

    key_map = {
        "loss": "val_loss",
        "val_loss": "val_loss",
        "acc": "val_acc",
        "balanced_acc": "val_balanced_acc",
        "balanced_crop_correctness": "val_balanced_crop_correctness",
        "balanced_view_acc": "val_balanced_crop_correctness",
        "crop_correctness": "val_balanced_crop_correctness",
        "macro_f1": "val_macro_f1",
    }

    if k in key_map:
        return float(history[key_map[k]][-1])

    if k.endswith("_recall") or k.endswith("_precision"):
        return float(history[f"val_{k}"][-1])

    raise ValueError(
        f"Unknown metric_key='{metric_key}'. "
        f"Options: loss | acc | balanced_acc | balanced_crop_correctness | macro_f1 | "
        f"{target_class}_recall | {target_class}_precision"
    )


def check_early_stopping(
    current_value: float,
    best_value: float,
    metric_key: str,
) -> bool:
    direction = metric_direction(metric_key)
    if direction == "min":
        return current_value < best_value
    return current_value > best_value
