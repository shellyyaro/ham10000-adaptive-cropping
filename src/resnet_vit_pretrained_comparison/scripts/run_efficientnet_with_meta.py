# resnet_vit_pretrained_comparison/scripts/run_efficientnet_with_meta.py
from __future__ import annotations

from pathlib import Path
import os
import time
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from resnet_vit_pretrained_comparison.training.common_imbalance import (
    build_loss_and_logits_transform,
)
from resnet_vit_pretrained_comparison.training.common_training import (
    set_seed,
    unpack_batch,
    forward_flexible,
    compute_class_counts,
    make_weighted_sampler,
    compute_imbalanced_metrics,
    compute_balanced_crop_correctness,
    metric_direction,
    get_val_metric_from_history,
)

from resnet_vit_pretrained_comparison.models.EfficientNet import (
    EfficientNetWithMeta,
    efficientnet_transforms,
)

from resnet_vit_pretrained_comparison.models.models_cnn_resnet import HAMMultiInputDataset
from resnet_vit_pretrained_comparison.data.datasets_masked_background import (
    HAMMaskedBackgroundDataset,
)
from resnet_vit_pretrained_comparison.data.mask_fusion_loader import FusedMaskLoader
from resnet_vit_pretrained_comparison.data.AGCWD import build_agcwd_from_cfg
from resnet_vit_pretrained_comparison.scripts.run_vit_with_meta import SimpleImageOnlyDataset

from resnet_vit_pretrained_comparison.scripts.common_io import (
    get_free_path,
    safe_out_path,
    build_run_name_compact,
    make_run_dir,
    write_config_json,
    save_training_plots,
    save_eval_artifacts,
)


# ============================================================
# Batch unpacking
# ============================================================

def unpack_batch_4(batch, device, non_blocking: bool = True):
    """
    Wrapper around common_training.unpack_batch that is robust to extra returned items.

    Expected first 4 items:
        images, meta_vecs, labels, masks

    Any extra returned fields, e.g. row_idx/image_id/path, are ignored.
    """
    items = unpack_batch(batch, device, non_blocking=non_blocking)

    if isinstance(items, dict):
        images = items["images"]
        meta_vecs = items.get("meta_vecs", None)
        labels = items["labels"]
        masks = items.get("masks", None)
        return images, meta_vecs, labels, masks

    if not isinstance(items, (tuple, list)):
        raise TypeError(
            f"unpack_batch returned unsupported type: {type(items)}. "
            f"Expected tuple/list/dict."
        )

    if len(items) < 4:
        raise ValueError(
            f"unpack_batch returned only {len(items)} values, expected at least 4."
        )

    images, meta_vecs, labels, masks = items[:4]
    return images, meta_vecs, labels, masks


def unpack_batch_crop_only(
    batch,
    device: torch.device,
    non_blocking: bool = True,
    *,
    use_meta: bool = True,
):
    """
    For HAMMaskedBackgroundDataset.

    HAMMaskedBackgroundDataset returns:
        orig_images, second_branch_images, meta_vecs, labels, extras

    In crop-only EfficientNet:
        images = second_branch_images

    Supported image shapes:
        single-view:
            images.shape = (B, C, H, W)

        multi-view:
            images.shape = (B, V, C, H, W)

    In both cases no mask is passed into EfficientNet.
    """
    if not isinstance(batch, (list, tuple)):
        raise ValueError(f"Expected tuple/list batch, got {type(batch)}")

    if len(batch) != 5:
        raise ValueError(
            "Expected batch from HAMMaskedBackgroundDataset: "
            "(orig_images, second_branch_images, meta_vecs, labels, extras). "
            f"Got len={len(batch)}"
        )

    orig_images, second_branch_images, meta_vecs, labels, extras = batch

    images = second_branch_images.to(device, non_blocking=non_blocking).float()

    if images.ndim not in (4, 5):
        raise ValueError(
            "Expected second_branch_images to be either "
            "(B,C,H,W) for single-view or (B,V,C,H,W) for multi-view. "
            f"Got shape={tuple(images.shape)}"
        )

    if use_meta:
        meta_vecs = meta_vecs.to(device, non_blocking=non_blocking).float()
    else:
        meta_vecs = None

    labels = labels.to(device, non_blocking=non_blocking).long()

    if extras is None:
        extras = {}

    if isinstance(extras, dict):
        normalized_extras = {}
        for k, v in extras.items():
            if torch.is_tensor(v):
                normalized_extras[k] = v.to(device, non_blocking=non_blocking)
            else:
                normalized_extras[k] = v
        extras = normalized_extras

    return images, meta_vecs, labels, extras


# ============================================================
# Strict mask loading helpers
# ============================================================

def _extract_image_ids_from_df(
    df: pd.DataFrame,
    img_col_name: str = "img_path",
) -> list[str]:
    if img_col_name not in df.columns:
        raise ValueError(f"Missing image path column '{img_col_name}' in DataFrame")

    return [Path(str(p)).stem for p in df[img_col_name].tolist()]


def _assert_loader_has_all_split_masks(
    *,
    loader: FusedMaskLoader,
    split_df: pd.DataFrame,
    split_name: str,
    img_col_name: str = "img_path",
) -> None:
    image_ids = _extract_image_ids_from_df(split_df, img_col_name=img_col_name)

    missing_ids = [img_id for img_id in image_ids if not loader.has_mask(img_id)]
    n_total = len(image_ids)
    n_missing = len(missing_ids)

    if n_missing > 0:
        preview = missing_ids[:20]
        raise ValueError(
            f"Mask coverage check failed for split='{split_name}'. "
            f"Missing masks for {n_missing}/{n_total} images. "
            f"First missing image_ids: {preview}"
        )

    print(
        f"✅ Mask coverage OK for split='{split_name}': "
        f"{n_total}/{n_total} images found in NPZ."
    )


def _load_required_fused_mask_loader(
    *,
    npz_path: str | Path | None,
    split_name: str,
    split_df: pd.DataFrame,
    debug: bool = False,
    img_col_name: str = "img_path",
) -> FusedMaskLoader:
    if npz_path is None or str(npz_path).strip() == "":
        raise FileNotFoundError(
            f"Missing required mask NPZ path for split='{split_name}'."
        )

    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Required mask NPZ file not found for split='{split_name}': {npz_path}"
        )

    loader = FusedMaskLoader.from_npz(
        npz_path,
        debug=debug,
        load_extra_fields=False,
    )
    print(f"✅ Loaded masks ({split_name}): {loader}")

    _assert_loader_has_all_split_masks(
        loader=loader,
        split_df=split_df,
        split_name=split_name,
        img_col_name=img_col_name,
    )

    return loader


def _maybe_load_optional_fused_mask_loader(
    *,
    npz_path: str | Path | None,
    split_name: str,
) -> FusedMaskLoader | None:
    if npz_path is None or str(npz_path).strip() == "":
        print(f"⚠️ Warning: mask_npz_{split_name} is empty.")
        return None

    npz_path = Path(npz_path)

    if not npz_path.exists():
        print(f"⚠️ Warning: mask_npz_{split_name} not found: {npz_path}")
        return None

    loader = FusedMaskLoader.from_npz(npz_path)
    print(f"✅ Loaded fused masks ({split_name}): {loader}")
    return loader


# ============================================================
# Crop policy helpers
# ============================================================

def _normalize_crop_policy_choices(
    crop_policy_choices: Optional[List[Dict[str, Any]]],
    *,
    fallback_threshold: float,
    fallback_margin_ratio: float,
) -> list[dict]:
    """
    Normalize deterministic crop policies for TTA / multi-view evaluation.

    Supports:
      fixed_margin:
        {"crop_mode": "fixed_margin", "crop_threshold": 0.20, "crop_margin_ratio": 0.05}

      adaptive:
        {"crop_mode": "adaptive", "crop_threshold": 0.20,
         "target_bbox_ratio": 0.40, "adaptive_scale_factor": 1.0}

      adaptive_mask_area:
        {"crop_mode": "adaptive_mask_area", "crop_threshold": 0.60,
         "target_mask_area_ratio": 0.10}

    The adaptive geometry itself must be implemented in HAMMaskedBackgroundDataset.
    """
    if not crop_policy_choices:
        return [
            {
                "crop_mode": "fixed_margin",
                "crop_threshold": float(fallback_threshold),
                "crop_margin_ratio": float(fallback_margin_ratio),
                "target_bbox_ratio": -1.0,
                "adaptive_scale_factor": 1.0,
                "adaptive_crop_shape": "square",
                "policy_index": 0,
            }
        ]

    normalized = []

    for i, choice in enumerate(crop_policy_choices):
        if not isinstance(choice, dict):
            raise ValueError(f"crop policy #{i} must be dict, got {type(choice)}")

        crop_mode = str(choice.get("crop_mode", "fixed_margin")).strip().lower()
        if crop_mode not in {"fixed_margin", "adaptive", "adaptive_mask_area"}:
            raise ValueError(
                f"crop policy #{i}.crop_mode must be 'fixed_margin', "
                f"'adaptive', or 'adaptive_mask_area', "
                f"got {crop_mode!r}"
            )

        if "crop_threshold" not in choice:
            raise ValueError(f"crop policy #{i} missing crop_threshold")

        thr = float(choice["crop_threshold"])
        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"crop policy #{i}.crop_threshold must be in [0,1], got {thr}")

        out = {
            "crop_mode": crop_mode,
            "crop_threshold": thr,
            "policy_index": int(choice.get("policy_index", i)),
        }

        if crop_mode == "fixed_margin":
            if "crop_margin_ratio" not in choice:
                raise ValueError(f"crop policy #{i} missing crop_margin_ratio")

            margin = float(choice["crop_margin_ratio"])
            if margin < 0.0:
                raise ValueError(
                    f"crop policy #{i}.crop_margin_ratio must be >= 0, got {margin}"
                )

            out["crop_margin_ratio"] = margin
            out["target_bbox_ratio"] = -1.0
            out["target_mask_area_ratio"] = -1.0
            out["adaptive_scale_factor"] = 1.0
            out["adaptive_crop_shape"] = str(
                choice.get("adaptive_crop_shape", "square")
            ).strip().lower()

        elif crop_mode == "adaptive":
            if "target_bbox_ratio" not in choice:
                raise ValueError(f"adaptive crop policy #{i} missing target_bbox_ratio")

            target = float(choice["target_bbox_ratio"])
            if not (0.01 <= target <= 1.0):
                raise ValueError(
                    f"adaptive crop policy #{i}.target_bbox_ratio must be in [0.01,1.0], "
                    f"got {target}"
                )

            adaptive_crop_shape = str(
                choice.get("adaptive_crop_shape", "square")
            ).strip().lower()
            if adaptive_crop_shape not in {"square", "preserve_aspect"}:
                raise ValueError(
                    f"adaptive crop policy #{i}.adaptive_crop_shape must be "
                    f"'square' or 'preserve_aspect', got {adaptive_crop_shape!r}"
                )

            # adaptive crop does not use margin ratio
            out["crop_margin_ratio"] = float(
                choice.get("crop_margin_ratio", 0.0)
            )
            out["target_bbox_ratio"] = target
            out["target_mask_area_ratio"] = -1.0
            out["adaptive_scale_factor"] = float(choice.get("adaptive_scale_factor", 1.0))
            out["adaptive_crop_shape"] = adaptive_crop_shape
        else:
            target_key = (
                "target_mask_area_ratio"
                if "target_mask_area_ratio" in choice
                else "target_bbox_ratio"
            )
            if target_key not in choice:
                raise ValueError(
                    f"adaptive_mask_area crop policy #{i} missing target_mask_area_ratio"
                )

            target = float(choice[target_key])
            if not (0.01 <= target <= 1.0):
                raise ValueError(
                    f"adaptive_mask_area crop policy #{i}.{target_key} must be "
                    f"in [0.01,1.0], got {target}"
                )

            out["crop_margin_ratio"] = 0.0
            out["target_bbox_ratio"] = -1.0
            out["target_mask_area_ratio"] = target
            out["adaptive_scale_factor"] = 1.0
            out["adaptive_crop_shape"] = "closest_rectangle"

        normalized.append(out)

    return normalized

def _build_crop_eval_dataset(
    *,
    split_df: pd.DataFrame,
    meta_features: np.ndarray,
    label2idx: dict,
    fused_mask_loader: FusedMaskLoader,
    agcwd_transform,
    img_size: int,
    crop_resize_mode: str,
    adaptive_crop_shape: str,
    crop_threshold: float,
    crop_margin_ratio: float,
    crop_min_area: int,
    single_crop_policy: Optional[Dict[str, Any]] = None,
    crop_fallback_mode: str,
    background_mode: str,
    blur_kernel_size: int,
    blur_sigma: float,
    softmultiply_alpha: float,
    mask_power: float,
    missing_mask_mode: str,
    debug_mask_pipeline: bool,
    debug_max_samples: int,
    return_mask_context_image: bool = False,
    mask_context_view_mode: str = "full",
) -> HAMMaskedBackgroundDataset:
    """
    Deterministic single-view crop dataset for val/test.
    """
    return HAMMaskedBackgroundDataset(
        df=split_df.reset_index(drop=True),
        meta_features=meta_features,
        label_col="dx",
        label2idx=label2idx,
        img_col_name="img_path",
        fused_mask_loader=fused_mask_loader,

        rotation_deg=0.0,
        hflip_p=0.0,
        vflip_p=0.0,

        agcwd_transform=agcwd_transform,
        sanity_check_all_ones_masks=False,

        background_mode=background_mode,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        softmultiply_alpha=softmultiply_alpha,
        mask_power=mask_power,
        missing_mask_mode=missing_mask_mode,

        debug=debug_mask_pipeline,
        debug_max_samples=debug_max_samples,

        img_size=int(img_size),
        crop_resize_mode=str(crop_resize_mode),
        adaptive_crop_shape=str(adaptive_crop_shape),
        use_color_jitter=False,

        use_crop=True,
        crop_threshold=float(crop_threshold),
        crop_margin_ratio=(
            float(single_crop_policy.get("crop_margin_ratio", crop_margin_ratio))
            if (
                    single_crop_policy is not None
                    and str(
                single_crop_policy.get("crop_mode", "fixed_margin")
            ).lower() == "fixed_margin"
            )
            else 0.0
        ),
        crop_min_area=int(crop_min_area),
        crop_fallback_mode=str(crop_fallback_mode),

        random_crop_policy=single_crop_policy is not None,
        crop_policy_choices=[single_crop_policy] if single_crop_policy is not None else None,

        multi_view_crop=False,
        multi_view_crop_policy_choices=None,
        return_mask_context_image=return_mask_context_image,
        mask_context_view_mode=mask_context_view_mode,
    )


def _build_multiview_crop_dataset(
    *,
    split_df: pd.DataFrame,
    meta_features: np.ndarray,
    label2idx: dict,
    fused_mask_loader: FusedMaskLoader,
    agcwd_transform,
    img_size: int,
    crop_resize_mode: str,
    adaptive_crop_shape: str,
    crop_threshold: float,
    crop_margin_ratio: float,
    crop_min_area: int,
    crop_fallback_mode: str,
    background_mode: str,
    blur_kernel_size: int,
    blur_sigma: float,
    softmultiply_alpha: float,
    mask_power: float,
    missing_mask_mode: str,
    debug_mask_pipeline: bool,
    debug_max_samples: int,
    multi_view_crop_policy_choices: List[Dict[str, Any]],
    use_color_jitter: bool,
    color_jitter_brightness: float,
    color_jitter_contrast: float,
    color_jitter_saturation: float,
    color_jitter_hue: float,
    per_view_color_jitter_brightness: bool = False,
    per_view_color_jitter_contrast: bool = False,
    per_view_color_jitter_saturation: bool = False,
    per_view_color_jitter_hue: bool = False,
    rotation_deg: float = 0.0,
    hflip_p: float = 0.0,
    vflip_p: float = 0.0,
    return_mask_context_image: bool = False,
    mask_context_view_mode: str = "full",
) -> HAMMaskedBackgroundDataset:
    """
    Multi-view crop dataset.

    Returns second_branch_images with shape:
        (V, C, H, W) per sample
    and after DataLoader:
        (B, V, C, H, W)
    """
    return HAMMaskedBackgroundDataset(
        df=split_df.reset_index(drop=True),
        meta_features=meta_features,
        label_col="dx",
        label2idx=label2idx,
        img_col_name="img_path",
        fused_mask_loader=fused_mask_loader,

        rotation_deg=float(rotation_deg),
        hflip_p=float(hflip_p),
        vflip_p=float(vflip_p),

        agcwd_transform=agcwd_transform,
        sanity_check_all_ones_masks=False,

        background_mode=background_mode,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        softmultiply_alpha=softmultiply_alpha,
        mask_power=mask_power,
        missing_mask_mode=missing_mask_mode,

        debug=debug_mask_pipeline,
        debug_max_samples=debug_max_samples,

        img_size=int(img_size),
        crop_resize_mode=str(crop_resize_mode),
        adaptive_crop_shape=str(adaptive_crop_shape),
        use_color_jitter=bool(use_color_jitter),
        color_jitter_brightness=float(color_jitter_brightness),
        color_jitter_contrast=float(color_jitter_contrast),
        color_jitter_saturation=float(color_jitter_saturation),
        color_jitter_hue=float(color_jitter_hue),
        per_view_color_jitter_brightness=bool(per_view_color_jitter_brightness),
        per_view_color_jitter_contrast=bool(per_view_color_jitter_contrast),
        per_view_color_jitter_saturation=bool(per_view_color_jitter_saturation),
        per_view_color_jitter_hue=bool(per_view_color_jitter_hue),

        use_crop=True,
        crop_threshold=float(crop_threshold),
        crop_margin_ratio=float(crop_margin_ratio),
        crop_min_area=int(crop_min_area),
        crop_fallback_mode=str(crop_fallback_mode),

        random_crop_policy=False,
        crop_policy_choices=None,

        multi_view_crop=True,
        multi_view_crop_policy_choices=multi_view_crop_policy_choices,
        return_mask_context_image=return_mask_context_image,
        mask_context_view_mode=mask_context_view_mode,
    )


# ============================================================
# Multi-view forward / loss helpers
# ============================================================

def _is_multiview_images(images: torch.Tensor) -> bool:
    return torch.is_tensor(images) and images.ndim == 5


def _flatten_multiview_batch(
    *,
    images: torch.Tensor,
    meta_vecs: torch.Tensor | None,
    use_meta: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
    """
    Convert:
        images:    (B, V, C, H, W)
        meta_vecs: (B, F)

    Into:
        images_flat: (B*V, C, H, W)
        meta_flat:   (B*V, F)

    Returns:
        images_flat, meta_flat, B, V
    """
    if images.ndim != 5:
        raise ValueError(
            f"Expected multi-view images shape (B,V,C,H,W), got {tuple(images.shape)}"
        )

    b, v, c, h, w = images.shape
    images_flat = images.reshape(b * v, c, h, w)

    meta_flat = None

    if use_meta:
        if meta_vecs is None:
            raise ValueError("use_meta=True but meta_vecs is None in multi-view batch.")

        if meta_vecs.ndim != 2:
            raise ValueError(
                f"Expected meta_vecs shape (B,F), got {tuple(meta_vecs.shape)}"
            )

        if meta_vecs.shape[0] != b:
            raise ValueError(
                f"Multi-view batch mismatch: images batch={b}, "
                f"meta_vecs batch={meta_vecs.shape[0]}"
            )

        f = meta_vecs.shape[1]
        meta_flat = (
            meta_vecs
            .unsqueeze(1)
            .expand(b, v, f)
            .reshape(b * v, f)
        )

    return images_flat, meta_flat, b, v


def _flatten_multiview_attention_masks(
    *,
    attention_masks: torch.Tensor | None,
    b: int,
    v: int,
) -> torch.Tensor | None:
    """Convert (B,V,1,H,W) attention masks to (B*V,1,H,W)."""
    if attention_masks is None:
        return None

    if not torch.is_tensor(attention_masks):
        raise TypeError(
            f"attention_masks must be a torch.Tensor or None, got {type(attention_masks)}"
        )

    if attention_masks.ndim != 5:
        raise ValueError(
            "Expected multi-view attention_masks shape (B,V,1,H,W), "
            f"got {tuple(attention_masks.shape)}"
        )

    if attention_masks.shape[0] != b or attention_masks.shape[1] != v:
        raise ValueError(
            "attention_masks are misaligned with images: "
            f"images batch/views=({b},{v}), masks shape={tuple(attention_masks.shape)}"
        )

    if attention_masks.shape[2] != 1:
        raise ValueError(
            f"Expected attention_masks channel dimension to be 1, got {attention_masks.shape[2]}"
        )

    return attention_masks.reshape(
        b * v,
        1,
        attention_masks.shape[-2],
        attention_masks.shape[-1],
    )


def _flatten_multiview_mask_context_images(
    *,
    mask_context_images: torch.Tensor | None,
    b: int,
    v: int,
) -> torch.Tensor | None:
    """
    Convert context-branch images to the flattened multi-view batch.

    Supported inputs:
        (B,3,H,W)     -> repeated for every crop view
        (B,V,3,H,W)   -> one context tensor per crop view
    """
    if mask_context_images is None:
        return None

    if not torch.is_tensor(mask_context_images):
        raise TypeError(
            "mask_context_images must be a torch.Tensor or None, "
            f"got {type(mask_context_images)}"
        )

    if mask_context_images.ndim == 4:
        if mask_context_images.shape[0] != b:
            raise ValueError(
                "mask_context_images are misaligned with images: "
                f"images batch={b}, context shape={tuple(mask_context_images.shape)}"
            )

        c, h, w = mask_context_images.shape[1:]
        return (
            mask_context_images
            .unsqueeze(1)
            .expand(b, v, c, h, w)
            .reshape(b * v, c, h, w)
        )

    if mask_context_images.ndim == 5:
        if mask_context_images.shape[0] != b or mask_context_images.shape[1] != v:
            raise ValueError(
                "mask_context_images are misaligned with images: "
                f"images batch/views=({b},{v}), context shape={tuple(mask_context_images.shape)}"
            )

        return mask_context_images.reshape(
            b * v,
            mask_context_images.shape[2],
            mask_context_images.shape[3],
            mask_context_images.shape[4],
        )

    raise ValueError(
        "Expected mask_context_images shape (B,3,H,W) or (B,V,3,H,W), "
        f"got {tuple(mask_context_images.shape)}"
    )


def _forward_model_optional_attention(
    *,
    model: nn.Module,
    images: torch.Tensor,
    meta_vecs: torch.Tensor | None,
    use_meta: bool,
    attention_masks: torch.Tensor | None = None,
    use_mask_attention: bool = False,
    mask_context_images: torch.Tensor | None = None,
    use_mask_context_branch: bool = False,
) -> torch.Tensor:
    """
    Preserve old behavior when use_mask_attention=False.
    When enabled, call EfficientNet directly with attention_mask.
    """
    if not bool(use_mask_attention) and not bool(use_mask_context_branch):
        return forward_flexible(
            model,
            images,
            meta_vecs,
            use_meta=use_meta,
            masks=None,
            use_mask=False,
            mask_mode="none",
        )

    if bool(use_mask_attention) and attention_masks is None:
        raise ValueError(
            "use_mask_attention=True but attention_masks is None. "
            "Make sure HAMMaskedBackgroundDataset returns extras['attention_mask']."
        )

    if bool(use_mask_context_branch) and mask_context_images is None:
        raise ValueError(
            "use_mask_context_branch=True but mask_context_images is None. "
            "Make sure HAMMaskedBackgroundDataset returns extras['mask_context_image']."
        )

    return model(
        images,
        meta_vecs if use_meta else None,
        attention_mask=attention_masks if bool(use_mask_attention) else None,
        mask_context_images=mask_context_images if bool(use_mask_context_branch) else None,
    )


def _forward_multiview_logits(
    *,
    model: nn.Module,
    images: torch.Tensor,
    meta_vecs: torch.Tensor | None,
    use_meta: bool,
    attention_masks: torch.Tensor | None = None,
    use_mask_attention: bool = False,
    mask_context_images: torch.Tensor | None = None,
    use_mask_context_branch: bool = False,
) -> torch.Tensor:
    """
    Forward multi-view crop batch.

    Args:
        images:
            (B, V, C, H, W)
        meta_vecs:
            (B, F)

    Returns:
        logits_views:
            (B, V, num_classes)
    """
    images_flat, meta_flat, b, v = _flatten_multiview_batch(
        images=images,
        meta_vecs=meta_vecs,
        use_meta=use_meta,
    )

    attention_flat = _flatten_multiview_attention_masks(
        attention_masks=attention_masks,
        b=b,
        v=v,
    )
    mask_context_flat = _flatten_multiview_mask_context_images(
        mask_context_images=mask_context_images,
        b=b,
        v=v,
    )

    logits_flat = _forward_model_optional_attention(
        model=model,
        images=images_flat,
        meta_vecs=meta_flat,
        use_meta=use_meta,
        attention_masks=attention_flat,
        use_mask_attention=use_mask_attention,
        mask_context_images=mask_context_flat,
        use_mask_context_branch=use_mask_context_branch,
    )

    if logits_flat.ndim != 2:
        raise ValueError(
            f"Expected logits_flat shape (B*V,num_classes), got {tuple(logits_flat.shape)}"
        )

    num_classes = logits_flat.shape[-1]
    logits_views = logits_flat.reshape(b, v, num_classes)

    return logits_views



def _forward_flexible_with_features(
    model: nn.Module,
    images: torch.Tensor,
    meta_vecs: torch.Tensor | None,
    *,
    use_meta: bool,
    attention_masks: torch.Tensor | None = None,
    use_mask_attention: bool = False,
    mask_context_images: torch.Tensor | None = None,
    use_mask_context_branch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward one regular 4D batch and return:
        logits:   (N, K)
        features: (N, F)

    This supports either:
      1. model.forward_features(images, meta_vecs), if available.
      2. EfficientNetWithMeta-like models exposing backbone/meta_net/classifier.
    """
    if hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
        if bool(use_mask_attention) or bool(use_mask_context_branch):
            if bool(use_mask_attention) and attention_masks is None:
                raise ValueError(
                    "use_mask_attention=True but attention_masks is None in forward_features."
                )
            if bool(use_mask_context_branch) and mask_context_images is None:
                raise ValueError(
                    "use_mask_context_branch=True but mask_context_images is None in forward_features."
                )
            logits, features = model.forward_features(
                images,
                meta_vecs,
                attention_mask=attention_masks if bool(use_mask_attention) else None,
                mask_context_images=(
                    mask_context_images if bool(use_mask_context_branch) else None
                ),
            )
        else:
            logits, features = model.forward_features(images, meta_vecs)
        return logits, features

    if not hasattr(model, "backbone") or not hasattr(model, "classifier"):
        raise AttributeError(
            "feature_consistency_weight > 0 requires either model.forward_features(...) "
            "or an EfficientNetWithMeta-like model with .backbone and .classifier."
        )

    if images.ndim != 4:
        raise ValueError(f"Expected images shape (N,C,H,W), got {tuple(images.shape)}")

    if bool(use_mask_attention):
        raise AttributeError(
            "use_mask_attention=True with feature_consistency requires "
            "model.forward_features(..., attention_mask=...)."
        )

    img_feat = model.backbone(images.float())
    model_uses_meta = bool(getattr(model, "use_meta", use_meta))

    if model_uses_meta:
        if meta_vecs is None:
            raise ValueError("Model uses metadata but meta_vecs is None.")
        if not hasattr(model, "meta_net") or model.meta_net is None:
            raise AttributeError("Model uses metadata but does not expose meta_net.")
        meta_feat = model.meta_net(meta_vecs.float())
        features = torch.cat([img_feat, meta_feat], dim=1)
    else:
        features = img_feat

    logits = model.classifier(features)
    return logits, features


def _forward_multiview_logits_and_features(
    *,
    model: nn.Module,
    images: torch.Tensor,
    meta_vecs: torch.Tensor | None,
    use_meta: bool,
    attention_masks: torch.Tensor | None = None,
    use_mask_attention: bool = False,
    mask_context_images: torch.Tensor | None = None,
    use_mask_context_branch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward multi-view crop batch and return both logits and feature vectors.

    Args:
        images: (B, V, C, H, W)
        meta_vecs: (B, F)

    Returns:
        logits_views:  (B, V, K)
        feature_views: (B, V, F)
    """
    images_flat, meta_flat, b, v = _flatten_multiview_batch(
        images=images,
        meta_vecs=meta_vecs,
        use_meta=use_meta,
    )

    attention_flat = _flatten_multiview_attention_masks(
        attention_masks=attention_masks,
        b=b,
        v=v,
    )
    mask_context_flat = _flatten_multiview_mask_context_images(
        mask_context_images=mask_context_images,
        b=b,
        v=v,
    )

    logits_flat, features_flat = _forward_flexible_with_features(
        model,
        images_flat,
        meta_flat,
        use_meta=use_meta,
        attention_masks=attention_flat,
        use_mask_attention=use_mask_attention,
        mask_context_images=mask_context_flat,
        use_mask_context_branch=use_mask_context_branch,
    )

    if logits_flat.ndim != 2:
        raise ValueError(f"Expected logits_flat shape (B*V,K), got {tuple(logits_flat.shape)}")
    if features_flat.ndim != 2:
        raise ValueError(f"Expected features_flat shape (B*V,F), got {tuple(features_flat.shape)}")

    num_classes = logits_flat.shape[-1]
    feat_dim = features_flat.shape[-1]

    logits_views = logits_flat.reshape(b, v, num_classes)
    feature_views = features_flat.reshape(b, v, feat_dim)

    return logits_views, feature_views


def _compute_feature_consistency_loss(
    feature_views: torch.Tensor,
    *,
    feature_consistency_loss_type: str = "cosine",
    logits_views: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Feature-level consistency across crop views of the same image.

    feature_views:
        (B, V, F)

    Options:
        cosine:
            Mean cosine distance between each view feature and the mean feature.
        mse:
            MSE between each view feature and the mean feature.
        correct_teacher_wrong_only_cosine:
            Version B feature consistency. For each image, use only crop views
            whose current top-1 prediction is correct as a detached feature
            teacher, then pull only the wrong crop-view features toward that
            teacher. Images with no correct crops or no wrong crops contribute
            zero to this auxiliary loss.
    """
    if feature_views.ndim != 3:
        raise ValueError(
            f"Expected feature_views shape (B,V,F), got {tuple(feature_views.shape)}"
        )

    b, v, f = feature_views.shape
    if v <= 1:
        return feature_views.new_tensor(0.0)

    loss_type = str(feature_consistency_loss_type or "cosine").strip().lower()

    if loss_type in {"mse", "mse_features"}:
        mean_feat = feature_views.mean(dim=1, keepdim=True)
        return ((feature_views - mean_feat) ** 2).mean()

    if loss_type in {"cosine", "cosine_distance"}:
        feat_norm = F.normalize(feature_views.float(), p=2, dim=-1)
        mean_feat = F.normalize(feat_norm.mean(dim=1, keepdim=True), p=2, dim=-1)
        cosine_sim = (feat_norm * mean_feat).sum(dim=-1)
        return (1.0 - cosine_sim).mean()

    if loss_type in {
        "correct_teacher_wrong_only_cosine",
        "correct_teacher_cosine",
        "correct_crop_teacher_cosine",
        "wrong_to_correct_teacher_cosine",
    }:
        if logits_views is None or labels is None:
            raise ValueError(
                f"feature_consistency_loss_type={feature_consistency_loss_type!r} "
                "requires logits_views and labels."
            )
        if logits_views.ndim != 3:
            raise ValueError(
                f"Expected logits_views shape (B,V,K), got {tuple(logits_views.shape)}"
            )
        if logits_views.shape[:2] != feature_views.shape[:2]:
            raise ValueError(
                "logits_views and feature_views must have matching (B,V) dimensions. "
                f"Got logits={tuple(logits_views.shape)}, features={tuple(feature_views.shape)}"
            )

        labels_expanded = labels.view(b, 1).expand(b, v)
        pred_views = logits_views.detach().argmax(dim=-1)
        correct_mask = pred_views.eq(labels_expanded)
        wrong_mask = ~correct_mask

        feat_norm = F.normalize(feature_views.float(), p=2, dim=-1)
        sample_losses = []

        for sample_idx in range(b):
            correct_feats = feat_norm[sample_idx][correct_mask[sample_idx]]
            wrong_feats = feat_norm[sample_idx][wrong_mask[sample_idx]]

            if correct_feats.numel() == 0 or wrong_feats.numel() == 0:
                continue

            teacher = F.normalize(
                correct_feats.mean(dim=0, keepdim=True),
                p=2,
                dim=-1,
            ).detach()
            cosine_sim = (wrong_feats * teacher).sum(dim=-1)
            sample_losses.append(1.0 - cosine_sim)

        if not sample_losses:
            return feature_views.new_tensor(0.0)

        return torch.cat(sample_losses, dim=0).mean().to(dtype=feature_views.dtype)

    raise ValueError(
        f"Unknown feature_consistency_loss_type={feature_consistency_loss_type!r}. "
        "Expected 'cosine', 'mse', or 'correct_teacher_wrong_only_cosine'."
    )


def _compute_consistency_loss(
    logits_views: torch.Tensor,
    *,
    consistency_loss_type: str = "mse_probs",
) -> torch.Tensor:
    """
    Consistency loss across views of the same image.

    logits_views:
        (B, V, K)

    Options:
        mse_probs:
            MSE between each view probability and mean probability.

        kl_probs:
            KL(mean_probs || view_probs), averaged over views.  The mean is not
            detached in this mode, so gradients flow through both the view
            distributions and their average, as specified in the report.
    """
    if logits_views.ndim != 3:
        raise ValueError(
            f"Expected logits_views shape (B,V,K), got {tuple(logits_views.shape)}"
        )

    per_sample_loss = _compute_consistency_loss_per_sample(
        logits_views,
        consistency_loss_type=consistency_loss_type,
    )
    if per_sample_loss.numel() == 0:
        return logits_views.new_tensor(0.0)
    return per_sample_loss.mean()


def _compute_consistency_loss_per_sample(
    logits_views: torch.Tensor,
    *,
    consistency_loss_type: str = "mse_probs",
) -> torch.Tensor:
    if logits_views.ndim != 3:
        raise ValueError(
            f"Expected logits_views shape (B,V,K), got {tuple(logits_views.shape)}"
        )

    b, v, _ = logits_views.shape

    if v <= 1:
        return logits_views.new_zeros((b,))

    consistency_loss_type = str(consistency_loss_type).strip().lower()

    probs_views = torch.softmax(logits_views, dim=-1)
    mean_probs = probs_views.mean(dim=1, keepdim=True).clamp_min(1e-8)

    if consistency_loss_type in {"mse", "mse_probs", "prob_mse"}:
        return ((probs_views - mean_probs) ** 2).mean(dim=(1, 2))

    if consistency_loss_type in {
        "kl",
        "kl_probs",
        "mean_kl",
        "kl_probs_detached_teacher",
        "kl_detached_teacher",
        "detached_teacher_kl",
    }:
        log_probs_views = torch.log_softmax(logits_views, dim=-1)
        if consistency_loss_type in {
            "kl_probs_detached_teacher",
            "kl_detached_teacher",
            "detached_teacher_kl",
        }:
            mean_probs = mean_probs.detach()
        mean_probs_expanded = mean_probs.expand_as(probs_views)

        kl_per_class = F.kl_div(
            log_probs_views,
            mean_probs_expanded,
            reduction="none",
        )
        return kl_per_class.sum(dim=-1).mean(dim=1)

    raise ValueError(
        f"Unknown consistency_loss_type={consistency_loss_type!r}. "
        "Expected 'mse_probs' or 'kl_probs'."
    )


def _compute_true_margin_consistency_gate(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.05,
    margin_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Soft gate for consistency based on the aggregated true-vs-best-wrong margin.

    The gate is near 1 when the averaged prediction leans toward the true class
    and near 0 when it leans toward a wrong class.
    """
    if scores.ndim != 2:
        raise ValueError(f"Expected scores shape (B,K), got {tuple(scores.shape)}")

    b, k = scores.shape
    temperature = max(float(temperature), 1e-6)

    probs = torch.softmax(scores.float(), dim=-1)
    labels_expanded = labels.view(b, 1)
    true_probs = probs.gather(dim=-1, index=labels_expanded).squeeze(-1)

    true_mask = torch.zeros_like(probs, dtype=torch.bool)
    true_mask.scatter_(dim=-1, index=labels_expanded, value=True)
    wrong_probs = probs.masked_fill(true_mask, float("-inf"))
    best_wrong_probs = wrong_probs.max(dim=-1).values

    margin = true_probs - best_wrong_probs - float(margin_threshold)
    return torch.sigmoid(margin / temperature).to(dtype=scores.dtype)


def _compute_consensus_kl_loss(
    logits_views: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    View-to-consensus distillation loss.

    For each image, compute the average probability distribution across crop
    views and detach it so it acts as a temporary teacher:

        P_teacher = mean_i softmax(logits_view_i / T).detach()

    Then train every crop view to match that teacher distribution:

        mean_i KL(P_teacher || P_view_i)

    This differs from the regular consistency loss because gradients do not
    flow through the averaged teacher distribution. The main CE loss is still
    responsible for pulling the averaged image-level prediction toward the true
    label; this auxiliary loss only asks each crop to agree with that consensus.
    """
    if logits_views.ndim != 3:
        raise ValueError(
            f"Expected logits_views shape (B,V,K), got {tuple(logits_views.shape)}"
        )

    b, v, k = logits_views.shape
    if v <= 1:
        return logits_views.new_tensor(0.0)

    temperature = max(float(temperature), 1e-6)
    logits_t = logits_views / temperature

    probs_views = torch.softmax(logits_t, dim=-1)
    teacher_probs = probs_views.mean(dim=1, keepdim=True).detach().clamp_min(1e-8)
    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)
    teacher_expanded = teacher_probs.expand_as(probs_views)

    log_probs_views = torch.log_softmax(logits_t, dim=-1)

    # Multiply by T^2, as in standard distillation, so the loss scale remains
    # comparable if a temperature other than 1.0 is used.
    return (temperature ** 2) * F.kl_div(
        log_probs_views.reshape(b * v, k),
        teacher_expanded.reshape(b * v, k),
        reduction="batchmean",
    )



def _aggregate_multiview_logits(
    logits_views: torch.Tensor,
    *,
    aggregation: str = "avg_logits",
) -> torch.Tensor:
    """
    Aggregate logits across views.

    Args:
        logits_views:
            Tensor of shape (B, V, K), where:
                B = batch size
                V = number of crop views
                K = number of classes

        aggregation:
            avg_logits:
                Mean raw logits across views.
                This is the original MVCROP3 behavior.

            avg_probs:
                Apply softmax to each view, average the probabilities,
                and return log(avg_probs).

                Returning log(avg_probs) allows the tensor to be used with
                CrossEntropyLoss/NLL-style training while preserving the
                average-probability aggregation logic.

    Returns:
        Tensor of shape (B, K).

        For avg_logits:
            raw averaged logits.

        For avg_probs:
            log of averaged probabilities.
    """
    if logits_views.ndim != 3:
        raise ValueError(
            f"Expected logits_views shape (B,V,K), got {tuple(logits_views.shape)}"
        )

    aggregation = str(aggregation or "avg_logits").strip().lower()

    if aggregation == "avg_logits":
        return logits_views.mean(dim=1)

    if aggregation == "avg_probs":
        num_views = int(logits_views.shape[1])
        if num_views <= 0:
            raise ValueError("avg_probs aggregation requires at least one view.")

        # Use float32 for numerical stability, especially with AMP.
        log_probs_views = F.log_softmax(logits_views.float(), dim=-1)

        # log(mean(p_i)) = logsumexp(log(p_i)) - log(num_views)
        log_avg_probs = torch.logsumexp(log_probs_views, dim=1) - math.log(num_views)

        return log_avg_probs

    raise ValueError(
        f"Unknown multi-view aggregation={aggregation!r}. "
        "Expected 'avg_logits' or 'avg_probs'."
    )




def _compute_view_ce_loss(
    logits_views: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_fn: nn.Module,
    train_logits_tf: Callable,
) -> torch.Tensor:
    """
    Supervised CE loss for each crop view separately.

    logits_views:
        Tensor of shape (B, V, K).

    labels:
        Tensor of shape (B,).

    Returns:
        Mean classification loss over all B*V individual crop views.

    Why this is needed:
        The regular multi-view loss supervises only the averaged prediction
        across the three crop views. This auxiliary term explicitly forces
        every crop view to be predictive of the true label.
    """
    if logits_views.ndim != 3:
        raise ValueError(
            f"Expected logits_views shape (B,V,K), got {tuple(logits_views.shape)}"
        )

    b, v, k = logits_views.shape

    logits_flat = logits_views.reshape(b * v, k)
    labels_flat = labels.unsqueeze(1).expand(b, v).reshape(b * v)

    logits_flat_for_loss = train_logits_tf(logits_flat)
    return loss_fn(logits_flat_for_loss, labels_flat)


def _compute_worst_view_ce_loss(
    logits_views: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_fn: nn.Module,
    train_logits_tf: Callable,
) -> torch.Tensor:
    """
    Supervised CE loss for the worst crop view of each image.

    This is stricter than mean per-view CE. For each sample, it computes CE for
    all crop views and keeps the largest CE, so one bad crop cannot be hidden by
    the other views.
    """
    if logits_views.ndim != 3:
        raise ValueError(
            f"Expected logits_views shape (B,V,K), got {tuple(logits_views.shape)}"
        )

    if not isinstance(loss_fn, nn.CrossEntropyLoss):
        raise ValueError(
            "worst_view_ce_weight currently supports loss_name='ce' only."
        )

    b, v, k = logits_views.shape

    logits_flat = logits_views.reshape(b * v, k)
    labels_flat = labels.unsqueeze(1).expand(b, v).reshape(b * v)
    logits_flat_for_loss = train_logits_tf(logits_flat)

    ce_flat = F.cross_entropy(
        logits_flat_for_loss,
        labels_flat,
        weight=loss_fn.weight,
        ignore_index=loss_fn.ignore_index,
        reduction="none",
        label_smoothing=float(getattr(loss_fn, "label_smoothing", 0.0)),
    )

    ce_views = ce_flat.reshape(b, v)
    return ce_views.max(dim=1).values.mean()


def _compute_wrong_crop_count_loss(
    logits_views: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Differentiable proxy for the fraction of crop views whose top-1 prediction
    is wrong.

    For each crop view, compare the true-class logit with the strongest wrong
    class logit:

        margin_wrong = best_wrong_logit - true_logit

    If margin_wrong > 0, a wrong class is beating the true class.
    If margin_wrong < 0, the true class is winning.

    We convert this into a smooth wrongness score:

        soft_wrong = sigmoid(margin_wrong / temperature)

    This is close to 0 for clearly correct crops and close to 1 for clearly
    wrong crops.

    reduction="mean" gives a differentiable approximation of "what fraction
    of crop views are wrong", without argmax.

    reduction="sum" gives a differentiable approximation of "how many crop
    views are wrong". This is stronger for our 6-view setting because each
    wrong crop adds its own penalty instead of being diluted by 1 / num_views.
    """
    if logits_views.ndim != 3:
        raise ValueError(
            f"Expected logits_views shape (B,V,K), got {tuple(logits_views.shape)}"
        )

    b, v, k = logits_views.shape
    if v <= 0:
        return logits_views.new_tensor(0.0)

    temperature = max(float(temperature), 1e-6)
    logits_float = logits_views.float()

    labels_expanded = labels.view(b, 1, 1).expand(b, v, 1)
    true_logits = logits_float.gather(dim=-1, index=labels_expanded).squeeze(-1)

    true_mask = torch.zeros_like(logits_float, dtype=torch.bool)
    true_mask.scatter_(dim=-1, index=labels_expanded, value=True)
    wrong_logits = logits_float.masked_fill(true_mask, float("-inf"))
    best_wrong_logits = wrong_logits.max(dim=-1).values

    margin_wrong = best_wrong_logits - true_logits
    soft_wrong = torch.sigmoid(margin_wrong / temperature)

    reduction = str(reduction or "mean").strip().lower()
    if reduction == "mean":
        loss = soft_wrong.mean()
    elif reduction == "sum":
        loss = soft_wrong.sum(dim=1).mean()
    else:
        raise ValueError(
            f"Unknown wrong_crop_count_reduction={reduction!r}. "
            "Expected 'mean' or 'sum'."
        )

    return loss.to(dtype=logits_views.dtype)


def _compute_true_label_alignment_loss(
    logits_views: torch.Tensor,
    labels: torch.Tensor,
    *,
    tolerance: float = 0.02,
    min_teacher_prob: float = 0.25,
) -> torch.Tensor:
    """
    Prediction-level crop invariance anchored to the true label.

    For each lesion, compute the probability assigned to the true class in
    every crop view. The best true-class probability across views becomes a
    detached teacher. We then penalize views whose true-class probability is
    lower than that teacher by more than a small tolerance:

        loss_i = max(best_p_true.detach() - p_true_i - tolerance, 0)

    A sample contributes only when the best crop has at least
    min_teacher_prob probability for the true label, so weak/random teachers do
    not force the other crops to follow noise.
    """
    if logits_views.ndim != 3:
        raise ValueError(
            f"Expected logits_views shape (B,V,K), got {tuple(logits_views.shape)}"
        )

    b, v, k = logits_views.shape
    if b <= 0 or v <= 0:
        return logits_views.new_tensor(0.0)

    probs = torch.softmax(logits_views.float(), dim=-1)
    labels_expanded = labels.view(b, 1, 1).expand(b, v, 1)
    true_probs = probs.gather(dim=-1, index=labels_expanded).squeeze(-1)

    teacher_true_probs = true_probs.max(dim=1, keepdim=True).values.detach()
    active_samples = teacher_true_probs.ge(float(min_teacher_prob)).to(true_probs.dtype)

    per_view_loss = torch.relu(
        teacher_true_probs - true_probs - float(tolerance)
    )
    active_count = active_samples.sum()
    if active_count.item() <= 0:
        return logits_views.new_tensor(0.0)

    loss = (per_view_loss * active_samples).sum() / (active_count * v)
    return loss.to(dtype=logits_views.dtype)


def _compute_margin_hinge_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin_target: float = 0.2,
) -> torch.Tensor:
    """
    Main image-level margin hinge loss.

    For each image, compare the true-class score against the strongest wrong
    class score:

        margin = true_score - best_wrong_score

    The loss is active only while this margin is below margin_target:

        loss = max(margin_target - margin, 0)

    This keeps pressure on the averaged image prediction until the true label
    beats the strongest wrong label by a chosen margin, but it stops rewarding
    extra confidence after that.
    """
    if scores.ndim != 2:
        raise ValueError(f"Expected scores shape (B,K), got {tuple(scores.shape)}")

    b, k = scores.shape
    if b <= 0:
        return scores.new_tensor(0.0)

    scores_float = scores.float()
    labels_expanded = labels.view(b, 1)
    true_scores = scores_float.gather(dim=1, index=labels_expanded).squeeze(1)

    true_mask = torch.zeros_like(scores_float, dtype=torch.bool)
    true_mask.scatter_(dim=1, index=labels_expanded, value=True)
    wrong_scores = scores_float.masked_fill(true_mask, float("-inf"))
    best_wrong_scores = wrong_scores.max(dim=1).values

    margin = true_scores - best_wrong_scores
    loss = torch.relu(float(margin_target) - margin).mean()
    return loss.to(dtype=scores.dtype)


def _compute_margin_softplus_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin_target: float = 0.2,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Smooth version of the main image-level margin loss.

    For each image:

        margin = true_score - best_wrong_score

    Instead of the hard hinge:

        max(margin_target - margin, 0)

    use a soft hinge:

        T * softplus((margin_target - margin) / T)

    When the margin is below the target, this behaves similarly to the hinge.
    Once the margin passes the target, the loss does not become exactly zero;
    it keeps a small, smoothly decaying gradient.
    """
    if scores.ndim != 2:
        raise ValueError(f"Expected scores shape (B,K), got {tuple(scores.shape)}")

    b, k = scores.shape
    if b <= 0:
        return scores.new_tensor(0.0)

    temperature = max(float(temperature), 1e-6)
    scores_float = scores.float()
    labels_expanded = labels.view(b, 1)
    true_scores = scores_float.gather(dim=1, index=labels_expanded).squeeze(1)

    true_mask = torch.zeros_like(scores_float, dtype=torch.bool)
    true_mask.scatter_(dim=1, index=labels_expanded, value=True)
    wrong_scores = scores_float.masked_fill(true_mask, float("-inf"))
    best_wrong_scores = wrong_scores.max(dim=1).values

    margin = true_scores - best_wrong_scores
    loss = temperature * F.softplus((float(margin_target) - margin) / temperature)
    return loss.mean().to(dtype=scores.dtype)


def _compute_logits_loss_and_pred(
    *,
    model: nn.Module,
    images: torch.Tensor,
    meta_vecs: torch.Tensor | None,
    labels: torch.Tensor,
    use_meta: bool,
    use_mask: bool,
    masks: torch.Tensor | None,
    mask_mode: str,
    train_logits_tf: Callable,
    eval_logits_tf: Callable,
    loss_fn: nn.Module,
    main_loss_type: str = "ce",
    main_loss_weight: float = 1.0,
    main_margin_target: float = 0.2,
    main_margin_temperature: float = 0.1,
    multi_view_loss_mode: str = "avg_logits",
    consistency_weight: float = 0.0,
    consistency_loss_type: str = "mse_probs",
    consistency_gate_mode: str = "none",
    consistency_gate_temperature: float = 0.05,
    consistency_gate_margin_threshold: float = 0.0,
    view_ce_weight: float = 0.0,
    worst_view_ce_weight: float = 0.0,
    consensus_kl_weight: float = 0.0,
    consensus_kl_temperature: float = 1.0,
    wrong_crop_count_weight: float = 0.0,
    wrong_crop_count_temperature: float = 0.5,
    wrong_crop_count_reduction: str = "mean",
    true_label_alignment_weight: float = 0.0,
    true_label_alignment_tolerance: float = 0.02,
    true_label_alignment_min_teacher_prob: float = 0.25,
    feature_consistency_weight: float = 0.0,
    feature_consistency_loss_type: str = "cosine",
    attention_masks: torch.Tensor | None = None,
    use_mask_attention: bool = False,
    mask_context_images: torch.Tensor | None = None,
    use_mask_context_branch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Handles both regular single-view and multi-view images.

    Multi-view modes:
        avg_logits:
            CE(mean(raw logits), label)

        avg_logits_consistency / avg_logits_plus_consistency:
            CE(mean(raw logits), label)
            + consistency_weight * consistency_loss

        avg_probs:
            CE(log(mean(softmax(view logits))), label)

        avg_probs_consistency / avg_probs_plus_consistency:
            CE(log(mean(softmax(view logits))), label)
            + consistency_weight * consistency_loss

    Optional auxiliary loss:
        view_ce_weight > 0:
            Adds supervised CE loss on each crop view separately:

            view_ce_loss = mean_i CE(logits_view_i, label)

        worst_view_ce_weight > 0:
            Adds supervised CE loss on the worst crop view per image:

            worst_view_ce_loss = mean_b max_i CE(logits_view_i, label)

            total_loss = main_loss
                       + consistency_weight * consistency_loss
                       + view_ce_weight * view_ce_loss
                       + worst_view_ce_weight * worst_view_ce_loss

    Returns:
        loss:
            total loss used for backward / validation

        logits_for_pred:
            scores used for argmax metrics, shape (B,K).
            In avg_logits modes this is averaged logits.
            In avg_probs modes this is log(avg_probs).

        preds:
            predicted class indices, shape (B,)

        loss_parts:
            dict with scalar tensors:
                main_loss
                consistency_loss
                view_ce_loss
                total_loss
                num_views
    """
    if _is_multiview_images(images):
        if masks is not None:
            raise ValueError("Multi-view crop training does not support mask tensor input.")

        multi_view_loss_mode = str(multi_view_loss_mode or "avg_logits").strip().lower()
        main_loss_type = str(main_loss_type or "ce").strip().lower()
        main_loss_weight = float(main_loss_weight)
        main_margin_target = float(main_margin_target)
        main_margin_temperature = float(main_margin_temperature)
        consistency_weight = float(consistency_weight)
        consistency_gate_mode = str(consistency_gate_mode or "none").strip().lower()
        consistency_gate_temperature = float(consistency_gate_temperature)
        consistency_gate_margin_threshold = float(consistency_gate_margin_threshold)
        view_ce_weight = float(view_ce_weight)
        worst_view_ce_weight = float(worst_view_ce_weight)
        consensus_kl_weight = float(consensus_kl_weight)
        consensus_kl_temperature = float(consensus_kl_temperature)
        wrong_crop_count_weight = float(wrong_crop_count_weight)
        wrong_crop_count_temperature = float(wrong_crop_count_temperature)
        wrong_crop_count_reduction = str(
            wrong_crop_count_reduction or "mean"
        ).strip().lower()
        true_label_alignment_weight = float(true_label_alignment_weight)
        true_label_alignment_tolerance = float(true_label_alignment_tolerance)
        true_label_alignment_min_teacher_prob = float(
            true_label_alignment_min_teacher_prob
        )

        allowed_modes = {
            "avg_logits",
            "avg_logits_consistency",
            "avg_logits_plus_consistency",
            "avg_probs",
            "avg_probs_consistency",
            "avg_probs_plus_consistency",
        }

        if multi_view_loss_mode not in allowed_modes:
            raise ValueError(
                f"Unknown multi_view_loss_mode={multi_view_loss_mode!r}. "
                f"Expected one of: {sorted(allowed_modes)}"
            )

        allowed_main_loss_types = {
            "ce",
            "avg_margin_hinge",
            "margin_hinge",
            "avg_margin_softplus",
            "margin_softplus",
        }
        if main_loss_type not in allowed_main_loss_types:
            raise ValueError(
                f"Unknown main_loss_type={main_loss_type!r}. "
                f"Expected one of: {sorted(allowed_main_loss_types)}"
            )

        allowed_consistency_gate_modes = {
            "none",
            "true_margin",
            "true_prob_margin",
        }
        if consistency_gate_mode not in allowed_consistency_gate_modes:
            raise ValueError(
                f"Unknown consistency_gate_mode={consistency_gate_mode!r}. "
                f"Expected one of: {sorted(allowed_consistency_gate_modes)}"
            )

        feature_consistency_weight = float(feature_consistency_weight)

        if feature_consistency_weight > 0.0:
            logits_views_raw, feature_views_raw = _forward_multiview_logits_and_features(
                model=model,
                images=images,
                meta_vecs=meta_vecs,
                use_meta=use_meta,
                attention_masks=attention_masks,
                use_mask_attention=use_mask_attention,
                mask_context_images=mask_context_images,
                use_mask_context_branch=use_mask_context_branch,
            )
        else:
            logits_views_raw = _forward_multiview_logits(
                model=model,
                images=images,
                meta_vecs=meta_vecs,
                use_meta=use_meta,
                attention_masks=attention_masks,
                use_mask_attention=use_mask_attention,
                mask_context_images=mask_context_images,
                use_mask_context_branch=use_mask_context_branch,
            )
            feature_views_raw = None

        if multi_view_loss_mode in {
            "avg_probs",
            "avg_probs_consistency",
            "avg_probs_plus_consistency",
        }:
            aggregation = "avg_probs"
        else:
            aggregation = "avg_logits"

        aggregated_raw = _aggregate_multiview_logits(
            logits_views_raw,
            aggregation=aggregation,
        )

        # ------------------------------------------------------------
        # Main classification loss on the aggregated prediction
        # ------------------------------------------------------------
        if aggregation == "avg_probs":
            # aggregated_raw is log(avg_probs), not raw logits.
            #
            # CrossEntropyLoss is still valid here because log(avg_probs)
            # is already a normalized log-probability distribution.
            #
            # This mode is intended for:
            #   loss_name="ce"
            #   logit_adjustment="none"
            if not isinstance(loss_fn, nn.CrossEntropyLoss):
                raise ValueError(
                    "multi_view_loss_mode='avg_probs' currently supports "
                    "loss_name='ce' only."
                )

            logits_for_loss = aggregated_raw
            if main_loss_type == "ce":
                main_loss = loss_fn(logits_for_loss, labels)
            elif main_loss_type in {"avg_margin_hinge", "margin_hinge"}:
                main_loss = _compute_margin_hinge_loss(
                    logits_for_loss,
                    labels,
                    margin_target=main_margin_target,
                )
            else:
                main_loss = _compute_margin_softplus_loss(
                    logits_for_loss,
                    labels,
                    margin_target=main_margin_target,
                    temperature=main_margin_temperature,
                )

            logits_for_pred = aggregated_raw.detach()
            preds = logits_for_pred.argmax(1)

        else:
            logits_for_loss = train_logits_tf(aggregated_raw)
            if main_loss_type == "ce":
                main_loss = loss_fn(logits_for_loss, labels)
            elif main_loss_type in {"avg_margin_hinge", "margin_hinge"}:
                main_loss = _compute_margin_hinge_loss(
                    logits_for_loss,
                    labels,
                    margin_target=main_margin_target,
                )
            else:
                main_loss = _compute_margin_softplus_loss(
                    logits_for_loss,
                    labels,
                    margin_target=main_margin_target,
                    temperature=main_margin_temperature,
                )

            logits_for_pred = eval_logits_tf(aggregated_raw.detach())
            preds = logits_for_pred.argmax(1)

        view_preds = logits_views_raw.detach().argmax(dim=-1)
        view_correct_counts = view_preds.eq(labels.view(-1, 1)).sum(dim=1).float()
        crop_correct_fraction = view_correct_counts / float(logits_views_raw.shape[1])

        # ------------------------------------------------------------
        # Optional consistency loss across views
        # ------------------------------------------------------------
        consistency_loss = logits_views_raw.new_tensor(0.0)
        consistency_gate_mean = logits_views_raw.new_tensor(1.0)

        if multi_view_loss_mode in {
            "avg_logits_consistency",
            "avg_logits_plus_consistency",
            "avg_probs_consistency",
            "avg_probs_plus_consistency",
        } and consistency_weight > 0.0:
            consistency_loss_per_sample = _compute_consistency_loss_per_sample(
                logits_views_raw,
                consistency_loss_type=consistency_loss_type,
            )

            if consistency_gate_mode in {"true_margin", "true_prob_margin"}:
                consistency_gate = _compute_true_margin_consistency_gate(
                    logits_for_loss.detach(),
                    labels,
                    temperature=consistency_gate_temperature,
                    margin_threshold=consistency_gate_margin_threshold,
                )
                consistency_loss = (
                    consistency_loss_per_sample * consistency_gate
                ).mean()
                consistency_gate_mean = consistency_gate.mean()
            else:
                consistency_loss = consistency_loss_per_sample.mean()

        # ------------------------------------------------------------
        # Optional supervised CE on each view separately
        # ------------------------------------------------------------
        view_ce_loss = logits_views_raw.new_tensor(0.0)

        if view_ce_weight > 0.0:
            view_ce_loss = _compute_view_ce_loss(
                logits_views_raw,
                labels,
                loss_fn=loss_fn,
                train_logits_tf=train_logits_tf,
            )

        # ------------------------------------------------------------
        # Optional supervised CE on the worst crop view per image
        # ------------------------------------------------------------
        worst_view_ce_loss = logits_views_raw.new_tensor(0.0)

        if worst_view_ce_weight > 0.0:
            worst_view_ce_loss = _compute_worst_view_ce_loss(
                logits_views_raw,
                labels,
                loss_fn=loss_fn,
                train_logits_tf=train_logits_tf,
            )

        # ------------------------------------------------------------
        # Optional view-to-consensus KL distillation
        # ------------------------------------------------------------
        consensus_kl_loss = logits_views_raw.new_tensor(0.0)

        if consensus_kl_weight > 0.0:
            consensus_kl_loss = _compute_consensus_kl_loss(
                logits_views_raw,
                temperature=consensus_kl_temperature,
            )

        # ------------------------------------------------------------
        # Optional soft wrong-crop-count loss
        # ------------------------------------------------------------
        wrong_crop_count_loss = logits_views_raw.new_tensor(0.0)

        if wrong_crop_count_weight > 0.0:
            wrong_crop_count_loss = _compute_wrong_crop_count_loss(
                logits_views_raw,
                labels,
                temperature=wrong_crop_count_temperature,
                reduction=wrong_crop_count_reduction,
            )

        # ------------------------------------------------------------
        # Optional true-label alignment across crop views
        # ------------------------------------------------------------
        true_label_alignment_loss = logits_views_raw.new_tensor(0.0)

        if true_label_alignment_weight > 0.0:
            true_label_alignment_loss = _compute_true_label_alignment_loss(
                logits_views_raw,
                labels,
                tolerance=true_label_alignment_tolerance,
                min_teacher_prob=true_label_alignment_min_teacher_prob,
            )

        # ------------------------------------------------------------
        # Optional feature consistency loss across views
        # ------------------------------------------------------------
        feature_consistency_loss = logits_views_raw.new_tensor(0.0)

        if feature_consistency_weight > 0.0:
            if feature_views_raw is None:
                raise RuntimeError(
                    "feature_views_raw is None but feature_consistency_weight > 0."
                )

            feature_consistency_loss = _compute_feature_consistency_loss(
                feature_views_raw,
                feature_consistency_loss_type=feature_consistency_loss_type,
                logits_views=logits_views_raw,
                labels=labels,
            )

        total_loss = (
            main_loss_weight * main_loss
            + consistency_weight * consistency_loss
            + view_ce_weight * view_ce_loss
            + worst_view_ce_weight * worst_view_ce_loss
            + consensus_kl_weight * consensus_kl_loss
            + wrong_crop_count_weight * wrong_crop_count_loss
            + true_label_alignment_weight * true_label_alignment_loss
            + feature_consistency_weight * feature_consistency_loss
        )

        loss_parts = {
            "main_loss": main_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "consistency_gate_mean": consistency_gate_mean.detach(),
            "view_ce_loss": view_ce_loss.detach(),
            "worst_view_ce_loss": worst_view_ce_loss.detach(),
            "consensus_kl_loss": consensus_kl_loss.detach(),
            "wrong_crop_count_loss": wrong_crop_count_loss.detach(),
            "true_label_alignment_loss": true_label_alignment_loss.detach(),
            "feature_consistency_loss": feature_consistency_loss.detach(),
            "total_loss": total_loss.detach(),
            "num_views": torch.tensor(
                float(logits_views_raw.shape[1]),
                device=logits_views_raw.device,
            ),
            "view_correct_counts": view_correct_counts.detach(),
            "crop_correct_fraction": crop_correct_fraction.detach(),
        }

        return total_loss, logits_for_pred, preds, loss_parts

    if bool(use_mask_attention) or bool(use_mask_context_branch):
        raw_logits = _forward_model_optional_attention(
            model=model,
            images=images,
            meta_vecs=meta_vecs,
            use_meta=use_meta,
            attention_masks=attention_masks,
            use_mask_attention=use_mask_attention,
            mask_context_images=mask_context_images,
            use_mask_context_branch=use_mask_context_branch,
        )
    else:
        raw_logits = forward_flexible(
            model,
            images,
            meta_vecs,
            use_meta=use_meta,
            masks=masks,
            use_mask=use_mask,
            mask_mode=mask_mode,
        )

    logits_for_loss = train_logits_tf(raw_logits)
    main_loss_type = str(main_loss_type or "ce").strip().lower()
    main_loss_weight = float(main_loss_weight)
    main_margin_target = float(main_margin_target)
    if main_loss_type == "ce":
        main_loss = loss_fn(logits_for_loss, labels)
    elif main_loss_type in {"avg_margin_hinge", "margin_hinge"}:
        main_loss = _compute_margin_hinge_loss(
            logits_for_loss,
            labels,
            margin_target=main_margin_target,
        )
    elif main_loss_type in {"avg_margin_softplus", "margin_softplus"}:
        main_loss = _compute_margin_softplus_loss(
            logits_for_loss,
            labels,
            margin_target=main_margin_target,
            temperature=main_margin_temperature,
        )
    else:
        raise ValueError(
            f"Unknown main_loss_type={main_loss_type!r}. "
            "Expected 'ce', 'avg_margin_hinge', or 'avg_margin_softplus'."
        )
    total_loss = main_loss_weight * main_loss

    logits_for_pred = eval_logits_tf(raw_logits.detach())
    preds = logits_for_pred.argmax(1)
    crop_correct_fraction = preds.detach().eq(labels).float()

    loss_parts = {
        "main_loss": main_loss.detach(),
        "consistency_loss": raw_logits.new_tensor(0.0),
        "view_ce_loss": raw_logits.new_tensor(0.0),
        "worst_view_ce_loss": raw_logits.new_tensor(0.0),
        "consensus_kl_loss": raw_logits.new_tensor(0.0),
        "wrong_crop_count_loss": raw_logits.new_tensor(0.0),
        "true_label_alignment_loss": raw_logits.new_tensor(0.0),
        "feature_consistency_loss": raw_logits.new_tensor(0.0),
        "total_loss": total_loss.detach(),
        "num_views": raw_logits.new_tensor(1.0),
        "view_correct_counts": crop_correct_fraction.detach(),
        "crop_correct_fraction": crop_correct_fraction.detach(),
    }

    return total_loss, logits_for_pred, preds, loss_parts

# ============================================================
# Training
# ============================================================

def _iter_backbone_modules(model: nn.Module):
    for attr_name in ("backbone", "mask_context_backbone"):
        module = getattr(model, attr_name, None)
        if module is not None:
            yield module


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for module in _iter_backbone_modules(model):
        for param in module.parameters():
            param.requires_grad = trainable


def _set_frozen_backbones_eval(model: nn.Module) -> None:
    for module in _iter_backbone_modules(model):
        module.eval()


def _build_discriminative_adamw(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
) -> torch.optim.Optimizer:
    backbone_lr = float(backbone_lr) if backbone_lr is not None else None
    head_lr = float(head_lr) if head_lr is not None else float(lr)

    if backbone_lr is None or backbone_lr <= 0.0:
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=weight_decay)

    backbone_param_ids = set()
    backbone_params = []
    for module in _iter_backbone_modules(model):
        for param in module.parameters():
            param_id = id(param)
            if param_id not in backbone_param_ids:
                backbone_param_ids.add(param_id)
                backbone_params.append(param)

    head_params = [
        param
        for param in model.parameters()
        if id(param) not in backbone_param_ids
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    if not param_groups:
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=weight_decay)

    print(
        "[optimizer] discriminative AdamW | "
        f"backbone_lr={backbone_lr:.8g} | head_lr={head_lr:.8g} | "
        f"backbone_params={len(backbone_params)} | head_params={len(head_params)}"
    )
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_efficientnet_with_meta(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    *,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
    train_logits_tf: Callable,
    eval_logits_tf: Callable,
    loss_fn: nn.Module,
    label2idx: dict,
    target_class: str = "mel",
    early_stopping_metric: str = "balanced_acc",
    main_loss_type: str = "ce",
    main_loss_weight: float = 1.0,
    main_margin_target: float = 0.2,
    main_margin_temperature: float = 0.1,
    use_amp: bool = True,
    seed: int = 0,
    use_meta: bool = True,
    use_mask: bool = False,
    mask_mode: str = "multiply",
    input_source: str = "original",
    multi_view_loss_mode: str = "avg_logits",
    consistency_weight: float = 0.0,
    consistency_loss_type: str = "mse_probs",
    consistency_gate_mode: str = "none",
    consistency_gate_temperature: float = 0.05,
    consistency_gate_margin_threshold: float = 0.0,
    view_ce_weight: float = 0.0,
    worst_view_ce_weight: float = 0.0,
    consensus_kl_weight: float = 0.0,
    consensus_kl_temperature: float = 1.0,
    wrong_crop_count_weight: float = 0.0,
    wrong_crop_count_temperature: float = 0.5,
    wrong_crop_count_reduction: str = "mean",
    true_label_alignment_weight: float = 0.0,
    true_label_alignment_tolerance: float = 0.02,
    true_label_alignment_min_teacher_prob: float = 0.25,
    feature_consistency_weight: float = 0.0,
    feature_consistency_loss_type: str = "cosine",
    use_mask_attention: bool = False,
    use_mask_context_branch: bool = False,
    train_microbatch_size: int | None = None,
    freeze_backbone_epochs: int = 0,
):
    """
    input_source:
        "original" -> regular EfficientNet behavior.
        "crop"     -> images come from HAMMaskedBackgroundDataset second branch.

    Multi-view behavior:
        If input_source="crop" and the dataset returns images with shape:
            (B, V, C, H, W)

        Then:
            1. Forward all B*V views through EfficientNet.
            2. Reshape logits to (B, V, num_classes).
            3. Aggregate the V view predictions according to multi_view_loss_mode.
            4. Compute classification loss on the aggregated prediction.
            5. Optionally add consistency loss across views.
            6. Optionally add supervised CE loss for each view separately.

    multi_view_loss_mode:
        "avg_logits":
            CE(mean_logits, labels)

        "avg_logits_consistency":
            CE(mean_logits, labels) + consistency_weight * consistency_loss

        "avg_probs":
            CE(log(mean probabilities), labels)

        "avg_probs_consistency":
            CE(log(mean probabilities), labels) + consistency_weight * consistency_loss

    view_ce_weight:
        If > 0, adds supervised per-view CE:
            view_ce_weight * mean_i CE(logits_view_i, labels)

    worst_view_ce_weight:
        If > 0, adds supervised CE on the worst crop view of each image:
            worst_view_ce_weight * mean_b max_i CE(logits_view_i, labels)

    consensus_kl_weight:
        If > 0, adds view-to-consensus KL distillation:
            consensus_kl_weight * mean_i KL(Pavg.detach() || Pview_i)

    wrong_crop_count_weight:
        If > 0, adds a differentiable proxy for wrong crop views:
            wrong_crop_count_weight * mean_i sigmoid(
                (best_wrong_logit_i - true_logit_i) / temperature
            )

        The exact crop-view reduction is controlled by
        wrong_crop_count_reduction:
            "mean" -> approximate fraction of wrong views.
            "sum"  -> approximate count of wrong views.

    true_label_alignment_weight:
        If > 0, adds prediction-level crop alignment anchored to the true
        label. For each lesion, the crop with the highest true-label
        probability acts as a detached teacher, and other crops are penalized
        when their true-label probability is lower than that teacher by more
        than true_label_alignment_tolerance.
    """
    input_source = str(input_source).strip().lower()
    if input_source not in {"original", "crop"}:
        raise ValueError(
            f"Unknown input_source={input_source!r}. Expected 'original' or 'crop'."
        )

    if input_source == "crop":
        use_mask = False
        mask_mode = "none"

    multi_view_loss_mode = str(multi_view_loss_mode or "avg_logits").strip().lower()
    main_loss_type = str(main_loss_type or "ce").strip().lower()
    main_loss_weight = float(main_loss_weight)
    main_margin_target = float(main_margin_target)
    main_margin_temperature = float(main_margin_temperature)
    if main_loss_type not in {
        "ce",
        "avg_margin_hinge",
        "margin_hinge",
        "avg_margin_softplus",
        "margin_softplus",
    }:
        raise ValueError(
            f"Unknown main_loss_type={main_loss_type!r}. "
            "Expected 'ce', 'avg_margin_hinge', or 'avg_margin_softplus'."
        )
    consistency_weight = float(consistency_weight)
    consistency_loss_type = str(consistency_loss_type or "mse_probs").strip().lower()
    consistency_gate_mode = str(consistency_gate_mode or "none").strip().lower()
    consistency_gate_temperature = float(consistency_gate_temperature)
    consistency_gate_margin_threshold = float(consistency_gate_margin_threshold)
    view_ce_weight = float(view_ce_weight)
    worst_view_ce_weight = float(worst_view_ce_weight)
    consensus_kl_weight = float(consensus_kl_weight)
    consensus_kl_temperature = float(consensus_kl_temperature)
    wrong_crop_count_weight = float(wrong_crop_count_weight)
    wrong_crop_count_temperature = float(wrong_crop_count_temperature)
    wrong_crop_count_reduction = str(
        wrong_crop_count_reduction or "mean"
    ).strip().lower()
    if wrong_crop_count_reduction not in {"mean", "sum"}:
        raise ValueError(
            f"Unknown wrong_crop_count_reduction={wrong_crop_count_reduction!r}. "
            "Expected 'mean' or 'sum'."
        )
    true_label_alignment_weight = float(true_label_alignment_weight)
    true_label_alignment_tolerance = float(true_label_alignment_tolerance)
    true_label_alignment_min_teacher_prob = float(
        true_label_alignment_min_teacher_prob
    )
    feature_consistency_weight = float(feature_consistency_weight)
    feature_consistency_loss_type = str(
        feature_consistency_loss_type or "cosine"
    ).strip().lower()
    use_mask_attention = bool(use_mask_attention)
    use_mask_context_branch = bool(use_mask_context_branch)
    freeze_backbone_epochs = max(0, int(freeze_backbone_epochs))
    train_microbatch_size = (
        int(train_microbatch_size)
        if train_microbatch_size is not None and int(train_microbatch_size) > 0
        else None
    )

    set_seed(seed)
    model = model.to(device)
    if freeze_backbone_epochs > 0:
        _set_backbone_trainable(model, False)
        print(
            f"[seed {seed}] Freezing EfficientNet backbone for "
            f"{freeze_backbone_epochs} warmup epochs"
        )

    optimizer = _build_discriminative_adamw(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        backbone_lr=backbone_lr,
        head_lr=head_lr,
    )
    scaler = amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    history = {
        "epoch": [],
        "train_loss": [],
        "train_main_loss": [],
        "train_consistency_loss": [],
        "train_view_ce_loss": [],
        "train_worst_view_ce_loss": [],
        "train_consensus_kl_loss": [],
        "train_wrong_crop_count_loss": [],
        "train_true_label_alignment_loss": [],
        "train_feature_consistency_loss": [],
        "train_num_views": [],
        "train_balanced_crop_correctness": [],
        "train_acc": [],
        "train_balanced_acc": [],
        "train_macro_f1": [],
        f"train_{target_class}_recall": [],
        f"train_{target_class}_precision": [],
        "val_loss": [],
        "val_main_loss": [],
        "val_consistency_loss": [],
        "val_view_ce_loss": [],
        "val_worst_view_ce_loss": [],
        "val_consensus_kl_loss": [],
        "val_wrong_crop_count_loss": [],
        "val_true_label_alignment_loss": [],
        "val_feature_consistency_loss": [],
        "val_num_views": [],
        "val_balanced_crop_correctness": [],
        "val_acc": [],
        "val_balanced_acc": [],
        "val_macro_f1": [],
        f"val_{target_class}_recall": [],
        f"val_{target_class}_precision": [],
    }

    direction = metric_direction(early_stopping_metric)
    best_metric_value = np.inf if direction == "min" else -np.inf
    best_state = None
    epochs_no_improve = 0

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for ep in range(1, epochs + 1):
        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------
        if freeze_backbone_epochs > 0 and ep == freeze_backbone_epochs + 1:
            _set_backbone_trainable(model, True)
            print(f"[seed {seed}] Unfreezing EfficientNet backbone at epoch {ep}")

        model.train()
        if freeze_backbone_epochs > 0 and ep <= freeze_backbone_epochs:
            _set_frozen_backbones_eval(model)

        tr_loss_sum, tr_main_loss_sum, tr_consistency_loss_sum, tr_view_ce_loss_sum = 0.0, 0.0, 0.0, 0.0
        tr_worst_view_ce_loss_sum = 0.0
        tr_consensus_kl_loss_sum = 0.0
        tr_wrong_crop_count_loss_sum = 0.0
        tr_true_label_alignment_loss_sum = 0.0
        tr_feature_consistency_loss_sum = 0.0
        tr_num_views_sum = 0.0
        tr_n = 0
        tr_true, tr_pred = [], []
        tr_crop_correct_fraction = []

        for batch in train_loader:
            if input_source == "crop":
                images, meta_vecs, labels, extras = unpack_batch_crop_only(
                    batch,
                    device,
                    use_meta=use_meta,
                )
                masks = None
                attention_masks = (
                    extras.get("attention_mask")
                    if (use_mask_attention and isinstance(extras, dict))
                    else None
                )
                mask_context_images = (
                    extras.get("mask_context_image")
                    if (use_mask_context_branch and isinstance(extras, dict))
                    else None
                )
            else:
                images, meta_vecs, labels, masks = unpack_batch_4(batch, device)
                attention_masks = None
                mask_context_images = None

            batch_n = labels.size(0)
            microbatch_size = min(
                train_microbatch_size or batch_n,
                batch_n,
            )
            optimizer.zero_grad(set_to_none=True)

            for micro_start in range(0, batch_n, microbatch_size):
                micro_end = min(micro_start + microbatch_size, batch_n)
                micro_n = micro_end - micro_start

                micro_images = images[micro_start:micro_end]
                micro_meta_vecs = (
                    meta_vecs[micro_start:micro_end]
                    if meta_vecs is not None
                    else None
                )
                micro_labels = labels[micro_start:micro_end]
                micro_masks = (
                    masks[micro_start:micro_end]
                    if masks is not None
                    else None
                )
                micro_attention_masks = (
                    attention_masks[micro_start:micro_end]
                    if attention_masks is not None
                    else None
                )
                micro_mask_context_images = (
                    mask_context_images[micro_start:micro_end]
                    if mask_context_images is not None
                    else None
                )

                with amp.autocast(
                    device_type=autocast_device,
                    enabled=(use_amp and device.type == "cuda"),
                ):
                    loss, logits_for_pred, preds, loss_parts = _compute_logits_loss_and_pred(
                        model=model,
                        images=micro_images,
                        meta_vecs=micro_meta_vecs,
                        labels=micro_labels,
                        use_meta=use_meta,
                        use_mask=use_mask,
                        masks=micro_masks,
                        mask_mode=mask_mode,
                        train_logits_tf=train_logits_tf,
                        eval_logits_tf=eval_logits_tf,
                        loss_fn=loss_fn,
                        main_loss_type=main_loss_type,
                        main_loss_weight=main_loss_weight,
                        main_margin_target=main_margin_target,
                        main_margin_temperature=main_margin_temperature,
                        multi_view_loss_mode=multi_view_loss_mode,
                        consistency_weight=consistency_weight,
                        consistency_loss_type=consistency_loss_type,
                        consistency_gate_mode=consistency_gate_mode,
                        consistency_gate_temperature=consistency_gate_temperature,
                        consistency_gate_margin_threshold=consistency_gate_margin_threshold,
                        view_ce_weight=view_ce_weight,
                        worst_view_ce_weight=worst_view_ce_weight,
                        consensus_kl_weight=consensus_kl_weight,
                        consensus_kl_temperature=consensus_kl_temperature,
                        wrong_crop_count_weight=wrong_crop_count_weight,
                        wrong_crop_count_temperature=wrong_crop_count_temperature,
                        wrong_crop_count_reduction=wrong_crop_count_reduction,
                        true_label_alignment_weight=true_label_alignment_weight,
                        true_label_alignment_tolerance=true_label_alignment_tolerance,
                        true_label_alignment_min_teacher_prob=true_label_alignment_min_teacher_prob,
                        feature_consistency_weight=feature_consistency_weight,
                        feature_consistency_loss_type=feature_consistency_loss_type,
                        attention_masks=micro_attention_masks,
                        use_mask_attention=use_mask_attention,
                        mask_context_images=micro_mask_context_images,
                        use_mask_context_branch=use_mask_context_branch,
                    )

                # Weight each microbatch by its share of the DataLoader batch so
                # one optimizer step represents the original batch of 16.
                scaler.scale(loss * (micro_n / batch_n)).backward()

                tr_loss_sum += float(loss_parts["total_loss"].item()) * micro_n
                tr_main_loss_sum += float(loss_parts["main_loss"].item()) * micro_n
                tr_consistency_loss_sum += (
                    float(loss_parts["consistency_loss"].item()) * micro_n
                )
                tr_view_ce_loss_sum += (
                    float(loss_parts["view_ce_loss"].item()) * micro_n
                )
                tr_worst_view_ce_loss_sum += (
                    float(loss_parts["worst_view_ce_loss"].item()) * micro_n
                )
                tr_consensus_kl_loss_sum += (
                    float(loss_parts["consensus_kl_loss"].item()) * micro_n
                )
                tr_wrong_crop_count_loss_sum += (
                    float(loss_parts["wrong_crop_count_loss"].item()) * micro_n
                )
                tr_true_label_alignment_loss_sum += (
                    float(loss_parts["true_label_alignment_loss"].item()) * micro_n
                )
                tr_feature_consistency_loss_sum += (
                    float(loss_parts["feature_consistency_loss"].item()) * micro_n
                )
                tr_num_views_sum += float(loss_parts["num_views"].item()) * micro_n
                tr_n += micro_n

                tr_true.append(micro_labels.detach().cpu().numpy())
                tr_pred.append(preds.detach().cpu().numpy())
                tr_crop_correct_fraction.append(
                    loss_parts["crop_correct_fraction"].detach().cpu().numpy()
                )

            scaler.step(optimizer)
            scaler.update()

        tr_true = np.concatenate(tr_true)
        tr_pred = np.concatenate(tr_pred)
        tr_crop_correct_fraction = np.concatenate(tr_crop_correct_fraction)

        train_loss = tr_loss_sum / tr_n if tr_n > 0 else 0.0
        train_main_loss = tr_main_loss_sum / tr_n if tr_n > 0 else 0.0
        train_consistency_loss = tr_consistency_loss_sum / tr_n if tr_n > 0 else 0.0
        train_view_ce_loss = tr_view_ce_loss_sum / tr_n if tr_n > 0 else 0.0
        train_worst_view_ce_loss = (
            tr_worst_view_ce_loss_sum / tr_n if tr_n > 0 else 0.0
        )
        train_consensus_kl_loss = (
            tr_consensus_kl_loss_sum / tr_n if tr_n > 0 else 0.0
        )
        train_wrong_crop_count_loss = (
            tr_wrong_crop_count_loss_sum / tr_n if tr_n > 0 else 0.0
        )
        train_true_label_alignment_loss = (
            tr_true_label_alignment_loss_sum / tr_n if tr_n > 0 else 0.0
        )
        train_feature_consistency_loss = (
            tr_feature_consistency_loss_sum / tr_n if tr_n > 0 else 0.0
        )
        train_num_views = tr_num_views_sum / tr_n if tr_n > 0 else 1.0

        train_acc = float(accuracy_score(tr_true, tr_pred))
        train_imb = compute_imbalanced_metrics(tr_true, tr_pred, label2idx, target_class)
        train_balanced_crop_correctness = compute_balanced_crop_correctness(
            tr_true,
            tr_crop_correct_fraction,
            label2idx,
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------
        model.eval()
        va_loss_sum, va_main_loss_sum, va_consistency_loss_sum, va_view_ce_loss_sum = 0.0, 0.0, 0.0, 0.0
        va_worst_view_ce_loss_sum = 0.0
        va_consensus_kl_loss_sum = 0.0
        va_wrong_crop_count_loss_sum = 0.0
        va_true_label_alignment_loss_sum = 0.0
        va_feature_consistency_loss_sum = 0.0
        va_num_views_sum = 0.0
        va_n = 0
        va_true, va_pred = [], []
        va_crop_correct_fraction = []

        with torch.no_grad():
            for batch in val_loader:
                if input_source == "crop":
                    images, meta_vecs, labels, extras = unpack_batch_crop_only(
                        batch,
                        device,
                        use_meta=use_meta,
                    )
                    masks = None
                    attention_masks = (
                        extras.get("attention_mask")
                        if (use_mask_attention and isinstance(extras, dict))
                        else None
                    )
                    mask_context_images = (
                        extras.get("mask_context_image")
                        if (use_mask_context_branch and isinstance(extras, dict))
                        else None
                    )
                else:
                    images, meta_vecs, labels, masks = unpack_batch_4(batch, device)
                    attention_masks = None
                    mask_context_images = None

                with amp.autocast(
                    device_type=autocast_device,
                    enabled=(use_amp and device.type == "cuda"),
                ):
                    loss, logits_for_pred, preds, loss_parts = _compute_logits_loss_and_pred(
                        model=model,
                        images=images,
                        meta_vecs=meta_vecs,
                        labels=labels,
                        use_meta=use_meta,
                        use_mask=use_mask,
                        masks=masks,
                        mask_mode=mask_mode,
                        train_logits_tf=train_logits_tf,
                        eval_logits_tf=eval_logits_tf,
                        loss_fn=loss_fn,
                        main_loss_type=main_loss_type,
                        main_loss_weight=main_loss_weight,
                        main_margin_target=main_margin_target,
                        main_margin_temperature=main_margin_temperature,
                        multi_view_loss_mode=multi_view_loss_mode,
                        consistency_weight=consistency_weight,
                        consistency_loss_type=consistency_loss_type,
                        consistency_gate_mode=consistency_gate_mode,
                        consistency_gate_temperature=consistency_gate_temperature,
                        consistency_gate_margin_threshold=consistency_gate_margin_threshold,
                        view_ce_weight=view_ce_weight,
                        worst_view_ce_weight=worst_view_ce_weight,
                        consensus_kl_weight=consensus_kl_weight,
                        consensus_kl_temperature=consensus_kl_temperature,
                        wrong_crop_count_weight=wrong_crop_count_weight,
                        wrong_crop_count_temperature=wrong_crop_count_temperature,
                        wrong_crop_count_reduction=wrong_crop_count_reduction,
                        true_label_alignment_weight=true_label_alignment_weight,
                        true_label_alignment_tolerance=true_label_alignment_tolerance,
                        true_label_alignment_min_teacher_prob=true_label_alignment_min_teacher_prob,
                        feature_consistency_weight=feature_consistency_weight,
                        feature_consistency_loss_type=feature_consistency_loss_type,
                        attention_masks=attention_masks,
                        use_mask_attention=use_mask_attention,
                        mask_context_images=mask_context_images,
                        use_mask_context_branch=use_mask_context_branch,
                    )

                batch_n = labels.size(0)
                va_loss_sum += float(loss_parts["total_loss"].item()) * batch_n
                va_main_loss_sum += float(loss_parts["main_loss"].item()) * batch_n
                va_consistency_loss_sum += float(loss_parts["consistency_loss"].item()) * batch_n
                va_view_ce_loss_sum += float(loss_parts["view_ce_loss"].item()) * batch_n
                va_worst_view_ce_loss_sum += (
                    float(loss_parts["worst_view_ce_loss"].item()) * batch_n
                )
                va_consensus_kl_loss_sum += (
                    float(loss_parts["consensus_kl_loss"].item()) * batch_n
                )
                va_wrong_crop_count_loss_sum += (
                    float(loss_parts["wrong_crop_count_loss"].item()) * batch_n
                )
                va_true_label_alignment_loss_sum += (
                    float(loss_parts["true_label_alignment_loss"].item()) * batch_n
                )
                va_feature_consistency_loss_sum += (
                    float(loss_parts["feature_consistency_loss"].item()) * batch_n
                )
                va_num_views_sum += float(loss_parts["num_views"].item()) * batch_n
                va_n += batch_n

                va_true.append(labels.detach().cpu().numpy())
                va_pred.append(preds.detach().cpu().numpy())
                va_crop_correct_fraction.append(
                    loss_parts["crop_correct_fraction"].detach().cpu().numpy()
                )

        va_true = np.concatenate(va_true)
        va_pred = np.concatenate(va_pred)
        va_crop_correct_fraction = np.concatenate(va_crop_correct_fraction)

        val_loss = va_loss_sum / va_n if va_n > 0 else 0.0
        val_main_loss = va_main_loss_sum / va_n if va_n > 0 else 0.0
        val_consistency_loss = va_consistency_loss_sum / va_n if va_n > 0 else 0.0
        val_view_ce_loss = va_view_ce_loss_sum / va_n if va_n > 0 else 0.0
        val_worst_view_ce_loss = (
            va_worst_view_ce_loss_sum / va_n if va_n > 0 else 0.0
        )
        val_consensus_kl_loss = (
            va_consensus_kl_loss_sum / va_n if va_n > 0 else 0.0
        )
        val_wrong_crop_count_loss = (
            va_wrong_crop_count_loss_sum / va_n if va_n > 0 else 0.0
        )
        val_true_label_alignment_loss = (
            va_true_label_alignment_loss_sum / va_n if va_n > 0 else 0.0
        )
        val_feature_consistency_loss = (
            va_feature_consistency_loss_sum / va_n if va_n > 0 else 0.0
        )
        val_num_views = va_num_views_sum / va_n if va_n > 0 else 1.0

        val_acc = float(accuracy_score(va_true, va_pred))
        val_imb = compute_imbalanced_metrics(va_true, va_pred, label2idx, target_class)
        val_balanced_crop_correctness = compute_balanced_crop_correctness(
            va_true,
            va_crop_correct_fraction,
            label2idx,
        )

        history["epoch"].append(ep)

        history["train_loss"].append(train_loss)
        history["train_main_loss"].append(train_main_loss)
        history["train_consistency_loss"].append(train_consistency_loss)
        history["train_view_ce_loss"].append(train_view_ce_loss)
        history["train_worst_view_ce_loss"].append(train_worst_view_ce_loss)
        history["train_consensus_kl_loss"].append(train_consensus_kl_loss)
        history["train_wrong_crop_count_loss"].append(train_wrong_crop_count_loss)
        history["train_true_label_alignment_loss"].append(
            train_true_label_alignment_loss
        )
        history["train_feature_consistency_loss"].append(train_feature_consistency_loss)
        history["train_num_views"].append(train_num_views)
        history["train_balanced_crop_correctness"].append(
            train_balanced_crop_correctness
        )
        history["train_acc"].append(train_acc)
        history["train_balanced_acc"].append(train_imb["balanced_acc"])
        history["train_macro_f1"].append(train_imb["macro_f1"])
        history[f"train_{target_class}_recall"].append(train_imb[f"{target_class}_recall"])
        history[f"train_{target_class}_precision"].append(train_imb[f"{target_class}_precision"])

        history["val_loss"].append(val_loss)
        history["val_main_loss"].append(val_main_loss)
        history["val_consistency_loss"].append(val_consistency_loss)
        history["val_view_ce_loss"].append(val_view_ce_loss)
        history["val_worst_view_ce_loss"].append(val_worst_view_ce_loss)
        history["val_consensus_kl_loss"].append(val_consensus_kl_loss)
        history["val_wrong_crop_count_loss"].append(val_wrong_crop_count_loss)
        history["val_true_label_alignment_loss"].append(
            val_true_label_alignment_loss
        )
        history["val_feature_consistency_loss"].append(val_feature_consistency_loss)
        history["val_num_views"].append(val_num_views)
        history["val_balanced_crop_correctness"].append(
            val_balanced_crop_correctness
        )
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_imb["balanced_acc"])
        history["val_macro_f1"].append(val_imb["macro_f1"])
        history[f"val_{target_class}_recall"].append(val_imb[f"{target_class}_recall"])
        history[f"val_{target_class}_precision"].append(val_imb[f"{target_class}_precision"])

        print(
            f"[seed {seed}] epoch {ep}/{epochs} | "
            f"input_source={input_source} | "
            f"mv_loss={multi_view_loss_mode} | "
            f"main_loss_type={main_loss_type} | "
            f"main_loss_weight={main_loss_weight:.4f} | "
            f"main_margin_target={main_margin_target:.4f} | "
            f"main_margin_temperature={main_margin_temperature:.4f} | "
            f"cons_w={consistency_weight:.4f} | "
            f"view_ce_w={view_ce_weight:.4f} | "
            f"worst_view_ce_w={worst_view_ce_weight:.4f} | "
            f"consensus_kl_w={consensus_kl_weight:.4f} | "
            f"wrong_crop_count_w={wrong_crop_count_weight:.4f} | "
            f"wrong_crop_count_reduction={wrong_crop_count_reduction} | "
            f"true_label_align_w={true_label_alignment_weight:.4f} | "
            f"feat_cons_w={feature_consistency_weight:.4f} | "
            f"mask_attn={use_mask_attention} | "
            f"mask_ctx={use_mask_context_branch} | "
            f"train_views={train_num_views:.1f}, "
            f"train_loss={train_loss:.4f}, "
            f"train_main={train_main_loss:.4f}, "
            f"train_cons={train_consistency_loss:.6f}, "
            f"train_view_ce={train_view_ce_loss:.6f}, "
            f"train_worst_view_ce={train_worst_view_ce_loss:.6f}, "
            f"train_consensus_kl={train_consensus_kl_loss:.6f}, "
            f"train_wrong_crop_count={train_wrong_crop_count_loss:.6f}, "
            f"train_true_label_align={train_true_label_alignment_loss:.6f}, "
            f"train_feat_cons={train_feature_consistency_loss:.6f}, "
            f"train_acc={train_acc:.3f}, "
            f"train_bal_acc={train_imb['balanced_acc']:.3f}, "
            f"train_bcrop={train_balanced_crop_correctness:.3f}, "
            f"train_f1={train_imb['macro_f1']:.3f} | "
            f"val_views={val_num_views:.1f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_main={val_main_loss:.4f}, "
            f"val_cons={val_consistency_loss:.6f}, "
            f"val_view_ce={val_view_ce_loss:.6f}, "
            f"val_worst_view_ce={val_worst_view_ce_loss:.6f}, "
            f"val_consensus_kl={val_consensus_kl_loss:.6f}, "
            f"val_wrong_crop_count={val_wrong_crop_count_loss:.6f}, "
            f"val_true_label_align={val_true_label_alignment_loss:.6f}, "
            f"val_feat_cons={val_feature_consistency_loss:.6f}, "
            f"val_acc={val_acc:.3f}, "
            f"val_bal_acc={val_imb['balanced_acc']:.3f}, "
            f"val_bcrop={val_balanced_crop_correctness:.3f}, "
            f"val_f1={val_imb['macro_f1']:.3f}"
        )

        current_metric_value = get_val_metric_from_history(
            history,
            early_stopping_metric,
            target_class=target_class,
        )

        improved = (
            current_metric_value < best_metric_value
            if direction == "min"
            else current_metric_value > best_metric_value
        )

        if improved:
            best_metric_value = float(current_metric_value)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= int(patience):
                print(
                    f"[seed {seed}] Early stopping after {ep} epochs "
                    f"(no improvement for {patience} epochs)"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[seed {seed}] best {early_stopping_metric} = {best_metric_value:.4f}")
    return model, float(best_metric_value), history


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_and_save_common(
    *,
    model: nn.Module,
    split_df: pd.DataFrame,
    meta_features: np.ndarray,
    label2idx: dict,
    batch_size: int,
    val_transform,
    out_dir: Path,
    run_name: str,
    device: torch.device,
    logits_tf,
    no_meta: bool = False,
    target_class: str = "mel",
    fused_mask_loader: Optional[FusedMaskLoader] = None,
    mask_mode: str = "none",
    agcwd_transform=None,
    input_source: str = "original",
    img_size: int = 224,
    crop_resize_mode: str = "stretch",
    adaptive_crop_shape: str = "square",
    crop_threshold: float = 0.2,
    crop_margin_ratio: float = 0.05,
    crop_min_area: int = 10,
    crop_fallback_mode: str = "keep_original",
    background_mode: str = "blur",
    blur_kernel_size: int = 11,
    blur_sigma: float = 4.0,
    softmultiply_alpha: float = 0.5,
    mask_power: float = 1.0,
    missing_mask_mode: str = "keep_original",
    debug_mask_pipeline: bool = False,
    eval_crop_mode: str = "fixed",
    eval_crop_policy_choices: Optional[List[Dict[str, Any]]] = None,
    use_mask_attention: bool = False,
    use_mask_context_branch: bool = False,
    mask_context_view_mode: str = "full",
):
    """
    Final evaluation.

    eval_crop_mode:
        fixed:
            One deterministic crop using crop_threshold/crop_margin_ratio.

        tta_logits_avg:
            Several deterministic crops, same model, average logits.

        tta_probs_avg:
            Several deterministic crops, same model, softmax each crop
            and average probabilities.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_source = str(input_source).strip().lower()
    if input_source not in {"original", "crop"}:
        raise ValueError(
            f"Unknown input_source={input_source!r}. Expected 'original' or 'crop'."
        )

    eval_crop_mode = str(eval_crop_mode or "fixed").strip().lower()
    if eval_crop_mode not in {"fixed", "tta_logits_avg", "tta_probs_avg"}:
        raise ValueError(
            f"Unknown eval_crop_mode={eval_crop_mode!r}. "
            "Expected 'fixed', 'tta_logits_avg', or 'tta_probs_avg'."
        )

    model.eval().to(device)
    df_reset = split_df.reset_index(drop=True)
    use_meta = not bool(no_meta)
    use_mask_attention = bool(use_mask_attention)
    use_mask_context_branch = bool(use_mask_context_branch)
    mask_context_view_mode = str(mask_context_view_mode or "full").strip().lower()
    if mask_context_view_mode not in {"full", "crop_aligned"}:
        raise ValueError(
            "mask_context_view_mode must be 'full' or 'crop_aligned', "
            f"got {mask_context_view_mode!r}"
        )

    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []

    if input_source == "crop":
        if fused_mask_loader is None:
            raise ValueError("input_source='crop' requires fused_mask_loader.")

        if eval_crop_mode == "fixed":
            ds = _build_crop_eval_dataset(
                split_df=df_reset,
                meta_features=meta_features,
                label2idx=label2idx,
                fused_mask_loader=fused_mask_loader,
                agcwd_transform=agcwd_transform,
                img_size=img_size,
                crop_resize_mode=crop_resize_mode,
                adaptive_crop_shape=adaptive_crop_shape,
                crop_threshold=crop_threshold,
                crop_margin_ratio=crop_margin_ratio,
                crop_min_area=crop_min_area,
                crop_fallback_mode=crop_fallback_mode,
                background_mode=background_mode,
                blur_kernel_size=blur_kernel_size,
                blur_sigma=blur_sigma,
                softmultiply_alpha=softmultiply_alpha,
                mask_power=mask_power,
                missing_mask_mode=missing_mask_mode,
                debug_mask_pipeline=debug_mask_pipeline,
                debug_max_samples=3,
                return_mask_context_image=use_mask_context_branch,
                mask_context_view_mode=mask_context_view_mode,
            )

            loader = DataLoader(
                ds,
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            for batch in loader:
                images, meta_vecs, labels, extras = unpack_batch_crop_only(
                    batch,
                    device,
                    use_meta=use_meta,
                )

                attention_masks = (
                    extras.get("attention_mask")
                    if (use_mask_attention and isinstance(extras, dict))
                    else None
                )
                mask_context_images = (
                    extras.get("mask_context_image")
                    if (use_mask_context_branch and isinstance(extras, dict))
                    else None
                )
                logits = _forward_model_optional_attention(
                    model=model,
                    images=images,
                    meta_vecs=meta_vecs,
                    use_meta=use_meta,
                    attention_masks=attention_masks,
                    use_mask_attention=use_mask_attention,
                    mask_context_images=mask_context_images,
                    use_mask_context_branch=use_mask_context_branch,
                )
                logits = logits_tf(logits)
                preds = logits.argmax(1)

                y_true_idx.extend(labels.detach().cpu().numpy().tolist())
                y_pred_idx.extend(preds.detach().cpu().numpy().tolist())

        else:
            policies = _normalize_crop_policy_choices(
                eval_crop_policy_choices,
                fallback_threshold=crop_threshold,
                fallback_margin_ratio=crop_margin_ratio,
            )

            print(
                f"[EVAL][{run_name}] eval_crop_mode={eval_crop_mode} | "
                f"n_policies={len(policies)} | policies={policies}"
            )

            loaders = []
            for pi, policy in enumerate(policies):
                ds_pi = _build_crop_eval_dataset(
                    split_df=df_reset,
                    meta_features=meta_features,
                    label2idx=label2idx,
                    fused_mask_loader=fused_mask_loader,
                    agcwd_transform=agcwd_transform,
                    img_size=img_size,
                    crop_resize_mode=crop_resize_mode,
                    adaptive_crop_shape=str(policy.get("adaptive_crop_shape", adaptive_crop_shape)),
                    crop_threshold=float(policy["crop_threshold"]),
                    crop_margin_ratio=(
                        float(policy.get("crop_margin_ratio", crop_margin_ratio))
                        if str(policy.get("crop_mode", "fixed_margin")).lower()
                           == "fixed_margin"
                        else 0.0
                    ),
                    crop_min_area=crop_min_area,
                    single_crop_policy=policy,
                    crop_fallback_mode=crop_fallback_mode,
                    background_mode=background_mode,
                    blur_kernel_size=blur_kernel_size,
                    blur_sigma=blur_sigma,
                    softmultiply_alpha=softmultiply_alpha,
                    mask_power=mask_power,
                    missing_mask_mode=missing_mask_mode,
                    debug_mask_pipeline=(debug_mask_pipeline and pi == 0),
                    debug_max_samples=3,
                    return_mask_context_image=use_mask_context_branch,
                    mask_context_view_mode=mask_context_view_mode,
                )

                loader_pi = DataLoader(
                    ds_pi,
                    batch_size=int(batch_size),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
                loaders.append(loader_pi)

            for batches in zip(*loaders):
                logits_per_policy = []
                labels_ref = None

                for batch in batches:
                    images, meta_vecs, labels, extras = unpack_batch_crop_only(
                        batch,
                        device,
                        use_meta=use_meta,
                    )

                    if labels_ref is None:
                        labels_ref = labels
                    else:
                        if not torch.equal(labels_ref, labels):
                            raise ValueError(
                                "TTA loaders are misaligned: labels differ between crop policies."
                            )

                    attention_masks = (
                        extras.get("attention_mask")
                        if (use_mask_attention and isinstance(extras, dict))
                        else None
                    )
                    mask_context_images = (
                        extras.get("mask_context_image")
                        if (use_mask_context_branch and isinstance(extras, dict))
                        else None
                    )
                    logits = _forward_model_optional_attention(
                        model=model,
                        images=images,
                        meta_vecs=meta_vecs,
                        use_meta=use_meta,
                        attention_masks=attention_masks,
                        use_mask_attention=use_mask_attention,
                        mask_context_images=mask_context_images,
                        use_mask_context_branch=use_mask_context_branch,
                    )
                    logits = logits_tf(logits)
                    logits_per_policy.append(logits)

                stacked_logits = torch.stack(logits_per_policy, dim=0)  # (V, B, K)

                if eval_crop_mode == "tta_logits_avg":
                    final_scores = stacked_logits.mean(dim=0)  # (B, K)

                elif eval_crop_mode == "tta_probs_avg":
                    probs_per_policy = torch.softmax(stacked_logits.float(), dim=-1)
                    final_scores = probs_per_policy.mean(dim=0)  # (B, K)

                else:
                    raise ValueError(f"Unsupported eval_crop_mode={eval_crop_mode!r}")

                preds = final_scores.argmax(1)

                y_true_idx.extend(labels_ref.detach().cpu().numpy().tolist())
                y_pred_idx.extend(preds.detach().cpu().numpy().tolist())

    else:
        use_mask = (fused_mask_loader is not None and mask_mode == "concat")
        effective_mask_mode = mask_mode

        if no_meta:
            ds = SimpleImageOnlyDataset(
                df=df_reset,
                transform=val_transform,
                label_col="dx",
                label2idx=label2idx,
                img_col_name="img_path",
                fused_mask_loader=fused_mask_loader,
                fused_mask_mode=mask_mode,
                rotation_deg=0.0,
                hflip_p=0.0,
                vflip_p=0.0,
                agcwd_transform=agcwd_transform,
            )
        else:
            ds = HAMMultiInputDataset(
                df=df_reset,
                meta_features=meta_features,
                hog_lbp_features=None,
                transform=val_transform,
                label_col="dx",
                label2idx=label2idx,
                use_masked_image=False,
                use_hog_lbp=False,
                mask_col_name="img_path_masked",
                img_col_name="img_path",
                fused_mask_loader=fused_mask_loader,
                fused_mask_mode=mask_mode,
                rotation_deg=0.0,
                hflip_p=0.0,
                vflip_p=0.0,
                agcwd_transform=agcwd_transform,
            )

        loader = DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        for batch in loader:
            images, meta_vecs, labels, masks = unpack_batch_4(batch, device)

            logits = forward_flexible(
                model,
                images,
                meta_vecs,
                use_meta=use_meta,
                masks=masks,
                use_mask=use_mask,
                mask_mode=effective_mask_mode,
            )
            logits = logits_tf(logits)
            preds = logits.argmax(1)

            y_true_idx.extend(labels.detach().cpu().numpy().tolist())
            y_pred_idx.extend(preds.detach().cpu().numpy().tolist())

    paths = save_eval_artifacts(
        model_key="efficientnet",
        run_name=run_name,
        out_dir=out_dir,
        y_true_idx=y_true_idx,
        y_pred_idx=y_pred_idx,
        label2idx=label2idx,
        split_df=df_reset,
        target_class=target_class,
    )

    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    imb = compute_imbalanced_metrics(
        np.array(y_true_idx),
        np.array(y_pred_idx),
        label2idx,
        target_class=target_class,
    )

    return {
        f"{run_name}_accuracy": acc,
        f"{run_name}_balanced_accuracy": float(imb["balanced_acc"]),
        f"{run_name}_macro_f1": float(imb["macro_f1"]),
        f"{run_name}_{target_class}_recall": float(imb[f"{target_class}_recall"]),
        f"{run_name}_{target_class}_precision": float(imb[f"{target_class}_precision"]),
        f"{run_name}_n_samples": int(len(y_true_idx)),
        f"{run_name}_preds_path": paths["preds_path"],
        f"{run_name}_classification_report_path": paths["classification_report_path"],
        f"{run_name}_cm_csv_path": paths["cm_csv_path"],
        f"{run_name}_cm_png_path": paths["cm_png_path"],
        f"{run_name}_eval_crop_mode": eval_crop_mode,
        f"{run_name}_eval_crop_policy_choices": str(eval_crop_policy_choices),
    }


# ============================================================
# Experiment runner
# ============================================================

def run_efficientnet_experiment(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta_features: np.ndarray,
    meta_feature_dim: int,
    out_root: Path,
    device: torch.device,
    seed: int = 0,
    split_seed: int = 0,
    param_grid: Optional[List[Dict[str, Any]]] = None,
    no_meta: bool = False,
    target_class: str = "mel",
    early_stopping_metric: str = "balanced_acc",
) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    all_labels = (
        set(train_df["dx"].unique())
        | set(val_df["dx"].unique())
        | set(test_df["dx"].unique())
    )
    labels = sorted(all_labels)
    label2idx = {l: i for i, l in enumerate(labels)}
    num_classes = len(labels)

    if param_grid is None:
        param_grid = [
            {
                "arch": "efficientnet_b0",
                "pretrained": True,
                "img_size": 224,
                "batch_size": 16,
                "lr": 1e-4,
                "backbone_lr": None,
                "head_lr": None,
                "weight_decay": 1e-4,
                "num_epochs": 50,
                "patience": 7,
                "freeze_backbone_epochs": 0,
                "pin_memory": True,
                "drop_last_train": False,
                "meta_hidden_dim": 128,
                "clf_hidden_dim": 256,
                "dropout": 0.3,

                "input_source": "original",
                "use_crop_as_input": False,

                "use_fused_masks": False,
                "mask_mode": "none",
                "mask_npz_train": None,
                "mask_npz_val": None,
                "mask_npz_test": None,

                "crop_threshold": 0.2,
                "crop_margin_ratio": 0.05,
                "crop_min_area": 10,
                "crop_fallback_mode": "keep_original",
                "crop_resize_mode": "stretch",
                "adaptive_crop_shape": "square",
                "use_mask_attention": False,
                "mask_attention_alpha": 0.2,
                "mask_attention_base_weight": 1.0,
                "use_mask_context_branch": False,
                "mask_context_view_mode": "full",
                "mask_context_branch_mode": "independent",

                "random_crop_policy": False,
                "crop_policy_choices": None,

                "multi_view_crop": False,
                "multi_view_crop_policy_choices": None,
                "multi_view_loss_mode": "avg_logits",
                "consistency_weight": 0.0,
                "consistency_loss_type": "mse_probs",
                "view_ce_weight": 0.0,
                "worst_view_ce_weight": 0.0,
                "consensus_kl_weight": 0.0,
                "consensus_kl_temperature": 1.0,
                "wrong_crop_count_weight": 0.0,
                "wrong_crop_count_temperature": 0.5,
                "wrong_crop_count_reduction": "mean",
                "true_label_alignment_weight": 0.0,
                "true_label_alignment_tolerance": 0.02,
                "true_label_alignment_min_teacher_prob": 0.25,
                "feature_consistency_weight": 0.0,
                "feature_consistency_loss_type": "cosine",

                "eval_crop_mode": "fixed",
                "eval_crop_policy_choices": None,

                "background_mode": "blur",
                "blur_kernel_size": 11,
                "blur_sigma": 4.0,
                "softmultiply_alpha": 0.5,
                "mask_power": 1.0,
                "missing_mask_mode": "keep_original",

                "loss_name": "ce",
                "main_loss_type": "ce",
                "main_loss_weight": 1.0,
                "main_margin_target": 0.2,
                "main_margin_temperature": 0.1,
                "class_weighting": "inverse_freq",
                "label_smoothing": 0.05,
                "logit_adjustment": "none",
                "tau": 1.0,
                "sampler": "none",
                "focal_gamma": 2.0,
                "focal_alpha_mode": "none",

                "rotation_deg": 0.0,
                "hflip": False,
                "vflip": False,
                "hflip_p": 0.0,
                "vflip_p": 0.0,

                "use_color_jitter": False,
                "color_jitter_brightness": 0.0,
                "color_jitter_contrast": 0.0,
                "color_jitter_saturation": 0.0,
                "color_jitter_hue": 0.0,
                "per_view_color_jitter_brightness": False,
                "per_view_color_jitter_contrast": False,
                "per_view_color_jitter_saturation": False,
                "per_view_color_jitter_hue": False,

                "agcwd_preset": "none",
                "debug_mask_pipeline": False,
            }
        ]

    class_counts_np = compute_class_counts(train_df, label2idx, label_col="dx")
    all_rows = []

    for cfg in param_grid:
        agcwd_transform = build_agcwd_from_cfg(cfg)
        print(f"AGCWD preset: {cfg.get('agcwd_preset', 'none')} -> {agcwd_transform}")

        use_crop_as_input = bool(cfg.get("use_crop_as_input", False))
        input_source = "crop" if use_crop_as_input else str(cfg.get("input_source", "original")).lower()

        if input_source not in {"original", "crop"}:
            raise ValueError(
                f"Unknown input_source={input_source!r}. "
                f"Use 'original' or set use_crop_as_input=True."
            )

        use_fused = bool(cfg.get("use_fused_masks", False))
        mask_mode = str(cfg.get("mask_mode", "none")).lower()
        debug_mask_pipeline = bool(cfg.get("debug_mask_pipeline", False))

        img_size = int(cfg["img_size"])
        batch_size = int(cfg["batch_size"])
        eval_batch_size = int(cfg.get("eval_batch_size", batch_size))
        pin_memory = bool(cfg.get("pin_memory", True))
        drop_last_train = bool(cfg.get("drop_last_train", False))
        backbone_lr = cfg.get("backbone_lr", None)
        backbone_lr = None if backbone_lr in {None, "", "none", "None"} else float(backbone_lr)
        head_lr = cfg.get("head_lr", None)
        head_lr = None if head_lr in {None, "", "none", "None"} else float(head_lr)

        crop_threshold = float(cfg.get("crop_threshold", 0.2))
        crop_margin_ratio = float(cfg.get("crop_margin_ratio", 0.05))
        crop_min_area = int(cfg.get("crop_min_area", 10))
        crop_fallback_mode = str(cfg.get("crop_fallback_mode", "keep_original"))
        crop_resize_mode = str(cfg.get("crop_resize_mode", "stretch")).strip().lower()
        adaptive_crop_shape = str(cfg.get("adaptive_crop_shape", "square")).strip().lower()
        use_mask_attention = bool(cfg.get("use_mask_attention", False))
        mask_attention_alpha = float(cfg.get("mask_attention_alpha", 0.2))
        mask_attention_base_weight = float(
            cfg.get("mask_attention_base_weight", 1.0)
        )
        freeze_backbone_epochs = int(cfg.get("freeze_backbone_epochs", 0))
        use_mask_context_branch = bool(cfg.get("use_mask_context_branch", False))
        mask_context_view_mode = str(
            cfg.get("mask_context_view_mode", "full")
        ).strip().lower()
        mask_context_branch_mode = str(
            cfg.get("mask_context_branch_mode", "independent")
        ).strip().lower()
        if mask_context_view_mode not in {"full", "crop_aligned"}:
            raise ValueError(
                "mask_context_view_mode must be 'full' or 'crop_aligned', "
                f"got {mask_context_view_mode!r}"
            )
        if mask_context_branch_mode not in {"independent", "shared"}:
            raise ValueError(
                "mask_context_branch_mode must be 'independent' or 'shared', "
                f"got {mask_context_branch_mode!r}"
            )

        if crop_resize_mode not in {"stretch", "letterbox"}:
            raise ValueError(
                f"crop_resize_mode must be 'stretch' or 'letterbox', got {crop_resize_mode!r}"
            )

        if adaptive_crop_shape not in {"square", "preserve_aspect", "closest_rectangle"}:
            raise ValueError(
                f"adaptive_crop_shape must be 'square', 'preserve_aspect', "
                f"or 'closest_rectangle', got {adaptive_crop_shape!r}"
            )

        random_crop_policy = bool(cfg.get("random_crop_policy", False))
        crop_policy_choices = cfg.get("crop_policy_choices", None)

        multi_view_crop = bool(cfg.get("multi_view_crop", False))
        multi_view_crop_policy_choices = cfg.get("multi_view_crop_policy_choices", None)
        multi_view_loss_mode = str(cfg.get("multi_view_loss_mode", "avg_logits")).strip().lower()
        main_loss_weight = float(cfg.get("main_loss_weight", 1.0))
        consistency_weight = float(cfg.get("consistency_weight", 0.0))
        consistency_loss_type = str(cfg.get("consistency_loss_type", "mse_probs")).strip().lower()
        consistency_gate_mode = str(
            cfg.get("consistency_gate_mode", "none")
        ).strip().lower()
        consistency_gate_temperature = float(
            cfg.get("consistency_gate_temperature", 0.05)
        )
        consistency_gate_margin_threshold = float(
            cfg.get("consistency_gate_margin_threshold", 0.0)
        )
        view_ce_weight = float(cfg.get("view_ce_weight", 0.0))
        worst_view_ce_weight = float(cfg.get("worst_view_ce_weight", 0.0))
        consensus_kl_weight = float(cfg.get("consensus_kl_weight", 0.0))
        consensus_kl_temperature = float(cfg.get("consensus_kl_temperature", 1.0))
        wrong_crop_count_weight = float(cfg.get("wrong_crop_count_weight", 0.0))
        wrong_crop_count_temperature = float(cfg.get("wrong_crop_count_temperature", 0.5))
        wrong_crop_count_reduction = str(
            cfg.get("wrong_crop_count_reduction", "mean")
        ).strip().lower()
        if wrong_crop_count_reduction not in {"mean", "sum"}:
            raise ValueError(
                f"wrong_crop_count_reduction must be 'mean' or 'sum', "
                f"got {wrong_crop_count_reduction!r}"
            )
        true_label_alignment_weight = float(
            cfg.get("true_label_alignment_weight", 0.0)
        )
        true_label_alignment_tolerance = float(
            cfg.get("true_label_alignment_tolerance", 0.02)
        )
        true_label_alignment_min_teacher_prob = float(
            cfg.get("true_label_alignment_min_teacher_prob", 0.25)
        )
        feature_consistency_weight = float(cfg.get("feature_consistency_weight", 0.0))
        feature_consistency_loss_type = str(
            cfg.get("feature_consistency_loss_type", "cosine")
        ).strip().lower()

        per_view_color_jitter_brightness = bool(
            cfg.get("per_view_color_jitter_brightness", False)
        )
        per_view_color_jitter_contrast = bool(
            cfg.get("per_view_color_jitter_contrast", False)
        )
        per_view_color_jitter_saturation = bool(
            cfg.get("per_view_color_jitter_saturation", False)
        )
        per_view_color_jitter_hue = bool(
            cfg.get("per_view_color_jitter_hue", False)
        )

        eval_crop_mode = str(cfg.get("eval_crop_mode", "fixed")).strip().lower()
        eval_crop_policy_choices = cfg.get("eval_crop_policy_choices", None)

        background_mode = str(cfg.get("background_mode", "blur"))
        blur_kernel_size = int(cfg.get("blur_kernel_size", 11))
        blur_sigma = float(cfg.get("blur_sigma", 4.0))
        softmultiply_alpha = float(cfg.get("softmultiply_alpha", 0.5))
        mask_power = float(cfg.get("mask_power", 1.0))
        missing_mask_mode = str(cfg.get("missing_mask_mode", "keep_original"))

        if multi_view_crop and random_crop_policy:
            raise ValueError(
                "multi_view_crop=True and random_crop_policy=True are mutually exclusive. "
                "For the new experiment, use multi_view_crop=True and random_crop_policy=False."
            )

        if multi_view_crop and input_source != "crop":
            raise ValueError("multi_view_crop=True requires input_source='crop'.")

        if use_mask_context_branch and input_source != "crop":
            raise ValueError(
                "use_mask_context_branch=True requires input_source='crop' "
                "(or use_crop_as_input=True), because the branch is built from fused masks."
            )

        if multi_view_crop and not multi_view_crop_policy_choices:
            raise ValueError(
                "multi_view_crop=True requires non-empty multi_view_crop_policy_choices."
            )

        print(
            f"[INFO] run_tag={cfg.get('run_tag')} | "
            f"input_source={input_source} | "
            f"use_crop_as_input={use_crop_as_input} | "
            f"use_fused_masks={use_fused} | "
            f"mask_mode={mask_mode} | "
            f"crop_threshold={crop_threshold} | "
            f"crop_margin_ratio={crop_margin_ratio} | "
            f"crop_min_area={crop_min_area} | "
            f"lr={cfg.get('lr')} | "
            f"backbone_lr={backbone_lr} | "
            f"head_lr={head_lr} | "
            f"crop_resize_mode={crop_resize_mode} | "
            f"adaptive_crop_shape={adaptive_crop_shape} | "
            f"use_mask_attention={use_mask_attention} | "
            f"mask_attention_alpha={mask_attention_alpha} | "
            f"mask_attention_base_weight={mask_attention_base_weight} | "
            f"use_mask_context_branch={use_mask_context_branch} | "
            f"mask_context_view_mode={mask_context_view_mode} | "
            f"mask_context_branch_mode={mask_context_branch_mode} | "
            f"random_crop_policy={random_crop_policy} | "
            f"crop_policy_choices={crop_policy_choices} | "
            f"multi_view_crop={multi_view_crop} | "
            f"multi_view_crop_policy_choices={multi_view_crop_policy_choices} | "
            f"multi_view_loss_mode={multi_view_loss_mode} | "
            f"main_loss_weight={main_loss_weight} | "
            f"consistency_weight={consistency_weight} | "
            f"consistency_loss_type={consistency_loss_type} | "
            f"consistency_gate_mode={consistency_gate_mode} | "
            f"consistency_gate_temperature={consistency_gate_temperature} | "
            f"consistency_gate_margin_threshold={consistency_gate_margin_threshold} | "
            f"view_ce_weight={view_ce_weight} | "
            f"worst_view_ce_weight={worst_view_ce_weight} | "
            f"consensus_kl_weight={consensus_kl_weight} | "
            f"consensus_kl_temperature={consensus_kl_temperature} | "
            f"wrong_crop_count_weight={wrong_crop_count_weight} | "
            f"wrong_crop_count_temperature={wrong_crop_count_temperature} | "
            f"wrong_crop_count_reduction={wrong_crop_count_reduction} | "
            f"true_label_alignment_weight={true_label_alignment_weight} | "
            f"true_label_alignment_tolerance={true_label_alignment_tolerance} | "
            f"true_label_alignment_min_teacher_prob={true_label_alignment_min_teacher_prob} | "
            f"feature_consistency_weight={feature_consistency_weight} | "
            f"feature_consistency_loss_type={feature_consistency_loss_type} | "
            f"freeze_backbone_epochs={freeze_backbone_epochs} | "
            f"eval_crop_mode={eval_crop_mode} | "
            f"eval_crop_policy_choices={eval_crop_policy_choices} | "
            f"per_view_color_jitter_brightness={per_view_color_jitter_brightness} | "
            f"per_view_color_jitter_contrast={per_view_color_jitter_contrast} | "
            f"per_view_color_jitter_saturation={per_view_color_jitter_saturation} | "
            f"per_view_color_jitter_hue={per_view_color_jitter_hue} | "
            f"drop_last_train={drop_last_train} | "
            f"debug_mask_pipeline={debug_mask_pipeline}"
        )

        fused_loader_train = None
        fused_loader_val = None
        fused_loader_test = None

        mask_npz_train = cfg.get("mask_npz_train")
        mask_npz_val = cfg.get("mask_npz_val")
        mask_npz_test = cfg.get("mask_npz_test")

        if input_source == "crop":
            mask_paths = [mask_npz_train, mask_npz_val, mask_npz_test]
            resolved_mask_paths = [
                Path(path).resolve()
                if path is not None and str(path).strip()
                else None
                for path in mask_paths
            ]
            if (
                resolved_mask_paths[0] is not None
                and len(set(resolved_mask_paths)) == 1
            ):
                # Repeated train/val/test splits use one all-images NPZ. Load
                # its decompressed mask array once and share the loader across
                # the three datasets instead of holding three identical copies.
                shared_loader = _load_required_fused_mask_loader(
                    npz_path=resolved_mask_paths[0],
                    split_name="train",
                    split_df=train_df,
                    debug=debug_mask_pipeline,
                    img_col_name="img_path",
                )
                _assert_loader_has_all_split_masks(
                    loader=shared_loader,
                    split_df=val_df,
                    split_name="val",
                    img_col_name="img_path",
                )
                _assert_loader_has_all_split_masks(
                    loader=shared_loader,
                    split_df=test_df,
                    split_name="test",
                    img_col_name="img_path",
                )
                fused_loader_train = shared_loader
                fused_loader_val = shared_loader
                fused_loader_test = shared_loader
                print("Using one shared fused-mask loader for train/val/test.")
            else:
                fused_loader_train = _load_required_fused_mask_loader(
                    npz_path=mask_npz_train,
                    split_name="train",
                    split_df=train_df,
                    debug=debug_mask_pipeline,
                    img_col_name="img_path",
                )
                fused_loader_val = _load_required_fused_mask_loader(
                    npz_path=mask_npz_val,
                    split_name="val",
                    split_df=val_df,
                    debug=debug_mask_pipeline,
                    img_col_name="img_path",
                )
                fused_loader_test = _load_required_fused_mask_loader(
                    npz_path=mask_npz_test,
                    split_name="test",
                    split_df=test_df,
                    debug=debug_mask_pipeline,
                    img_col_name="img_path",
                )

            use_mask = False
            effective_mask_mode = "none"

        elif use_fused and mask_mode != "none":
            fused_loader_train = _maybe_load_optional_fused_mask_loader(
                npz_path=mask_npz_train,
                split_name="train",
            )
            fused_loader_val = _maybe_load_optional_fused_mask_loader(
                npz_path=mask_npz_val,
                split_name="val",
            )
            fused_loader_test = _maybe_load_optional_fused_mask_loader(
                npz_path=mask_npz_test,
                split_name="test",
            )

            use_mask = (fused_loader_train is not None and mask_mode == "concat")
            effective_mask_mode = mask_mode

        else:
            use_mask = False
            effective_mask_mode = "none"

        use_spatial_mask = (
            input_source == "original"
            and use_fused
            and mask_mode not in ("none", "concat")
        )

        tf_rotation_deg = 0.0 if use_spatial_mask else float(cfg.get("rotation_deg", 0.0))
        tf_hflip = False if use_spatial_mask else bool(cfg.get("hflip", False))
        tf_vflip = False if use_spatial_mask else bool(cfg.get("vflip", False))

        train_tf, eval_tf = efficientnet_transforms(
            pretrained=bool(cfg["pretrained"]),
            img_size=img_size,
            rotation_deg=tf_rotation_deg,
            vflip=tf_vflip,
            use_color_jitter=bool(cfg.get("use_color_jitter", False)),
            color_jitter_brightness=float(cfg.get("color_jitter_brightness", 0.0)),
            color_jitter_contrast=float(cfg.get("color_jitter_contrast", 0.0)),
            color_jitter_saturation=float(cfg.get("color_jitter_saturation", 0.0)),
            color_jitter_hue=float(cfg.get("color_jitter_hue", 0.0)),
            hflip=tf_hflip,
        )

        train_logits_tf, eval_logits_tf, loss_fn = build_loss_and_logits_transform(
            loss_name=str(cfg.get("loss_name", "ce")),
            class_weighting=str(cfg.get("class_weighting", "none")),
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
            focal_gamma=float(cfg.get("focal_gamma", 2.0)),
            focal_alpha_mode=str(cfg.get("focal_alpha_mode", "none")),
            logit_adjustment=str(cfg.get("logit_adjustment", "none")),
            tau=float(cfg.get("tau", 1.0)),
            class_counts_np=class_counts_np,
            device=device,
        )
        main_loss_type = str(cfg.get("main_loss_type", "ce")).strip().lower()
        main_margin_target = float(cfg.get("main_margin_target", 0.2))
        main_margin_temperature = float(cfg.get("main_margin_temperature", 0.1))
        if main_loss_type not in {
            "ce",
            "avg_margin_hinge",
            "margin_hinge",
            "avg_margin_softplus",
            "margin_softplus",
        }:
            raise ValueError(
                "main_loss_type must be 'ce', 'avg_margin_hinge', "
                f"or 'avg_margin_softplus', "
                f"got {main_loss_type!r}"
            )

        run_name = build_run_name_compact(
            model_key="efficientnet",
            cfg=cfg,
            no_meta=bool(no_meta),
            run_tag=cfg.get("run_tag"),
        )
        run_dir = make_run_dir(out_root, run_name)

        write_config_json(
            run_dir,
            cfg,
            extras={
                "seed": seed,
                "split_seed": split_seed,
                "no_meta": bool(no_meta),
                "target_class": target_class,
                "early_stopping_metric": early_stopping_metric,
                "input_source": input_source,
                "use_crop_as_input": use_crop_as_input,
                "strict_mask_loading": input_source == "crop",
                "crop_threshold": crop_threshold,
                "crop_margin_ratio": crop_margin_ratio,
                "crop_min_area": crop_min_area,
                "crop_fallback_mode": crop_fallback_mode,
                "use_mask_attention": use_mask_attention,
                "mask_attention_alpha": mask_attention_alpha,
                "mask_attention_base_weight": mask_attention_base_weight,
                "use_mask_context_branch": use_mask_context_branch,
                "mask_context_view_mode": mask_context_view_mode,
                "mask_context_branch_mode": mask_context_branch_mode,
                "random_crop_policy_train_only": random_crop_policy,
                "crop_policy_choices_train_only": crop_policy_choices,
                "multi_view_crop": multi_view_crop,
                "multi_view_crop_policy_choices": multi_view_crop_policy_choices,
                "main_loss_type": main_loss_type,
                "main_loss_weight": main_loss_weight,
                "main_margin_target": main_margin_target,
                "main_margin_temperature": main_margin_temperature,
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
                "freeze_backbone_epochs": freeze_backbone_epochs,
                "multi_view_loss_mode": multi_view_loss_mode,
                "consistency_weight": consistency_weight,
                "consistency_loss_type": consistency_loss_type,
                "consistency_gate_mode": consistency_gate_mode,
                "consistency_gate_temperature": consistency_gate_temperature,
                "consistency_gate_margin_threshold": consistency_gate_margin_threshold,
                "view_ce_weight": view_ce_weight,
                "worst_view_ce_weight": worst_view_ce_weight,
                "consensus_kl_weight": consensus_kl_weight,
                "consensus_kl_temperature": consensus_kl_temperature,
                "wrong_crop_count_weight": wrong_crop_count_weight,
                "wrong_crop_count_temperature": wrong_crop_count_temperature,
                "wrong_crop_count_reduction": wrong_crop_count_reduction,
                "true_label_alignment_weight": true_label_alignment_weight,
                "true_label_alignment_tolerance": true_label_alignment_tolerance,
                "true_label_alignment_min_teacher_prob": true_label_alignment_min_teacher_prob,
                "feature_consistency_weight": feature_consistency_weight,
                "feature_consistency_loss_type": feature_consistency_loss_type,
                "per_view_color_jitter_brightness": per_view_color_jitter_brightness,
                "per_view_color_jitter_contrast": per_view_color_jitter_contrast,
                "per_view_color_jitter_saturation": per_view_color_jitter_saturation,
                "per_view_color_jitter_hue": per_view_color_jitter_hue,
                "eval_crop_mode": eval_crop_mode,
                "eval_crop_policy_choices": eval_crop_policy_choices,
            },
            filename="config.json",
        )

        # ========================================================
        # Datasets
        # ========================================================
        if input_source == "crop":
            if multi_view_crop:
                ds_train = _build_multiview_crop_dataset(
                    split_df=train_df,
                    meta_features=meta_features,
                    label2idx=label2idx,
                    fused_mask_loader=fused_loader_train,
                    agcwd_transform=agcwd_transform,
                    img_size=img_size,
                    crop_resize_mode=crop_resize_mode,
                    adaptive_crop_shape=adaptive_crop_shape,
                    crop_threshold=crop_threshold,
                    crop_margin_ratio=crop_margin_ratio,
                    crop_min_area=crop_min_area,
                    crop_fallback_mode=crop_fallback_mode,
                    background_mode=background_mode,
                    blur_kernel_size=blur_kernel_size,
                    blur_sigma=blur_sigma,
                    softmultiply_alpha=softmultiply_alpha,
                    mask_power=mask_power,
                    missing_mask_mode=missing_mask_mode,
                    debug_mask_pipeline=debug_mask_pipeline,
                    debug_max_samples=6,
                    multi_view_crop_policy_choices=multi_view_crop_policy_choices,
                    use_color_jitter=bool(cfg.get("use_color_jitter", False)),
                    color_jitter_brightness=float(cfg.get("color_jitter_brightness", 0.0)),
                    color_jitter_contrast=float(cfg.get("color_jitter_contrast", 0.0)),
                    color_jitter_saturation=float(cfg.get("color_jitter_saturation", 0.0)),
                    color_jitter_hue=float(cfg.get("color_jitter_hue", 0.0)),
                    per_view_color_jitter_brightness=per_view_color_jitter_brightness,
                    per_view_color_jitter_contrast=per_view_color_jitter_contrast,
                    per_view_color_jitter_saturation=per_view_color_jitter_saturation,
                    per_view_color_jitter_hue=per_view_color_jitter_hue,
                    rotation_deg=float(cfg.get("rotation_deg", 0.0)),
                    hflip_p=float(cfg.get("hflip_p", 0.0)),
                    vflip_p=float(cfg.get("vflip_p", 0.0)),
                    return_mask_context_image=use_mask_context_branch,
                    mask_context_view_mode=mask_context_view_mode,
                )

                ds_val = _build_multiview_crop_dataset(
                    split_df=val_df,
                    meta_features=meta_features,
                    label2idx=label2idx,
                    fused_mask_loader=fused_loader_val,
                    agcwd_transform=agcwd_transform,
                    img_size=img_size,
                    crop_resize_mode=crop_resize_mode,
                    adaptive_crop_shape=adaptive_crop_shape,
                    crop_threshold=crop_threshold,
                    crop_margin_ratio=crop_margin_ratio,
                    crop_min_area=crop_min_area,
                    crop_fallback_mode=crop_fallback_mode,
                    background_mode=background_mode,
                    blur_kernel_size=blur_kernel_size,
                    blur_sigma=blur_sigma,
                    softmultiply_alpha=softmultiply_alpha,
                    mask_power=mask_power,
                    missing_mask_mode=missing_mask_mode,
                    debug_mask_pipeline=debug_mask_pipeline,
                    debug_max_samples=4,
                    multi_view_crop_policy_choices=multi_view_crop_policy_choices,
                    use_color_jitter=False,
                    color_jitter_brightness=0.0,
                    color_jitter_contrast=0.0,
                    color_jitter_saturation=0.0,
                    color_jitter_hue=0.0,
                    per_view_color_jitter_brightness=False,
                    per_view_color_jitter_contrast=False,
                    per_view_color_jitter_saturation=False,
                    per_view_color_jitter_hue=False,
                    rotation_deg=0.0,
                    hflip_p=0.0,
                    vflip_p=0.0,
                    return_mask_context_image=use_mask_context_branch,
                    mask_context_view_mode=mask_context_view_mode,
                )

            else:
                ds_train = HAMMaskedBackgroundDataset(
                    df=train_df,
                    meta_features=meta_features,
                    label_col="dx",
                    label2idx=label2idx,
                    img_col_name="img_path",
                    fused_mask_loader=fused_loader_train,

                    rotation_deg=float(cfg.get("rotation_deg", 0.0)),
                    hflip_p=float(cfg.get("hflip_p", 0.0)),
                    vflip_p=float(cfg.get("vflip_p", 0.0)),

                    agcwd_transform=agcwd_transform,
                    sanity_check_all_ones_masks=False,

                    background_mode=background_mode,
                    blur_kernel_size=blur_kernel_size,
                    blur_sigma=blur_sigma,
                    softmultiply_alpha=softmultiply_alpha,
                    mask_power=mask_power,
                    missing_mask_mode=missing_mask_mode,

                    debug=debug_mask_pipeline,
                    debug_max_samples=6,

                    img_size=img_size,
                    crop_resize_mode=crop_resize_mode,
                    adaptive_crop_shape=adaptive_crop_shape,
                    use_color_jitter=bool(cfg.get("use_color_jitter", False)),
                    color_jitter_brightness=float(cfg.get("color_jitter_brightness", 0.0)),
                    color_jitter_contrast=float(cfg.get("color_jitter_contrast", 0.0)),
                    color_jitter_saturation=float(cfg.get("color_jitter_saturation", 0.0)),
                    color_jitter_hue=float(cfg.get("color_jitter_hue", 0.0)),
                    per_view_color_jitter_brightness=False,
                    per_view_color_jitter_contrast=False,
                    per_view_color_jitter_saturation=False,
                    per_view_color_jitter_hue=False,

                    use_crop=True,
                    crop_threshold=crop_threshold,
                    crop_margin_ratio=crop_margin_ratio,
                    crop_min_area=crop_min_area,
                    crop_fallback_mode=crop_fallback_mode,

                    random_crop_policy=random_crop_policy,
                    crop_policy_choices=crop_policy_choices,

                    multi_view_crop=False,
                    multi_view_crop_policy_choices=None,
                    return_mask_context_image=use_mask_context_branch,
                    mask_context_view_mode=mask_context_view_mode,
                )

                ds_val = HAMMaskedBackgroundDataset(
                    df=val_df,
                    meta_features=meta_features,
                    label_col="dx",
                    label2idx=label2idx,
                    img_col_name="img_path",
                    fused_mask_loader=fused_loader_val,

                    rotation_deg=0.0,
                    hflip_p=0.0,
                    vflip_p=0.0,

                    agcwd_transform=agcwd_transform,
                    sanity_check_all_ones_masks=False,

                    background_mode=background_mode,
                    blur_kernel_size=blur_kernel_size,
                    blur_sigma=blur_sigma,
                    softmultiply_alpha=softmultiply_alpha,
                    mask_power=mask_power,
                    missing_mask_mode=missing_mask_mode,

                    debug=debug_mask_pipeline,
                    debug_max_samples=4,

                    img_size=img_size,
                    crop_resize_mode=crop_resize_mode,
                    adaptive_crop_shape=adaptive_crop_shape,
                    use_color_jitter=False,

                    use_crop=True,
                    crop_threshold=crop_threshold,
                    crop_margin_ratio=crop_margin_ratio,
                    crop_min_area=crop_min_area,
                    crop_fallback_mode=crop_fallback_mode,

                    random_crop_policy=False,
                    crop_policy_choices=None,

                    multi_view_crop=False,
                    multi_view_crop_policy_choices=None,
                    return_mask_context_image=use_mask_context_branch,
                    mask_context_view_mode=mask_context_view_mode,
                )

        else:
            if no_meta:
                ds_train = SimpleImageOnlyDataset(
                    df=train_df,
                    transform=train_tf,
                    label_col="dx",
                    label2idx=label2idx,
                    img_col_name="img_path",
                    fused_mask_loader=fused_loader_train,
                    fused_mask_mode=effective_mask_mode,
                    rotation_deg=float(cfg.get("rotation_deg", 0.0)) if use_spatial_mask else 0.0,
                    hflip_p=0.5 if (use_spatial_mask and cfg.get("hflip", False)) else 0.0,
                    vflip_p=0.2 if (use_spatial_mask and cfg.get("vflip", False)) else 0.0,
                    agcwd_transform=agcwd_transform,
                )
                ds_val = SimpleImageOnlyDataset(
                    df=val_df,
                    transform=eval_tf,
                    label_col="dx",
                    label2idx=label2idx,
                    img_col_name="img_path",
                    fused_mask_loader=fused_loader_val,
                    fused_mask_mode=effective_mask_mode,
                    rotation_deg=0.0,
                    hflip_p=0.0,
                    vflip_p=0.0,
                    agcwd_transform=agcwd_transform,
                )
            else:
                ds_train = HAMMultiInputDataset(
                    df=train_df,
                    meta_features=meta_features,
                    hog_lbp_features=None,
                    transform=train_tf,
                    label_col="dx",
                    label2idx=label2idx,
                    use_masked_image=False,
                    use_hog_lbp=False,
                    mask_col_name="img_path_masked",
                    img_col_name="img_path",
                    fused_mask_loader=fused_loader_train,
                    fused_mask_mode=effective_mask_mode,
                    rotation_deg=float(cfg.get("rotation_deg", 0.0)) if use_spatial_mask else 0.0,
                    hflip_p=0.5 if (use_spatial_mask and cfg.get("hflip", False)) else 0.0,
                    vflip_p=0.2 if (use_spatial_mask and cfg.get("vflip", False)) else 0.0,
                    agcwd_transform=agcwd_transform,
                )
                ds_val = HAMMultiInputDataset(
                    df=val_df,
                    meta_features=meta_features,
                    hog_lbp_features=None,
                    transform=eval_tf,
                    label_col="dx",
                    label2idx=label2idx,
                    use_masked_image=False,
                    use_hog_lbp=False,
                    mask_col_name="img_path_masked",
                    img_col_name="img_path",
                    fused_mask_loader=fused_loader_val,
                    fused_mask_mode=effective_mask_mode,
                    rotation_deg=0.0,
                    hflip_p=0.0,
                    vflip_p=0.0,
                    agcwd_transform=agcwd_transform,
                )

        print(f"[INFO] ds_train len={len(ds_train)}")
        print(f"[INFO] ds_val len={len(ds_val)}")

        sampler_mode = str(cfg.get("sampler", "none")).lower()
        sampler = None
        shuffle = True
        if sampler_mode == "weighted":
            sampler = make_weighted_sampler(train_df, label2idx, label_col="dx")
            shuffle = False

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            pin_memory=pin_memory,
            generator=g,
            drop_last=drop_last_train,
        )

        val_loader = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        use_meta = not bool(no_meta)

        in_channels = 4 if (input_source == "original" and effective_mask_mode == "concat") else 3

        model = EfficientNetWithMeta(
            num_classes=num_classes,
            arch=str(cfg.get("arch", "efficientnet_b0")),
            pretrained=bool(cfg["pretrained"]),
            meta_feature_dim=int(meta_feature_dim) if use_meta else 0,
            use_meta=use_meta,
            meta_hidden_dim=int(cfg.get("meta_hidden_dim", 128)),
            clf_hidden_dim=int(cfg.get("clf_hidden_dim", 256)),
            dropout=float(cfg["dropout"]),
            in_channels=in_channels,
            use_mask_attention=use_mask_attention,
            mask_attention_alpha=mask_attention_alpha,
            mask_attention_base_weight=mask_attention_base_weight,
            use_mask_context_branch=use_mask_context_branch,
            mask_context_branch_mode=mask_context_branch_mode,
        )

        model, best_metric, history = train_efficientnet_with_meta(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=int(cfg["num_epochs"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            patience=int(cfg["patience"]),
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            train_logits_tf=train_logits_tf,
            eval_logits_tf=eval_logits_tf,
            loss_fn=loss_fn,
            label2idx=label2idx,
            target_class=target_class,
            early_stopping_metric=early_stopping_metric,
            main_loss_type=main_loss_type,
            main_loss_weight=main_loss_weight,
            main_margin_target=main_margin_target,
            main_margin_temperature=main_margin_temperature,
            use_amp=True,
            seed=seed,
            use_meta=use_meta,
            use_mask=use_mask,
            mask_mode=effective_mask_mode,
            input_source=input_source,
            multi_view_loss_mode=multi_view_loss_mode,
            consistency_weight=consistency_weight,
            consistency_loss_type=consistency_loss_type,
            consistency_gate_mode=consistency_gate_mode,
            consistency_gate_temperature=consistency_gate_temperature,
            consistency_gate_margin_threshold=consistency_gate_margin_threshold,
            view_ce_weight=view_ce_weight,
            worst_view_ce_weight=worst_view_ce_weight,
            consensus_kl_weight=consensus_kl_weight,
            consensus_kl_temperature=consensus_kl_temperature,
            wrong_crop_count_weight=wrong_crop_count_weight,
            wrong_crop_count_temperature=wrong_crop_count_temperature,
            wrong_crop_count_reduction=wrong_crop_count_reduction,
            true_label_alignment_weight=true_label_alignment_weight,
            true_label_alignment_tolerance=true_label_alignment_tolerance,
            true_label_alignment_min_teacher_prob=true_label_alignment_min_teacher_prob,
            feature_consistency_weight=feature_consistency_weight,
            feature_consistency_loss_type=feature_consistency_loss_type,
            use_mask_attention=use_mask_attention,
            use_mask_context_branch=use_mask_context_branch,
            train_microbatch_size=cfg.get("train_microbatch_size"),
            freeze_backbone_epochs=freeze_backbone_epochs,
        )

        arch_tag = str(cfg.get("arch", "efficientnet_b0"))
        pretrained_tag = "pretrained" if bool(cfg["pretrained"]) else "scratch"
        meta_file_tag = "nometa" if no_meta else "meta"

        if input_source == "crop":
            if multi_view_crop:
                n_views = len(multi_view_crop_policy_choices or [])

                if multi_view_loss_mode in {
                    "avg_logits_consistency",
                    "avg_logits_plus_consistency",
                }:
                    input_tag = f"_crop_mv{n_views}_avglogits_cons{int(round(consistency_weight * 100)):03d}"

                elif multi_view_loss_mode in {
                    "avg_probs_consistency",
                    "avg_probs_plus_consistency",
                }:
                    input_tag = f"_crop_mv{n_views}_avgprobs_cons{int(round(consistency_weight * 100)):03d}"

                elif multi_view_loss_mode == "avg_probs":
                    input_tag = f"_crop_mv{n_views}_avgprobs"

                else:
                    input_tag = f"_crop_mv{n_views}_avglogits"

                if main_loss_type in {"avg_margin_hinge", "margin_hinge"}:
                    input_tag += f"_avgmargin{int(round(main_margin_target * 100)):03d}"
                elif main_loss_type in {"avg_margin_softplus", "margin_softplus"}:
                    input_tag += (
                        f"_softmargin{int(round(main_margin_target * 100)):03d}"
                        f"_mt{int(round(main_margin_temperature * 1000)):03d}"
                    )
                if abs(main_loss_weight - 1.0) > 1e-12:
                    input_tag += f"_mainw{int(round(main_loss_weight * 100)):03d}"

                if view_ce_weight > 0.0:
                    input_tag += f"_viewce{int(round(view_ce_weight * 100)):03d}"

                if worst_view_ce_weight > 0.0:
                    input_tag += f"_worstviewce{int(round(worst_view_ce_weight * 100)):03d}"

                if consensus_kl_weight > 0.0:
                    input_tag += f"_consensuskl{int(round(consensus_kl_weight * 100)):03d}"

                if wrong_crop_count_weight > 0.0:
                    input_tag += f"_wrongcrop{int(round(wrong_crop_count_weight * 100)):03d}"
                    if wrong_crop_count_reduction != "mean":
                        input_tag += f"_{wrong_crop_count_reduction}"

                if true_label_alignment_weight > 0.0:
                    input_tag += f"_truealign{int(round(true_label_alignment_weight * 100)):03d}"

                if feature_consistency_weight > 0.0:
                    input_tag += f"_featcons{int(round(feature_consistency_weight * 100)):03d}"

                if use_mask_attention:
                    input_tag += (
                        f"_maskatt_a{int(round(mask_attention_alpha * 100)):03d}"
                        f"_b{int(round(mask_attention_base_weight * 100)):03d}"
                    )

                if use_mask_context_branch:
                    input_tag += (
                        f"_maskctx_{mask_context_view_mode}"
                        f"_{mask_context_branch_mode}"
                    )

            elif random_crop_policy:
                input_tag = "_crop_random_policy"
                if use_mask_attention:
                    input_tag += (
                        f"_maskatt_a{int(round(mask_attention_alpha * 100)):03d}"
                        f"_b{int(round(mask_attention_base_weight * 100)):03d}"
                    )
                if use_mask_context_branch:
                    input_tag += (
                        f"_maskctx_{mask_context_view_mode}"
                        f"_{mask_context_branch_mode}"
                    )
            else:
                input_tag = (
                    f"_crop_thr{int(round(crop_threshold * 100)):03d}"
                    f"_margin{int(round(crop_margin_ratio * 100)):03d}"
                )
                if use_mask_attention:
                    input_tag += (
                        f"_maskatt_a{int(round(mask_attention_alpha * 100)):03d}"
                        f"_b{int(round(mask_attention_base_weight * 100)):03d}"
                    )
                if use_mask_context_branch:
                    input_tag += (
                        f"_maskctx_{mask_context_view_mode}"
                        f"_{mask_context_branch_mode}"
                    )
        elif use_fused and effective_mask_mode != "none":
            input_tag = f"_{effective_mask_mode}"
        else:
            input_tag = ""

        ckpt = safe_out_path(
            run_dir,
            f"{arch_tag}_{pretrained_tag}_{meta_file_tag}{input_tag}.pt",
            max_path_len=220,
        )

        def safe_torch_save(state_dict, path: Path, retries: int = 3, sleep_sec: float = 0.5):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Keep the temp filename very short. Some experiment directories are
            # already close to the Windows legacy 260-character limit, and
            # appending ".tmp" to a shortened checkpoint name can cross it.
            tmp = path.parent / f"tmp_{os.getpid()}.pt"

            last_err = None
            for _ in range(retries):
                try:
                    torch.save(state_dict, tmp, _use_new_zipfile_serialization=False)
                    os.replace(tmp, path)
                    return
                except Exception as e:
                    last_err = e
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
                    time.sleep(sleep_sec)

            raise last_err

        safe_torch_save(model.state_dict(), ckpt)

        plots = save_training_plots(history, run_dir, run_name="train")

        eval_val = evaluate_and_save_common(
            model=model,
            split_df=val_df,
            meta_features=meta_features,
            label2idx=label2idx,
            batch_size=eval_batch_size,
            val_transform=eval_tf,
            out_dir=run_dir,
            run_name="val",
            device=device,
            logits_tf=eval_logits_tf,
            no_meta=no_meta,
            target_class=target_class,
            fused_mask_loader=fused_loader_val,
            mask_mode=effective_mask_mode,
            agcwd_transform=agcwd_transform,
            input_source=input_source,
            img_size=img_size,
            crop_resize_mode=crop_resize_mode,
            adaptive_crop_shape=adaptive_crop_shape,
            crop_threshold=crop_threshold,
            crop_margin_ratio=crop_margin_ratio,
            crop_min_area=crop_min_area,
            crop_fallback_mode=crop_fallback_mode,
            background_mode=background_mode,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            softmultiply_alpha=softmultiply_alpha,
            mask_power=mask_power,
            missing_mask_mode=missing_mask_mode,
            debug_mask_pipeline=debug_mask_pipeline,
            eval_crop_mode=eval_crop_mode,
            eval_crop_policy_choices=eval_crop_policy_choices,
            use_mask_attention=use_mask_attention,
            use_mask_context_branch=use_mask_context_branch,
            mask_context_view_mode=mask_context_view_mode,
        )

        eval_test = evaluate_and_save_common(
            model=model,
            split_df=test_df,
            meta_features=meta_features,
            label2idx=label2idx,
            batch_size=eval_batch_size,
            val_transform=eval_tf,
            out_dir=run_dir,
            run_name="test",
            device=device,
            logits_tf=eval_logits_tf,
            no_meta=no_meta,
            target_class=target_class,
            fused_mask_loader=fused_loader_test,
            mask_mode=effective_mask_mode,
            agcwd_transform=agcwd_transform,
            input_source=input_source,
            img_size=img_size,
            crop_resize_mode=crop_resize_mode,
            adaptive_crop_shape=adaptive_crop_shape,
            crop_threshold=crop_threshold,
            crop_margin_ratio=crop_margin_ratio,
            crop_min_area=crop_min_area,
            crop_fallback_mode=crop_fallback_mode,
            background_mode=background_mode,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            softmultiply_alpha=softmultiply_alpha,
            mask_power=mask_power,
            missing_mask_mode=missing_mask_mode,
            debug_mask_pipeline=debug_mask_pipeline,
            eval_crop_mode=eval_crop_mode,
            eval_crop_policy_choices=eval_crop_policy_choices,
            use_mask_attention=use_mask_attention,
            use_mask_context_branch=use_mask_context_branch,
            mask_context_view_mode=mask_context_view_mode,
        )

        all_rows.append(
            {
                "config_name": run_name,
                **cfg,
                "seed": seed,
                "split_seed": split_seed,
                "no_meta": bool(no_meta),
                "target_class": target_class,
                "input_source": input_source,
                "use_crop_as_input": use_crop_as_input,
                "best_metric_value": float(best_metric),
                "early_stopping_metric": early_stopping_metric,
                "weights_path": str(ckpt),
                "random_crop_policy": random_crop_policy,
                "crop_policy_choices": str(crop_policy_choices),
                "multi_view_crop": multi_view_crop,
                "multi_view_crop_policy_choices": str(multi_view_crop_policy_choices),
                "main_loss_type": main_loss_type,
                "main_loss_weight": main_loss_weight,
                "main_margin_target": main_margin_target,
                "main_margin_temperature": main_margin_temperature,
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
                "freeze_backbone_epochs": freeze_backbone_epochs,
                "multi_view_loss_mode": multi_view_loss_mode,
                "consistency_weight": consistency_weight,
                "consistency_loss_type": consistency_loss_type,
                "consistency_gate_mode": consistency_gate_mode,
                "consistency_gate_temperature": consistency_gate_temperature,
                "consistency_gate_margin_threshold": consistency_gate_margin_threshold,
                "view_ce_weight": view_ce_weight,
                "worst_view_ce_weight": worst_view_ce_weight,
                "consensus_kl_weight": consensus_kl_weight,
                "consensus_kl_temperature": consensus_kl_temperature,
                "wrong_crop_count_weight": wrong_crop_count_weight,
                "wrong_crop_count_temperature": wrong_crop_count_temperature,
                "wrong_crop_count_reduction": wrong_crop_count_reduction,
                "true_label_alignment_weight": true_label_alignment_weight,
                "true_label_alignment_tolerance": true_label_alignment_tolerance,
                "true_label_alignment_min_teacher_prob": true_label_alignment_min_teacher_prob,
                "feature_consistency_weight": feature_consistency_weight,
                "feature_consistency_loss_type": feature_consistency_loss_type,
                "use_mask_attention": use_mask_attention,
                "mask_attention_alpha": mask_attention_alpha,
                "mask_attention_base_weight": mask_attention_base_weight,
                "use_mask_context_branch": use_mask_context_branch,
                "mask_context_view_mode": mask_context_view_mode,
                "mask_context_branch_mode": mask_context_branch_mode,
                "eval_crop_mode": eval_crop_mode,
                "eval_crop_policy_choices": str(eval_crop_policy_choices),
                **plots,
                **eval_val,
                **eval_test,
            }
        )

    summary_df = pd.DataFrame(all_rows)

    # Ensure the output directory exists before writing the final summary.
    # This prevents FileNotFoundError if the run folder was not created
    # or was removed before summary export.
    summary_path = safe_out_path(
        out_root,
        f"efficientnet_val_summary_nometa{int(no_meta)}.csv",
        max_path_len=250,
    )

    summary_df.to_csv(summary_path, index=False)
    return summary_path
