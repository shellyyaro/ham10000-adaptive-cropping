from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from resnet_vit_pretrained_comparison.data.adaptive_crop_policy import (
    adaptive_crop_for_target_ratio,
)
from resnet_vit_pretrained_comparison.data.mask_fusion_loader import FusedMaskLoader


def _sync_geometric_to_mask(
    mask_np: np.ndarray | None,
    target_size: tuple[int, int],
    hflip: bool,
    vflip: bool,
    angle: float,
) -> torch.Tensor:
    """
    Assumes mask_np already comes normalized from FusedMaskLoader.get_mask():
    - 2D
    - float32
    - clipped to [0, 1]

    Important:
    Resize the mask to the current image size BEFORE applying geometric
    augmentation, so the mask and image are transformed in the same spatial frame.
    """
    h_out, w_out = target_size

    if mask_np is None:
        return torch.zeros(1, h_out, w_out, dtype=torch.float32)

    mask_np = np.asarray(mask_np, dtype=np.float32)

    if mask_np.ndim != 2:
        raise ValueError(
            f"Expected normalized 2D mask from loader, got shape={mask_np.shape}"
        )

    mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()  # (1, H, W)

    if mask_t.shape[-2] != h_out or mask_t.shape[-1] != w_out:
        mask_t = F.interpolate(
            mask_t.unsqueeze(0),
            size=(h_out, w_out),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if hflip:
        mask_t = torch.flip(mask_t, dims=[-1])

    if vflip:
        mask_t = torch.flip(mask_t, dims=[-2])

    if angle != 0.0:
        mask_t = TF.rotate(
            mask_t,
            angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0.0,
        )

    return mask_t.clamp(0.0, 1.0).to(torch.float32)


def _ensure_odd_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size < 1:
        kernel_size = 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _safe_float_or_none(x) -> float | None:
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    try:
        return float(x)
    except Exception:
        return None


def _get_mask_bbox_from_tensor(
    mask_1ch: torch.Tensor,
    threshold: float = 0.2,
    min_area: int = 10,
) -> tuple[int, int, int, int] | None:
    """
    Get bbox around the positive mask region.

    Args:
        mask_1ch:
            Tensor of shape (1, H, W), values in [0, 1].
        threshold:
            Pixels >= threshold are considered part of the lesion.
        min_area:
            Minimum number of positive pixels required.

    Returns:
        (x1, y1, x2, y2), where x2/y2 are exclusive.
        Returns None if the mask is empty / too small.
    """
    if mask_1ch.ndim != 3 or mask_1ch.shape[0] != 1:
        raise ValueError(f"Expected mask shape (1,H,W), got {tuple(mask_1ch.shape)}")

    mask_bin = mask_1ch[0] >= float(threshold)
    ys, xs = torch.where(mask_bin)

    if xs.numel() < int(min_area) or ys.numel() < int(min_area):
        return None

    x1 = int(xs.min().item())
    x2 = int(xs.max().item()) + 1
    y1 = int(ys.min().item())
    y2 = int(ys.max().item()) + 1

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    img_h: int,
    img_w: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    """
    Expand bbox by margin_ratio of bbox size.

    bbox format:
        (x1, y1, x2, y2), x2/y2 exclusive.
    """
    x1, y1, x2, y2 = bbox

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    margin_x = int(round(box_w * float(margin_ratio)))
    margin_y = int(round(box_h * float(margin_ratio)))

    x1_new = max(0, x1 - margin_x)
    y1_new = max(0, y1 - margin_y)
    x2_new = min(img_w, x2 + margin_x)
    y2_new = min(img_h, y2 + margin_y)

    if x2_new <= x1_new:
        x2_new = min(img_w, x1_new + 1)
    if y2_new <= y1_new:
        y2_new = min(img_h, y1_new + 1)

    return x1_new, y1_new, x2_new, y2_new

def _adaptive_bbox_from_raw_bbox(
    raw_bbox: tuple[int, int, int, int],
    *,
    img_h: int,
    img_w: int,
    target_bbox_ratio: float,
    adaptive_scale_factor: float = 1.0,
    adaptive_crop_shape: str = "square",
    min_crop_frac: float = 0.20,
    max_crop_frac: float = 1.00,
) -> tuple[int, int, int, int]:
    """
    Build an adaptive crop box so that:

        lesion_bbox_area / crop_area ≈ target_bbox_ratio

    adaptive_crop_shape:
        "square":
            Build a square crop around the lesion bbox.
            This is the original adaptive crop behavior.

        "preserve_aspect":
            Build a rectangular crop that preserves the bbox aspect ratio.
            This avoids forcing elongated lesions into square crops before resize.

    Smaller target_bbox_ratio => wider/context crop.
    Larger target_bbox_ratio  => tighter/focused crop.
    """
    x1, y1, x2, y2 = raw_bbox

    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area = bbox_w * bbox_h

    target_bbox_ratio = float(target_bbox_ratio)
    adaptive_scale_factor = float(adaptive_scale_factor)
    adaptive_crop_shape = str(adaptive_crop_shape or "square").strip().lower()

    if adaptive_crop_shape not in {"square", "preserve_aspect"}:
        raise ValueError(
            "adaptive_crop_shape must be 'square' or 'preserve_aspect', "
            f"got {adaptive_crop_shape!r}"
        )

    # Desired crop area according to the target lesion-to-crop ratio.
    crop_area = bbox_area / target_bbox_ratio

    if adaptive_crop_shape == "square":
        crop_w = crop_h = int(round(np.sqrt(crop_area)))
        crop_w = max(crop_w, bbox_w, bbox_h)
        crop_h = crop_w
    else:
        # Preserve bbox aspect ratio.
        # Since crop_area = bbox_area * scale^2:
        # scale = sqrt(crop_area / bbox_area) = sqrt(1 / target_bbox_ratio)
        scale = float(np.sqrt(crop_area / max(1.0, float(bbox_area))))
        crop_w = int(round(bbox_w * scale))
        crop_h = int(round(bbox_h * scale))
        crop_w = max(crop_w, bbox_w)
        crop_h = max(crop_h, bbox_h)

    crop_w = int(round(crop_w * adaptive_scale_factor))
    crop_h = int(round(crop_h * adaptive_scale_factor))

    # Clamp crop dimensions. For preserve_aspect, each side is clamped
    # independently, so extreme image boundaries may slightly alter the ratio.
    min_side = int(round(min(img_h, img_w) * float(min_crop_frac)))
    max_w = int(round(img_w * float(max_crop_frac)))
    max_h = int(round(img_h * float(max_crop_frac)))

    crop_w = int(np.clip(crop_w, max(1, min_side), max(1, max_w)))
    crop_h = int(np.clip(crop_h, max(1, min_side), max(1, max_h)))

    # Ensure the bbox can fit inside the crop.
    crop_w = max(crop_w, bbox_w)
    crop_h = max(crop_h, bbox_h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    left = int(round(cx - crop_w / 2))
    top = int(round(cy - crop_h / 2))
    right = left + crop_w
    bottom = top + crop_h

    if left < 0:
        right -= left
        left = 0

    if top < 0:
        bottom -= top
        top = 0

    if right > img_w:
        left -= right - img_w
        right = img_w

    if bottom > img_h:
        top -= bottom - img_h
        bottom = img_h

    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    if right <= left:
        right = min(img_w, left + 1)
    if bottom <= top:
        bottom = min(img_h, top + 1)

    return left, top, right, bottom

def _crop_tensor_by_bbox(
    x: torch.Tensor,
    bbox: tuple[int, int, int, int],
) -> torch.Tensor:
    """
    Crop tensor using bbox.

    Args:
        x:
            Tensor of shape (C, H, W).
        bbox:
            (x1, y1, x2, y2), x2/y2 exclusive.

    Returns:
        Tensor of shape (C, crop_h, crop_w).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected tensor shape (C,H,W), got {tuple(x.shape)}")

    x1, y1, x2, y2 = bbox
    return x[:, y1:y2, x1:x2]




def _resize_letterbox_tensor(
    x: torch.Tensor,
    target_size: int,
    *,
    fill_value: float = 0.0,
    interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize tensor while preserving aspect ratio, then pad to a square.

    Args:
        x:
            Tensor of shape (C,H,W), values usually in [0,1].
        target_size:
            Final square size, e.g. 224.
        fill_value:
            Padding value before normalization.

    Returns:
        Tensor of shape (C,target_size,target_size).

    This avoids geometric distortion caused by direct resize to (224,224).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected tensor shape (C,H,W), got {tuple(x.shape)}")

    target_size = int(target_size)
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    _, h, w = x.shape
    h = int(h)
    w = int(w)

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid spatial shape for letterbox resize: H={h}, W={w}")

    scale = float(target_size) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = TF.resize(
        x,
        [new_h, new_w],
        interpolation=interpolation,
        antialias=True,
    )

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    # TF.pad uses padding order: [left, top, right, bottom]
    out = TF.pad(
        resized,
        padding=[pad_left, pad_top, pad_right, pad_bottom],
        fill=float(fill_value),
    )

    if out.shape[-2] != target_size or out.shape[-1] != target_size:
        # Safety fallback for rare rounding/padding edge cases.
        out = TF.resize(
            out,
            [target_size, target_size],
            interpolation=interpolation,
            antialias=True,
        )

    return out

def _normalize_crop_policy_choices(
    choices: Optional[list[dict]],
    *,
    context_name: str,
) -> list[dict]:
    if not choices:
        raise ValueError(f"{context_name} requires non-empty crop policy choices")

    normalized_choices = []

    for i, choice in enumerate(choices):
        if not isinstance(choice, dict):
            raise ValueError(f"{context_name}[{i}] must be dict, got {type(choice)}")

        crop_mode = str(choice.get("crop_mode", "fixed_margin")).strip().lower()

        if crop_mode not in {"fixed_margin", "adaptive", "adaptive_mask_area"}:
            raise ValueError(
                f"{context_name}[{i}].crop_mode must be 'fixed_margin', "
                f"'adaptive', or 'adaptive_mask_area', "
                f"got {crop_mode}"
            )

        if "crop_threshold" not in choice:
            raise ValueError(f"{context_name}[{i}] is missing key 'crop_threshold'")

        thr = float(choice["crop_threshold"])

        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"{context_name}[{i}].crop_threshold must be in [0,1]")

        out = {
            "crop_mode": crop_mode,
            "crop_threshold": thr,
            "policy_index": int(choice.get("policy_index", i)),
        }

        if crop_mode == "fixed_margin":
            if "crop_margin_ratio" not in choice:
                raise ValueError(
                    f"{context_name}[{i}] fixed_margin policy missing crop_margin_ratio"
                )

            margin = float(choice["crop_margin_ratio"])
            if margin < 0.0:
                raise ValueError(f"{context_name}[{i}].crop_margin_ratio must be >= 0")

            out["crop_margin_ratio"] = margin
            out["target_bbox_ratio"] = -1.0
            out["target_mask_area_ratio"] = -1.0
            out["adaptive_scale_factor"] = 1.0
            out["adaptive_crop_shape"] = str(
                choice.get("adaptive_crop_shape", "square")
            ).strip().lower()

        elif crop_mode == "adaptive":

            if "target_bbox_ratio" not in choice:
                raise ValueError(

                    f"{context_name}[{i}] adaptive policy missing target_bbox_ratio"

                )

            target = float(choice["target_bbox_ratio"])

            if not (0.01 <= target <= 1.0):
                raise ValueError(

                    f"{context_name}[{i}].target_bbox_ratio must be in [0.01,1.0]"

                )

            # adaptive mode does not use margin

            out["crop_margin_ratio"] = 0.0

            out["target_bbox_ratio"] = target
            out["target_mask_area_ratio"] = -1.0

            out["adaptive_scale_factor"] = float(

                choice.get("adaptive_scale_factor", 1.0)

            )

            adaptive_crop_shape = str(
                choice.get("adaptive_crop_shape", "square")
            ).strip().lower()

            if adaptive_crop_shape not in {"square", "preserve_aspect"}:
                raise ValueError(
                    f"{context_name}[{i}].adaptive_crop_shape must be "
                    f"'square' or 'preserve_aspect', got {adaptive_crop_shape!r}"
                )

            out["adaptive_crop_shape"] = adaptive_crop_shape
        else:
            target_key = (
                "target_mask_area_ratio"
                if "target_mask_area_ratio" in choice
                else "target_bbox_ratio"
            )
            if target_key not in choice:
                raise ValueError(
                    f"{context_name}[{i}] adaptive_mask_area policy missing "
                    f"target_mask_area_ratio"
                )

            target = float(choice[target_key])
            if not (0.01 <= target <= 1.0):
                raise ValueError(
                    f"{context_name}[{i}].{target_key} must be in [0.01,1.0], "
                    f"got {target}"
                )

            out["crop_margin_ratio"] = 0.0
            out["target_bbox_ratio"] = -1.0
            out["target_mask_area_ratio"] = target
            out["adaptive_scale_factor"] = 1.0
            out["adaptive_crop_shape"] = "closest_rectangle"

        normalized_choices.append(out)

    return normalized_choices


def _default_crop_info(
    *,
    crop_threshold: float,
    crop_margin_ratio: float,
    crop_policy_index: int = -1,
    crop_mode: str = "fixed_margin",
    target_bbox_ratio: float = -1.0,
    target_mask_area_ratio: float = -1.0,
    adaptive_scale_factor: float = 1.0,
    adaptive_crop_shape: str = "square",
) -> dict:
    return {
        "crop_fallback_used": False,
        "crop_bbox_found": False,
        "crop_bbox_x1": -1,
        "crop_bbox_y1": -1,
        "crop_bbox_x2": -1,
        "crop_bbox_y2": -1,
        "crop_area_ratio": 1.0,
        "effective_crop_threshold": float(crop_threshold),
        "effective_crop_margin_ratio": float(crop_margin_ratio),
        "crop_policy_index": int(crop_policy_index),

        # new
        "crop_mode": str(crop_mode),
        "target_bbox_ratio": float(target_bbox_ratio),
        "target_mask_area_ratio": float(target_mask_area_ratio),
        "adaptive_scale_factor": float(adaptive_scale_factor),
        "adaptive_crop_shape": str(adaptive_crop_shape),
        "achieved_bbox_ratio_in_crop": -1.0,
        "achieved_mask_area_ratio_in_crop": -1.0,
        "adaptive_mask_area_target_possible": False,
        "adaptive_mask_area_mode": "",
        "adaptive_mask_area_distortion_log_abs": -1.0,
        "raw_bbox_area_ratio": -1.0,
    }


class HAMMaskedBackgroundDataset(Dataset):
    """
    Dataset that returns BOTH:
    1) original RGB image tensor
    2) second-branch RGB image tensor

    The second branch can be:
    - masked-background image:
        background_mode='blur' / 'multiply' / 'softmultiply' / 'mask_weighted_blur'
    - single crop around lesion mask:
        use_crop=True, multi_view_crop=False
    - multiple crops around lesion mask:
        use_crop=True, multi_view_crop=True

    In crop-only EfficientNet experiments, the model receives second_branch_image_tensor
    as the main image input. In that case, no mask tensor is passed to the model;
    the mask is used only to define the crop.

    Important modes:
    - random_crop_policy=True:
        Each __getitem__ chooses ONE crop policy randomly.
        second_branch_image_tensor shape: (C, H, W)

    - multi_view_crop=True:
        Each __getitem__ returns ALL crop policies together.
        second_branch_image_tensor shape: (V, C, H, W)

    These modes are intentionally mutually exclusive.

    Returns:
        original_image_tensor, second_branch_image_tensor, meta_vec, label_idx, extras
    """

    VALID_BG_MODES = {"none", "multiply", "blur", "softmultiply", "mask_weighted_blur"}
    VALID_MISSING_MASK_MODES = {"keep_original", "zeros_mask", "ones_mask"}
    VALID_CROP_FALLBACK_MODES = {"keep_original", "background_mode"}

    def __init__(
        self,
        df: pd.DataFrame,
        meta_features: np.ndarray,
        label_col: str,
        label2idx: dict,
        img_col_name: str = "img_path",
        fused_mask_loader: Optional[FusedMaskLoader] = None,
        rotation_deg: float = 0.0,
        hflip_p: float = 0.5,
        vflip_p: float = 0.0,
        agcwd_transform=None,
        sanity_check_all_ones_masks: bool = False,
        background_mode: str = "blur",
        blur_kernel_size: int = 11,
        blur_sigma: float | tuple[float, float] = 4.0,
        softmultiply_alpha: float = 0.5,
        mask_power: float = 1.0,
        missing_mask_mode: str = "keep_original",
        debug: bool = False,
        debug_max_samples: int = 10,
        validate_loader_alignment: bool = True,
        img_size: int = 224,
        crop_resize_mode: str = "stretch",
        crop_letterbox_fill: float = 0.0,
        adaptive_crop_shape: str = "square",
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_color_jitter: bool = False,
        color_jitter_brightness: float = 0.0,
        color_jitter_contrast: float = 0.0,
        color_jitter_saturation: float = 0.0,
        color_jitter_hue: float = 0.0,
        per_view_color_jitter_brightness: bool = False,
        per_view_color_jitter_contrast: bool = False,
        per_view_color_jitter_saturation: bool = False,
        per_view_color_jitter_hue: bool = False,
        use_crop: bool = False,
        crop_threshold: float = 0.2,
        crop_margin_ratio: float = 0.30,
        crop_min_area: int = 10,
        crop_fallback_mode: str = "keep_original",
        random_crop_policy: bool = False,
        crop_policy_choices: Optional[list[dict]] = None,
        multi_view_crop: bool = False,
        multi_view_crop_policy_choices: Optional[list[dict]] = None,
        return_mask_context_image: bool = False,
        mask_context_view_mode: str = "full",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.meta_features = meta_features
        self.label_col = label_col
        self.label2idx = label2idx
        self.img_col_name = img_col_name
        self.fused_mask_loader = fused_mask_loader

        self.rotation_deg = float(rotation_deg)
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)

        if not (0.0 <= self.hflip_p <= 1.0):
            raise ValueError(f"hflip_p must be in [0, 1], got {self.hflip_p}")
        if not (0.0 <= self.vflip_p <= 1.0):
            raise ValueError(f"vflip_p must be in [0, 1], got {self.vflip_p}")

        self.agcwd_transform = agcwd_transform
        self.sanity_check_all_ones_masks = bool(sanity_check_all_ones_masks)

        self.background_mode = str(background_mode).strip().lower()
        if self.background_mode not in self.VALID_BG_MODES:
            raise ValueError(
                f"Unknown background_mode={background_mode!r}. "
                f"Options: {sorted(self.VALID_BG_MODES)}"
            )

        self.blur_kernel_size = _ensure_odd_kernel_size(blur_kernel_size)
        self.blur_sigma = blur_sigma

        self.softmultiply_alpha = float(softmultiply_alpha)
        if not (0.0 <= self.softmultiply_alpha <= 1.0):
            raise ValueError("softmultiply_alpha must be in [0, 1]")

        self.mask_power = float(mask_power)
        if self.mask_power <= 0:
            raise ValueError("mask_power must be > 0")

        self.missing_mask_mode = str(missing_mask_mode).strip().lower()
        if self.missing_mask_mode not in self.VALID_MISSING_MASK_MODES:
            raise ValueError(
                f"Unknown missing_mask_mode={missing_mask_mode!r}. "
                f"Options: {sorted(self.VALID_MISSING_MASK_MODES)}"
            )

        self.use_crop = bool(use_crop)
        self.crop_threshold = float(crop_threshold)
        self.crop_margin_ratio = float(crop_margin_ratio)
        self.crop_min_area = int(crop_min_area)

        if not (0.0 <= self.crop_threshold <= 1.0):
            raise ValueError(
                f"crop_threshold must be in [0, 1], got {self.crop_threshold}"
            )

        if self.crop_margin_ratio < 0.0:
            raise ValueError(
                f"crop_margin_ratio must be >= 0, got {self.crop_margin_ratio}"
            )

        if self.crop_min_area < 1:
            raise ValueError(f"crop_min_area must be >= 1, got {self.crop_min_area}")

        self.crop_fallback_mode = str(crop_fallback_mode).strip().lower()
        if self.crop_fallback_mode not in self.VALID_CROP_FALLBACK_MODES:
            raise ValueError(
                f"Unknown crop_fallback_mode={crop_fallback_mode!r}. "
                f"Options: {sorted(self.VALID_CROP_FALLBACK_MODES)}"
            )

        # ------------------------------------------------------------
        # Crop policy modes
        # ------------------------------------------------------------
        self.random_crop_policy = bool(random_crop_policy)
        self.multi_view_crop = bool(multi_view_crop)
        self.return_mask_context_image = bool(return_mask_context_image)
        self.mask_context_view_mode = str(mask_context_view_mode or "full").strip().lower()
        if self.mask_context_view_mode not in {"full", "crop_aligned"}:
            raise ValueError(
                "mask_context_view_mode must be 'full' or 'crop_aligned', "
                f"got {self.mask_context_view_mode!r}"
            )

        if self.random_crop_policy and self.multi_view_crop:
            raise ValueError(
                "random_crop_policy=True and multi_view_crop=True are mutually exclusive. "
                "Use random_crop_policy for one random crop per sample, or "
                "multi_view_crop for all crop views per sample."
            )

        self.crop_policy_choices = None
        if self.random_crop_policy:
            if not self.use_crop:
                raise ValueError("random_crop_policy=True requires use_crop=True")

            self.crop_policy_choices = _normalize_crop_policy_choices(
                crop_policy_choices,
                context_name="crop_policy_choices",
            )

        self.multi_view_crop_policy_choices = None
        if self.multi_view_crop:
            if not self.use_crop:
                raise ValueError("multi_view_crop=True requires use_crop=True")

            self.multi_view_crop_policy_choices = _normalize_crop_policy_choices(
                multi_view_crop_policy_choices,
                context_name="multi_view_crop_policy_choices",
            )

        self.debug = bool(debug)
        self.debug_max_samples = int(debug_max_samples)
        self.validate_loader_alignment = bool(validate_loader_alignment)
        self._debug_seen_count = 0

        self.img_size = int(img_size)

        self.crop_resize_mode = str(crop_resize_mode or "stretch").strip().lower()
        if self.crop_resize_mode not in {"stretch", "letterbox"}:
            raise ValueError(
                f"crop_resize_mode must be 'stretch' or 'letterbox', "
                f"got {self.crop_resize_mode!r}"
            )

        self.crop_letterbox_fill = float(crop_letterbox_fill)

        self.adaptive_crop_shape = str(adaptive_crop_shape or "square").strip().lower()
        if self.adaptive_crop_shape not in {"square", "preserve_aspect", "closest_rectangle"}:
            raise ValueError(
                f"adaptive_crop_shape must be 'square', 'preserve_aspect', "
                f"or 'closest_rectangle', "
                f"got {self.adaptive_crop_shape!r}"
            )

        self.normalize_mean = tuple(float(x) for x in normalize_mean)
        self.normalize_std = tuple(float(x) for x in normalize_std)

        self.use_color_jitter = bool(use_color_jitter)
        self.color_jitter_brightness = float(color_jitter_brightness)
        self.color_jitter_contrast = float(color_jitter_contrast)
        self.color_jitter_saturation = float(color_jitter_saturation)
        self.color_jitter_hue = float(color_jitter_hue)

        # ------------------------------------------------------------
        # Per-view color jitter
        # ------------------------------------------------------------
        # Default behavior is fully backward-compatible:
        #   all flags False -> one ColorJitter sample is shared by the
        #   original image and all crop views.
        #
        # New behavior for MVCROP:
        #   each flag controls whether that specific factor is re-sampled
        #   independently for every crop/view.
        #
        # Example:
        #   per_view_color_jitter_contrast=True
        #   -> brightness / saturation / hue / operation order stay shared,
        #      but contrast_factor is re-sampled separately for each crop view.
        self.per_view_color_jitter_brightness = bool(per_view_color_jitter_brightness)
        self.per_view_color_jitter_contrast = bool(per_view_color_jitter_contrast)
        self.per_view_color_jitter_saturation = bool(per_view_color_jitter_saturation)
        self.per_view_color_jitter_hue = bool(per_view_color_jitter_hue)

        self.per_view_color_jitter_any = any(
            [
                self.per_view_color_jitter_brightness,
                self.per_view_color_jitter_contrast,
                self.per_view_color_jitter_saturation,
                self.per_view_color_jitter_hue,
            ]
        )

        if self.img_col_name not in self.df.columns:
            raise ValueError(f"Missing column: {self.img_col_name}")
        if self.label_col not in self.df.columns:
            raise ValueError(f"Missing column: {self.label_col}")
        if "row_idx" not in self.df.columns:
            raise ValueError("Missing column: row_idx")
        if self.df["row_idx"].isna().any():
            raise ValueError("row_idx contains NaN values")

        try:
            self.df["row_idx"] = self.df["row_idx"].astype(int)
        except Exception as e:
            raise ValueError("row_idx column must be integer-like") from e

        if not isinstance(self.meta_features, np.ndarray):
            self.meta_features = np.asarray(self.meta_features)

        if self.meta_features.size == 0:
            raise ValueError("meta_features is empty")

        if self.meta_features.ndim != 2:
            raise ValueError(
                f"meta_features must be 2D with shape (N, F). "
                f"Got shape={getattr(self.meta_features, 'shape', None)}"
            )

        row_idx_min = int(self.df["row_idx"].min())
        row_idx_max = int(self.df["row_idx"].max())

        if row_idx_min < 0:
            raise ValueError(f"row_idx contains negative values: min={row_idx_min}")

        if row_idx_max >= len(self.meta_features):
            raise ValueError(
                f"row_idx max={row_idx_max} is out of bounds for meta_features "
                f"with len={len(self.meta_features)}"
            )

        unknown_labels = sorted(
            set(self.df[self.label_col].unique()) - set(self.label2idx.keys())
        )
        if unknown_labels:
            raise ValueError(
                f"Found labels not present in label2idx: {unknown_labels}"
            )

        if self.debug:
            print("\n" + "=" * 80)
            print("[DEBUG][Dataset::__init__] HAMMaskedBackgroundDataset initialized")
            print(f"[DEBUG][Dataset::__init__] n_samples={len(self.df)}")
            print(f"[DEBUG][Dataset::__init__] background_mode={self.background_mode}")
            print(f"[DEBUG][Dataset::__init__] blur_kernel_size={self.blur_kernel_size}")
            print(f"[DEBUG][Dataset::__init__] blur_sigma={self.blur_sigma}")
            print(f"[DEBUG][Dataset::__init__] softmultiply_alpha={self.softmultiply_alpha}")
            print(f"[DEBUG][Dataset::__init__] mask_power={self.mask_power}")
            print(f"[DEBUG][Dataset::__init__] missing_mask_mode={self.missing_mask_mode}")
            print(f"[DEBUG][Dataset::__init__] use_crop={self.use_crop}")
            print(f"[DEBUG][Dataset::__init__] crop_threshold={self.crop_threshold}")
            print(f"[DEBUG][Dataset::__init__] crop_margin_ratio={self.crop_margin_ratio}")
            print(f"[DEBUG][Dataset::__init__] crop_min_area={self.crop_min_area}")
            print(f"[DEBUG][Dataset::__init__] crop_fallback_mode={self.crop_fallback_mode}")
            print(f"[DEBUG][Dataset::__init__] random_crop_policy={self.random_crop_policy}")
            print(f"[DEBUG][Dataset::__init__] crop_policy_choices={self.crop_policy_choices}")
            print(f"[DEBUG][Dataset::__init__] multi_view_crop={self.multi_view_crop}")
            print(
                "[DEBUG][Dataset::__init__] "
                f"multi_view_crop_policy_choices={self.multi_view_crop_policy_choices}"
            )
            print(
                "[DEBUG][Dataset::__init__] "
                f"return_mask_context_image={self.return_mask_context_image}"
            )
            print(
                "[DEBUG][Dataset::__init__] "
                f"mask_context_view_mode={self.mask_context_view_mode}"
            )
            print(f"[DEBUG][Dataset::__init__] img_size={self.img_size}")
            print(f"[DEBUG][Dataset::__init__] crop_resize_mode={self.crop_resize_mode}")
            print(f"[DEBUG][Dataset::__init__] crop_letterbox_fill={self.crop_letterbox_fill}")
            print(f"[DEBUG][Dataset::__init__] adaptive_crop_shape={self.adaptive_crop_shape}")
            print(f"[DEBUG][Dataset::__init__] normalize_mean={self.normalize_mean}")
            print(f"[DEBUG][Dataset::__init__] normalize_std={self.normalize_std}")
            print(f"[DEBUG][Dataset::__init__] use_color_jitter={self.use_color_jitter}")
            print(
                f"[DEBUG][Dataset::__init__] color_jitter="
                f"(b={self.color_jitter_brightness}, "
                f"c={self.color_jitter_contrast}, "
                f"s={self.color_jitter_saturation}, "
                f"h={self.color_jitter_hue})"
            )
            print(
                f"[DEBUG][Dataset::__init__] "
                f"per_view_color_jitter_brightness="
                f"{self.per_view_color_jitter_brightness}, "
                f"per_view_color_jitter_contrast="
                f"{self.per_view_color_jitter_contrast}, "
                f"per_view_color_jitter_saturation="
                f"{self.per_view_color_jitter_saturation}, "
                f"per_view_color_jitter_hue="
                f"{self.per_view_color_jitter_hue}"
            )
            print(
                f"[DEBUG][Dataset::__init__] "
                f"sanity_check_all_ones_masks={self.sanity_check_all_ones_masks}"
            )
            print(
                f"[DEBUG][Dataset::__init__] "
                f"fused_mask_loader_is_none={self.fused_mask_loader is None}"
            )
            print(
                "[DEBUG][Dataset::__init__] has_majority_flag_in_df="
                f"{'has_majority_flag' in self.df.columns}"
            )
            print(
                "[DEBUG][Dataset::__init__] mean_teacher_confidence_in_df="
                f"{'mean_teacher_confidence' in self.df.columns}"
            )
            print(
                f"[DEBUG][Dataset::__init__] meta_features.shape="
                f"{getattr(self.meta_features, 'shape', None)}"
            )
            print(
                f"[DEBUG][Dataset::__init__] row_idx range="
                f"[{row_idx_min}, {row_idx_max}]"
            )
            if self.df["row_idx"].duplicated().any():
                print("[DEBUG][Dataset::__init__] WARNING: row_idx contains duplicates")
            if self.fused_mask_loader is not None:
                print(f"[DEBUG][Dataset::__init__] fused_mask_loader={self.fused_mask_loader}")
            print("=" * 80 + "\n")

    def __len__(self) -> int:
        return len(self.df)

    def _should_debug_sample(self) -> bool:
        return self.debug and (self._debug_seen_count < self.debug_max_samples)

    def _build_fallback_mask(self, h: int, w: int) -> torch.Tensor:
        if self.missing_mask_mode == "keep_original":
            return torch.ones(1, h, w, dtype=torch.float32)
        if self.missing_mask_mode == "zeros_mask":
            return torch.zeros(1, h, w, dtype=torch.float32)
        if self.missing_mask_mode == "ones_mask":
            return torch.ones(1, h, w, dtype=torch.float32)
        raise RuntimeError(f"Unsupported missing_mask_mode={self.missing_mask_mode}")

    def _apply_background_mode(
        self,
        x: torch.Tensor,
        mask_1ch: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"Expected x shape (3,H,W), got {tuple(x.shape)}")

        if mask_1ch.ndim != 3 or mask_1ch.shape[0] != 1:
            raise ValueError(f"Expected mask shape (1,H,W), got {tuple(mask_1ch.shape)}")

        mask_3ch = mask_1ch.expand_as(x)

        if self.background_mode == "none":
            return x

        if self.background_mode == "multiply":
            return x * mask_3ch

        if self.background_mode == "blur":
            if self.blur_kernel_size > 1:
                blurred_x = TF.gaussian_blur(
                    x,
                    kernel_size=[self.blur_kernel_size, self.blur_kernel_size],
                    sigma=self.blur_sigma,
                )
            else:
                blurred_x = x.clone()

            return mask_3ch * x + (1.0 - mask_3ch) * blurred_x

        if self.background_mode == "mask_weighted_blur":
            if self.blur_kernel_size > 1:
                blurred_x = TF.gaussian_blur(
                    x,
                    kernel_size=[self.blur_kernel_size, self.blur_kernel_size],
                    sigma=self.blur_sigma,
                )
            else:
                blurred_x = x.clone()

            # The mask values now have two jobs:
            #   1. preserve sharp pixels near high-confidence lesion regions
            #   2. softly suppress low-confidence/background regions
            # A nonzero floor keeps surrounding context visible.
            base = mask_3ch * x + (1.0 - mask_3ch) * blurred_x
            min_weight = float(self.softmultiply_alpha)
            min_weight = max(0.0, min(1.0, min_weight))
            emphasis = min_weight + (1.0 - min_weight) * mask_3ch
            return base * emphasis

        if self.background_mode == "softmultiply":
            alpha = self.softmultiply_alpha
            bg_weight = alpha + (1.0 - alpha) * mask_3ch
            return x * bg_weight

        raise RuntimeError(f"Unsupported background_mode={self.background_mode}")

    def _sample_crop_policy(self) -> dict:
        if self.random_crop_policy and self.crop_policy_choices:
            return random.choice(self.crop_policy_choices)

        return {
            "crop_mode": "fixed_margin",
            "crop_threshold": float(self.crop_threshold),
            "crop_margin_ratio": float(self.crop_margin_ratio),
            "target_bbox_ratio": -1.0,
            "target_mask_area_ratio": -1.0,
            "adaptive_scale_factor": 1.0,
            "adaptive_crop_shape": "square",
            "policy_index": -1,
        }

    def _build_crop_branch_image(
            self,
            base_x: torch.Tensor,
            fused_mask_tensor: torch.Tensor,
            *,
            crop_threshold: float | None = None,
            crop_margin_ratio: float | None = None,
            crop_policy_index: int = -1,
            crop_mode: str = "fixed_margin",
            target_bbox_ratio: float = -1.0,
            target_mask_area_ratio: float = -1.0,
            adaptive_scale_factor: float = 1.0,
            adaptive_crop_shape: str = "square",
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:

        if base_x.ndim != 3 or base_x.shape[0] != 3:
            raise ValueError(f"Expected base_x shape (3,H,W), got {tuple(base_x.shape)}")

        if fused_mask_tensor.ndim != 3 or fused_mask_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected fused_mask_tensor shape (1,H,W), got {tuple(fused_mask_tensor.shape)}"
            )

        if crop_threshold is None:
            crop_threshold = self.crop_threshold
        if crop_margin_ratio is None:
            crop_margin_ratio = self.crop_margin_ratio

        crop_mode = str(crop_mode or "fixed_margin").strip().lower()
        crop_threshold = float(crop_threshold)
        crop_margin_ratio = float(crop_margin_ratio)
        crop_policy_index = int(crop_policy_index)
        target_bbox_ratio = float(target_bbox_ratio)
        target_mask_area_ratio = float(target_mask_area_ratio)
        adaptive_scale_factor = float(adaptive_scale_factor)
        adaptive_crop_shape = str(adaptive_crop_shape or "square").strip().lower()

        if adaptive_crop_shape not in {"square", "preserve_aspect", "closest_rectangle"}:
            raise ValueError(
                f"adaptive_crop_shape must be 'square', 'preserve_aspect', "
                f"or 'closest_rectangle', "
                f"got {adaptive_crop_shape!r}"
            )

        _, img_h, img_w = base_x.shape

        raw_bbox = _get_mask_bbox_from_tensor(
            fused_mask_tensor,
            threshold=crop_threshold,
            min_area=self.crop_min_area,
        )

        crop_info = _default_crop_info(
            crop_threshold=crop_threshold,
            crop_margin_ratio=crop_margin_ratio,
            crop_policy_index=crop_policy_index,
            crop_mode=crop_mode,
            target_bbox_ratio=target_bbox_ratio,
            target_mask_area_ratio=target_mask_area_ratio,
            adaptive_scale_factor=adaptive_scale_factor,
            adaptive_crop_shape=adaptive_crop_shape,
        )

        crop_info["crop_bbox_found"] = raw_bbox is not None

        if raw_bbox is None:
            crop_info["crop_fallback_used"] = True

            if self.crop_fallback_mode == "keep_original":
                return base_x, fused_mask_tensor, crop_info

            if self.crop_fallback_mode == "background_mode":
                return self._apply_background_mode(base_x, fused_mask_tensor), fused_mask_tensor, crop_info

            raise RuntimeError(f"Unsupported crop_fallback_mode={self.crop_fallback_mode}")

        if crop_mode == "adaptive":
            bbox = _adaptive_bbox_from_raw_bbox(
                raw_bbox,
                img_h=img_h,
                img_w=img_w,
                target_bbox_ratio=target_bbox_ratio,
                adaptive_scale_factor=adaptive_scale_factor,
                adaptive_crop_shape=adaptive_crop_shape,
            )
            adaptive_mask_result = None
        elif crop_mode == "adaptive_mask_area":
            if target_mask_area_ratio <= 0.0:
                if target_bbox_ratio > 0.0:
                    target_mask_area_ratio = target_bbox_ratio
                else:
                    raise ValueError(
                        "crop_mode='adaptive_mask_area' requires "
                        "target_mask_area_ratio in the policy."
                    )

            mask_np = fused_mask_tensor.detach().cpu().squeeze(0).float().numpy()
            adaptive_mask_result = adaptive_crop_for_target_ratio(
                mask_np,
                target_ratio=target_mask_area_ratio,
                image_size=(img_w, img_h),
                mask_threshold=crop_threshold,
                final_size=self.img_size,
                min_mask_area=self.crop_min_area,
            )
            bbox = adaptive_mask_result.bbox
        elif crop_mode == "fixed_margin":
            bbox = _expand_bbox(
                raw_bbox,
                img_h=img_h,
                img_w=img_w,
                margin_ratio=crop_margin_ratio,
            )
            adaptive_mask_result = None
        else:
            raise ValueError(f"Unknown crop_mode={crop_mode!r}")

        x1, y1, x2, y2 = bbox
        crop_x = _crop_tensor_by_bbox(base_x, bbox)
        crop_mask = _crop_tensor_by_bbox(fused_mask_tensor, bbox)

        crop_area = float((x2 - x1) * (y2 - y1))
        full_area = float(max(1, img_h * img_w))

        rx1, ry1, rx2, ry2 = raw_bbox
        raw_bbox_area = float((rx2 - rx1) * (ry2 - ry1))

        crop_area_ratio = crop_area / full_area
        raw_bbox_area_ratio = raw_bbox_area / full_area
        achieved_bbox_ratio = raw_bbox_area / max(1.0, crop_area)
        mask_area_in_crop = float((crop_mask >= crop_threshold).sum().item())
        achieved_mask_area_ratio = mask_area_in_crop / max(1.0, crop_area)

        crop_info.update(
            {
                "crop_bbox_found": True,
                "crop_bbox_x1": x1,
                "crop_bbox_y1": y1,
                "crop_bbox_x2": x2,
                "crop_bbox_y2": y2,
                "crop_area_ratio": float(crop_area_ratio),
                "effective_crop_threshold": float(crop_threshold),
                "effective_crop_margin_ratio": float(crop_margin_ratio),
                "crop_policy_index": int(crop_policy_index),

                "crop_mode": crop_mode,
                "target_bbox_ratio": float(target_bbox_ratio),
                "target_mask_area_ratio": float(target_mask_area_ratio),
                "adaptive_scale_factor": float(adaptive_scale_factor),
                "adaptive_crop_shape": str(adaptive_crop_shape),
                "achieved_bbox_ratio_in_crop": float(achieved_bbox_ratio),
                "achieved_mask_area_ratio_in_crop": float(achieved_mask_area_ratio),
                "raw_bbox_area_ratio": float(raw_bbox_area_ratio),
            }
        )

        if adaptive_mask_result is not None:
            crop_info.update(
                {
                    "adaptive_mask_area_target_possible": bool(
                        adaptive_mask_result.target_possible
                    ),
                    "adaptive_mask_area_mode": str(adaptive_mask_result.mode),
                    "adaptive_mask_area_distortion_log_abs": float(
                        adaptive_mask_result.distortion_log_abs
                    ),
                    "adaptive_mask_area_resize_scale_x": float(
                        adaptive_mask_result.resize_scale_x
                    ),
                    "adaptive_mask_area_resize_scale_y": float(
                        adaptive_mask_result.resize_scale_y
                    ),
                }
            )

        return crop_x, crop_mask, crop_info

    def _aggregate_crop_infos(self, crop_infos: list[dict]) -> dict:
        """
        Aggregate multi-view crop infos into scalar-compatible fields.
        Existing code expects scalar crop metadata, so we keep those fields
        as means / any-flags, and also return view-level tensors in extras.
        """
        if not crop_infos:
            return _default_crop_info(
                crop_threshold=self.crop_threshold,
                crop_margin_ratio=self.crop_margin_ratio,
                crop_policy_index=-1,
            )

        area_ratios = [float(x.get("crop_area_ratio", 1.0)) for x in crop_infos]
        fallback_used = [bool(x.get("crop_fallback_used", False)) for x in crop_infos]
        bbox_found = [bool(x.get("crop_bbox_found", False)) for x in crop_infos]

        achieved_ratios = [
            float(x.get("achieved_bbox_ratio_in_crop", -1.0))
            for x in crop_infos
            if float(x.get("achieved_bbox_ratio_in_crop", -1.0)) >= 0.0
        ]
        raw_bbox_area_ratios = [
            float(x.get("raw_bbox_area_ratio", -1.0))
            for x in crop_infos
            if float(x.get("raw_bbox_area_ratio", -1.0)) >= 0.0
        ]

        first = dict(crop_infos[0])
        first.update(
            {
                "crop_fallback_used": any(fallback_used),
                "crop_bbox_found": all(bbox_found),
                "crop_area_ratio": float(np.mean(area_ratios)),
                "effective_crop_threshold": -1.0,
                "effective_crop_margin_ratio": -1.0,
                "crop_policy_index": -1,
                "achieved_bbox_ratio_in_crop": (
                    float(np.mean(achieved_ratios)) if achieved_ratios else -1.0
                ),
                "raw_bbox_area_ratio": (
                    float(np.mean(raw_bbox_area_ratios)) if raw_bbox_area_ratios else -1.0
                ),
            }
        )
        return first

    def _extract_mask_metadata(
        self,
        *,
        row: pd.Series,
        image_id: str,
        mask_available: bool,
        fused_mask_tensor: torch.Tensor,
        crop_info: dict | None = None,
        crop_infos: list[dict] | None = None,
    ) -> dict[str, torch.Tensor]:
        has_majority_flag = None
        mean_teacher_confidence = None

        if "has_majority_flag" in self.df.columns:
            has_majority_flag = _safe_float_or_none(row.get("has_majority_flag"))

        if "mean_teacher_confidence" in self.df.columns:
            mean_teacher_confidence = _safe_float_or_none(row.get("mean_teacher_confidence"))

        if has_majority_flag is None and self.fused_mask_loader is not None:
            if hasattr(self.fused_mask_loader, "get_has_majority_flag"):
                try:
                    has_majority_flag = _safe_float_or_none(
                        self.fused_mask_loader.get_has_majority_flag(image_id)
                    )
                except Exception:
                    has_majority_flag = None

        if mean_teacher_confidence is None and self.fused_mask_loader is not None:
            if hasattr(self.fused_mask_loader, "get_mean_teacher_confidence"):
                try:
                    mean_teacher_confidence = _safe_float_or_none(
                        self.fused_mask_loader.get_mean_teacher_confidence(image_id)
                    )
                except Exception:
                    mean_teacher_confidence = None

        has_majority_missing = float(has_majority_flag is None)
        mean_teacher_confidence_missing = float(mean_teacher_confidence is None)

        if has_majority_flag is None:
            has_majority_flag = 0.0
        if mean_teacher_confidence is None:
            mean_teacher_confidence = 0.0

        mask_mean = float(fused_mask_tensor.mean().item())
        mask_min = float(fused_mask_tensor.min().item())
        mask_max = float(fused_mask_tensor.max().item())

        mask_is_all_zeros = float(mask_max <= 1e-8)
        mask_is_all_ones = float(mask_min >= 1.0 - 1e-8 and mask_max >= 1.0 - 1e-8)

        if crop_info is None:
            crop_info = {}

        extras = {
            "has_majority_flag": torch.tensor(has_majority_flag, dtype=torch.float32),
            "mean_teacher_confidence": torch.tensor(
                mean_teacher_confidence, dtype=torch.float32
            ),
            "has_majority_flag_missing": torch.tensor(
                has_majority_missing, dtype=torch.float32
            ),
            "mean_teacher_confidence_missing": torch.tensor(
                mean_teacher_confidence_missing, dtype=torch.float32
            ),
            "mask_available": torch.tensor(float(mask_available), dtype=torch.float32),
            "mask_missing": torch.tensor(float(not mask_available), dtype=torch.float32),
            "mask_mean": torch.tensor(mask_mean, dtype=torch.float32),
            "mask_is_all_zeros": torch.tensor(mask_is_all_zeros, dtype=torch.float32),
            "mask_is_all_ones": torch.tensor(mask_is_all_ones, dtype=torch.float32),
            "use_crop": torch.tensor(float(self.use_crop), dtype=torch.float32),
            "multi_view_crop": torch.tensor(float(self.multi_view_crop), dtype=torch.float32),
            "num_crop_views": torch.tensor(
                float(len(crop_infos) if crop_infos is not None else 1),
                dtype=torch.float32,
            ),
            "crop_fallback_used": torch.tensor(
                float(crop_info.get("crop_fallback_used", False)),
                dtype=torch.float32,
            ),
            "crop_bbox_found": torch.tensor(
                float(crop_info.get("crop_bbox_found", False)),
                dtype=torch.float32,
            ),
            "crop_area_ratio": torch.tensor(
                float(crop_info.get("crop_area_ratio", 1.0)),
                dtype=torch.float32,
            ),
            "effective_crop_threshold": torch.tensor(
                float(crop_info.get("effective_crop_threshold", self.crop_threshold)),
                dtype=torch.float32,
            ),
            "effective_crop_margin_ratio": torch.tensor(
                float(crop_info.get("effective_crop_margin_ratio", self.crop_margin_ratio)),
                dtype=torch.float32,
            ),
            "crop_policy_index": torch.tensor(
                float(crop_info.get("crop_policy_index", -1)),
                dtype=torch.float32,
            ),
            "target_bbox_ratio": torch.tensor(
                float(crop_info.get("target_bbox_ratio", -1.0)),
                dtype=torch.float32,
            ),
            "target_mask_area_ratio": torch.tensor(
                float(crop_info.get("target_mask_area_ratio", -1.0)),
                dtype=torch.float32,
            ),
            "adaptive_scale_factor": torch.tensor(
                float(crop_info.get("adaptive_scale_factor", 1.0)),
                dtype=torch.float32,
            ),
            "achieved_bbox_ratio_in_crop": torch.tensor(
                float(crop_info.get("achieved_bbox_ratio_in_crop", -1.0)),
                dtype=torch.float32,
            ),
            "achieved_mask_area_ratio_in_crop": torch.tensor(
                float(crop_info.get("achieved_mask_area_ratio_in_crop", -1.0)),
                dtype=torch.float32,
            ),
            "adaptive_mask_area_target_possible": torch.tensor(
                float(crop_info.get("adaptive_mask_area_target_possible", False)),
                dtype=torch.float32,
            ),
            "adaptive_mask_area_distortion_log_abs": torch.tensor(
                float(crop_info.get("adaptive_mask_area_distortion_log_abs", -1.0)),
                dtype=torch.float32,
            ),
            "raw_bbox_area_ratio": torch.tensor(
                float(crop_info.get("raw_bbox_area_ratio", -1.0)),
                dtype=torch.float32,
            ),
        }

        if crop_infos is not None:
            extras.update(
                {
                    "view_crop_area_ratios": torch.tensor(
                        [float(x.get("crop_area_ratio", 1.0)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_effective_crop_thresholds": torch.tensor(
                        [float(x.get("effective_crop_threshold", -1.0)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_effective_crop_margin_ratios": torch.tensor(
                        [float(x.get("effective_crop_margin_ratio", -1.0)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_crop_policy_indices": torch.tensor(
                        [float(x.get("crop_policy_index", -1)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_crop_fallback_used": torch.tensor(
                        [float(x.get("crop_fallback_used", False)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_crop_bbox_found": torch.tensor(
                        [float(x.get("crop_bbox_found", False)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_target_bbox_ratios": torch.tensor(
                        [float(x.get("target_bbox_ratio", -1.0)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_target_mask_area_ratios": torch.tensor(
                        [float(x.get("target_mask_area_ratio", -1.0)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_adaptive_scale_factors": torch.tensor(
                        [float(x.get("adaptive_scale_factor", 1.0)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_achieved_bbox_ratios": torch.tensor(
                        [float(x.get("achieved_bbox_ratio_in_crop", -1.0)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_achieved_mask_area_ratios": torch.tensor(
                        [float(x.get("achieved_mask_area_ratio_in_crop", -1.0)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_adaptive_mask_area_target_possible": torch.tensor(
                        [float(x.get("adaptive_mask_area_target_possible", False)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_adaptive_mask_area_distortion_log_abs": torch.tensor(
                        [float(x.get("adaptive_mask_area_distortion_log_abs", -1.0)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                    "view_raw_bbox_area_ratios": torch.tensor(
                        [float(x.get("raw_bbox_area_ratio", -1.0)) for x in crop_infos],
                        dtype=torch.float32,
                    ),
                }
            )
        else:
            extras.update(
                {
                    "view_crop_area_ratios": torch.tensor(
                        [float(crop_info.get("crop_area_ratio", 1.0))],
                        dtype=torch.float32,
                    ),
                    "view_effective_crop_thresholds": torch.tensor(
                        [float(crop_info.get("effective_crop_threshold", self.crop_threshold))],
                        dtype=torch.float32,
                    ),
                    "view_effective_crop_margin_ratios": torch.tensor(
                        [float(crop_info.get("effective_crop_margin_ratio", self.crop_margin_ratio))],
                        dtype=torch.float32,
                    ),
                    "view_crop_policy_indices": torch.tensor(
                        [float(crop_info.get("crop_policy_index", -1))],
                        dtype=torch.float32,
                    ),
                    "view_crop_fallback_used": torch.tensor(
                        [float(crop_info.get("crop_fallback_used", False))],
                        dtype=torch.float32,
                    ),
                    "view_crop_bbox_found": torch.tensor(
                        [float(crop_info.get("crop_bbox_found", False))],
                        dtype=torch.float32,
                    ),
                    "view_target_bbox_ratios": torch.tensor(
                        [float(crop_info.get("target_bbox_ratio", -1.0))],
                        dtype=torch.float32,
                    ),
                    "view_target_mask_area_ratios": torch.tensor(
                        [float(crop_info.get("target_mask_area_ratio", -1.0))],
                        dtype=torch.float32,
                    ),
                    "view_adaptive_scale_factors": torch.tensor(
                        [float(crop_info.get("adaptive_scale_factor", 1.0))],
                        dtype=torch.float32,
                    ),
                    "view_achieved_bbox_ratios": torch.tensor(
                        [float(crop_info.get("achieved_bbox_ratio_in_crop", -1.0))],
                        dtype=torch.float32,
                    ),
                    "view_achieved_mask_area_ratios": torch.tensor(
                        [float(crop_info.get("achieved_mask_area_ratio_in_crop", -1.0))],
                        dtype=torch.float32,
                    ),
                    "view_adaptive_mask_area_target_possible": torch.tensor(
                        [float(crop_info.get("adaptive_mask_area_target_possible", False))],
                        dtype=torch.float32,
                    ),
                    "view_adaptive_mask_area_distortion_log_abs": torch.tensor(
                        [float(crop_info.get("adaptive_mask_area_distortion_log_abs", -1.0))],
                        dtype=torch.float32,
                    ),
                    "view_raw_bbox_area_ratios": torch.tensor(
                        [float(crop_info.get("raw_bbox_area_ratio", -1.0))],
                        dtype=torch.float32,
                    ),
                }
            )

        return extras

    def _sample_color_jitter_params(self):
        if not self.use_color_jitter:
            return None

        fn_idx = torch.randperm(4).tolist()

        b = None
        c = None
        s = None
        h = None

        if self.color_jitter_brightness > 0:
            b = random.uniform(
                max(0.0, 1.0 - self.color_jitter_brightness),
                1.0 + self.color_jitter_brightness,
            )
        if self.color_jitter_contrast > 0:
            c = random.uniform(
                max(0.0, 1.0 - self.color_jitter_contrast),
                1.0 + self.color_jitter_contrast,
            )
        if self.color_jitter_saturation > 0:
            s = random.uniform(
                max(0.0, 1.0 - self.color_jitter_saturation),
                1.0 + self.color_jitter_saturation,
            )
        if self.color_jitter_hue > 0:
            h = random.uniform(
                -self.color_jitter_hue,
                self.color_jitter_hue,
            )

        return {
            "fn_idx": fn_idx,
            "brightness_factor": b,
            "contrast_factor": c,
            "saturation_factor": s,
            "hue_factor": h,
        }

    def _sample_per_view_color_jitter_params(
        self,
        base_params: dict | None,
    ) -> dict | None:
        """
        Copy the regular ColorJitter parameters, but optionally re-sample
        selected factors independently per crop/view.

        Each flag controls one factor:
            per_view_color_jitter_brightness
            per_view_color_jitter_contrast
            per_view_color_jitter_saturation
            per_view_color_jitter_hue

        Notes:
            - The operation order fn_idx stays shared across views.
            - Any factor whose flag is False stays shared across views.
            - Any factor whose regular jitter magnitude is 0 stays None/no-op.
        """
        if base_params is None:
            return None

        if not self.per_view_color_jitter_any:
            return base_params

        view_params = dict(base_params)

        if (
            self.per_view_color_jitter_brightness
            and self.color_jitter_brightness > 0
        ):
            view_params["brightness_factor"] = random.uniform(
                max(0.0, 1.0 - self.color_jitter_brightness),
                1.0 + self.color_jitter_brightness,
            )

        if self.per_view_color_jitter_contrast and self.color_jitter_contrast > 0:
            view_params["contrast_factor"] = random.uniform(
                max(0.0, 1.0 - self.color_jitter_contrast),
                1.0 + self.color_jitter_contrast,
            )

        if (
            self.per_view_color_jitter_saturation
            and self.color_jitter_saturation > 0
        ):
            view_params["saturation_factor"] = random.uniform(
                max(0.0, 1.0 - self.color_jitter_saturation),
                1.0 + self.color_jitter_saturation,
            )

        if self.per_view_color_jitter_hue and self.color_jitter_hue > 0:
            view_params["hue_factor"] = random.uniform(
                -self.color_jitter_hue,
                self.color_jitter_hue,
            )

        return view_params

    def _apply_same_color_jitter(
        self,
        x: torch.Tensor,
        jitter_params: dict | None,
    ) -> torch.Tensor:
        if jitter_params is None:
            return x

        out = x
        for fn_id in jitter_params["fn_idx"]:
            if fn_id == 0 and jitter_params["brightness_factor"] is not None:
                out = TF.adjust_brightness(out, jitter_params["brightness_factor"])
            elif fn_id == 1 and jitter_params["contrast_factor"] is not None:
                out = TF.adjust_contrast(out, jitter_params["contrast_factor"])
            elif fn_id == 2 and jitter_params["saturation_factor"] is not None:
                out = TF.adjust_saturation(out, jitter_params["saturation_factor"])
            elif fn_id == 3 and jitter_params["hue_factor"] is not None:
                out = TF.adjust_hue(out, jitter_params["hue_factor"])

        return out.clamp(0.0, 1.0)

    def _apply_same_geometric(
        self,
        x: torch.Tensor,
        *,
        hflip: bool,
        vflip: bool,
        angle: float,
    ) -> torch.Tensor:
        """
        Apply the same sampled geometric augmentation to an image tensor.

        This is used after crop+resize in crop experiments so the crop bbox is
        computed from the original mask/image frame, and only then the cropped
        view is augmented.

        Args:
            x:
                Tensor of shape (C,H,W), values in [0,1].
            hflip / vflip / angle:
                Sampled once per original image and shared by all crop views.

        Returns:
            Tensor with the same shape as x.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected tensor shape (C,H,W), got {tuple(x.shape)}")

        out = x

        if hflip:
            out = torch.flip(out, dims=[-1])

        if vflip:
            out = torch.flip(out, dims=[-2])

        if float(angle) != 0.0:
            out = TF.rotate(
                out,
                float(angle),
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0,
            )

        return out.clamp(0.0, 1.0)


    def _final_preprocess_masks_many(
        self,
        mask_views: list[torch.Tensor],
        *,
        hflip: bool = False,
        vflip: bool = False,
        angle: float = 0.0,
        apply_geometric_after_resize: bool = False,
    ) -> torch.Tensor:
        """
        Final preprocessing for masks that must stay aligned with second-branch images.

        Returns:
            single-view input list:
                (1, 1, H, W) stacked internally, caller may index [0]

            multi-view input list:
                (V, 1, H, W)

        Important:
            No color jitter and no ImageNet normalization are applied to masks.
            The resize mode mirrors crop image preprocessing:
                stretch    -> direct resize to img_size x img_size
                letterbox  -> preserve aspect ratio + padding
        """
        if not mask_views:
            raise ValueError("mask_views is empty")

        for i, mask_x in enumerate(mask_views):
            if mask_x.ndim != 3 or mask_x.shape[0] != 1:
                raise ValueError(
                    f"Expected mask_views[{i}] to be (1,H,W), "
                    f"got {tuple(mask_x.shape)}"
                )

        if self.use_crop and self.crop_resize_mode == "letterbox":
            resized_masks = [
                _resize_letterbox_tensor(
                    mask_x.float().clamp(0.0, 1.0),
                    self.img_size,
                    fill_value=0.0,
                    interpolation=TF.InterpolationMode.BILINEAR,
                ).clamp(0.0, 1.0)
                for mask_x in mask_views
            ]
        else:
            resized_masks = [
                TF.resize(
                    mask_x.float().clamp(0.0, 1.0),
                    [self.img_size, self.img_size],
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                ).clamp(0.0, 1.0)
                for mask_x in mask_views
            ]

        if apply_geometric_after_resize:
            resized_masks = [
                self._apply_same_geometric(
                    mask_x,
                    hflip=bool(hflip),
                    vflip=bool(vflip),
                    angle=float(angle),
                ).clamp(0.0, 1.0)
                for mask_x in resized_masks
            ]

        return torch.stack(resized_masks, dim=0).to(torch.float32)

    def _final_preprocess_many(
        self,
        orig_x: torch.Tensor,
        second_branch_views: list[torch.Tensor],
        *,
        hflip: bool = False,
        vflip: bool = False,
        angle: float = 0.0,
        apply_geometric_after_resize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """
        Final preprocessing for original image and one/many second-branch views.

        Returns:
            orig_x_norm:
                (C, H, W)
            second_branch_views_norm:
                (V, C, H, W)

        Important:
            By default, the SAME ColorJitter params are applied to original image
            and all views.

            If any per_view_color_jitter_* flag is True, the original image still
            receives the base ColorJitter params, while each second-branch crop
            view may get a copied params dict with selected factors re-sampled
            per view.

            If apply_geometric_after_resize=True, the SAME sampled hflip/vflip/rotation
            is applied to the resized original image and to every resized crop view.
            This is the desired behavior for crop experiments:
                crop -> resize -> geometric augmentation -> color jitter -> normalize
        """
        if orig_x.ndim != 3:
            raise ValueError(f"Expected orig_x to be (C,H,W), got {tuple(orig_x.shape)}")

        if not second_branch_views:
            raise ValueError("second_branch_views is empty")

        for i, view_x in enumerate(second_branch_views):
            if view_x.ndim != 3:
                raise ValueError(
                    f"Expected second_branch_views[{i}] to be (C,H,W), "
                    f"got {tuple(view_x.shape)}"
                )

        orig_x = TF.resize(
            orig_x,
            [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

        if self.use_crop and self.crop_resize_mode == "letterbox":
            resized_views = [
                _resize_letterbox_tensor(
                    view_x,
                    self.img_size,
                    fill_value=self.crop_letterbox_fill,
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
                for view_x in second_branch_views
            ]
        else:
            resized_views = [
                TF.resize(
                    view_x,
                    [self.img_size, self.img_size],
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                for view_x in second_branch_views
            ]

        if apply_geometric_after_resize:
            orig_x = self._apply_same_geometric(
                orig_x,
                hflip=bool(hflip),
                vflip=bool(vflip),
                angle=float(angle),
            )
            resized_views = [
                self._apply_same_geometric(
                    view_x,
                    hflip=bool(hflip),
                    vflip=bool(vflip),
                    angle=float(angle),
                )
                for view_x in resized_views
            ]

        jitter_params = self._sample_color_jitter_params()

        orig_x = self._apply_same_color_jitter(orig_x, jitter_params)
        resized_views = [
            self._apply_same_color_jitter(
                view_x,
                self._sample_per_view_color_jitter_params(jitter_params),
            )
            for view_x in resized_views
        ]

        orig_x = TF.normalize(orig_x, mean=self.normalize_mean, std=self.normalize_std)

        normalized_views = [
            TF.normalize(view_x, mean=self.normalize_mean, std=self.normalize_std)
            for view_x in resized_views
        ]

        second_branch_views_x = torch.stack(normalized_views, dim=0)

        return orig_x, second_branch_views_x, jitter_params

    def _final_preprocess_pair(
        self,
        orig_x: torch.Tensor,
        second_branch_x: torch.Tensor,
        *,
        hflip: bool = False,
        vflip: bool = False,
        angle: float = 0.0,
        apply_geometric_after_resize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        orig_x_norm, second_branch_views_x, jitter_params = self._final_preprocess_many(
            orig_x=orig_x,
            second_branch_views=[second_branch_x],
            hflip=hflip,
            vflip=vflip,
            angle=angle,
            apply_geometric_after_resize=apply_geometric_after_resize,
        )

        return orig_x_norm, second_branch_views_x[0], jitter_params

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row[self.img_col_name])
        image_id = Path(img_path).stem
        row_idx = int(row["row_idx"])
        row_label = row[self.label_col]

        if row_idx < 0 or row_idx >= len(self.meta_features):
            raise IndexError(
                f"row_idx={row_idx} is out of bounds for meta_features "
                f"with len={len(self.meta_features)}"
            )

        if row_label not in self.label2idx:
            raise KeyError(
                f"Label {row_label!r} not found in label2idx. "
                f"Known labels: {sorted(self.label2idx.keys())}"
            )

        with Image.open(img_path) as img:
            pil_img = img.convert("RGB")

        if self.agcwd_transform is not None:
            pil_img = self.agcwd_transform(pil_img)

        do_hflip = (self.hflip_p > 0.0) and (random.random() < self.hflip_p)
        do_vflip = (self.vflip_p > 0.0) and (random.random() < self.vflip_p)
        angle = (
            random.uniform(-self.rotation_deg, self.rotation_deg)
            if self.rotation_deg > 0.0
            else 0.0
        )

        # ------------------------------------------------------------
        # Geometric augmentation order
        # ------------------------------------------------------------
        # For crop experiments, the desired order is:
        #     original image + original mask -> crop -> resize -> hflip/vflip/rotation
        #
        # Therefore, when self.use_crop=True, we do NOT apply hflip/vflip/rotation
        # to the full image before computing the crop bbox.
        #
        # For non-crop masked-background experiments, we keep the older behavior:
        #     hflip/vflip/rotation image and mask in the same full-image frame
        #     then build the second branch.
        apply_geometric_before_crop = not self.use_crop

        if apply_geometric_before_crop:
            if do_hflip:
                pil_img = TF.hflip(pil_img)
            if do_vflip:
                pil_img = TF.vflip(pil_img)
            if angle != 0.0:
                pil_img = TF.rotate(
                    pil_img,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0,
                )

        base_x = TF.to_tensor(pil_img)

        mask_hflip = do_hflip if apply_geometric_before_crop else False
        mask_vflip = do_vflip if apply_geometric_before_crop else False
        mask_angle = angle if apply_geometric_before_crop else 0.0

        mask_source = "missing"
        raw_mask_was_found = False
        loader_has_image = False
        loader_row_idx = None
        loader_label = None
        mask_available = False

        if self.sanity_check_all_ones_masks:
            fused_mask_tensor = torch.ones(
                1, base_x.shape[1], base_x.shape[2], dtype=torch.float32
            )
            mask_source = "sanity_all_ones"
            mask_available = True
        else:
            fused_mask_tensor: Optional[torch.Tensor] = None

            if self.fused_mask_loader is not None:
                loader_has_image = self.fused_mask_loader.has_mask(image_id)

                if loader_has_image:
                    if self.validate_loader_alignment:
                        loader_row_idx = self.fused_mask_loader.get_row_idx(image_id)
                        loader_label = self.fused_mask_loader.get_label(image_id)

                        if loader_row_idx is not None and int(loader_row_idx) != row_idx:
                            raise ValueError(
                                f"Row mismatch for image_id='{image_id}': "
                                f"dataset row_idx={row_idx}, loader row_idx={loader_row_idx}"
                            )

                        if loader_label is not None and str(loader_label) != str(row_label):
                            raise ValueError(
                                f"Label mismatch for image_id='{image_id}': "
                                f"dataset label={row_label}, loader label={loader_label}"
                            )

                    mask_np = self.fused_mask_loader.get_mask(image_id)

                    if mask_np is not None:
                        raw_mask_was_found = True
                        fused_mask_tensor = _sync_geometric_to_mask(
                            mask_np=mask_np,
                            target_size=(base_x.shape[1], base_x.shape[2]),
                            hflip=mask_hflip,
                            vflip=mask_vflip,
                            angle=mask_angle,
                        )
                        mask_source = "loader"
                        mask_available = True

            if fused_mask_tensor is None:
                fused_mask_tensor = self._build_fallback_mask(
                    base_x.shape[1],
                    base_x.shape[2],
                )
                mask_source = f"fallback_{self.missing_mask_mode}"
                mask_available = False

        if self.mask_power != 1.0:
            fused_mask_tensor = fused_mask_tensor.clamp(0.0, 1.0).pow(self.mask_power)

        fused_mask_tensor = fused_mask_tensor.clamp(0.0, 1.0).to(
            dtype=base_x.dtype,
            device=base_x.device,
        )

        crop_info = _default_crop_info(
            crop_threshold=self.crop_threshold,
            crop_margin_ratio=self.crop_margin_ratio,
            crop_policy_index=-1,
        )
        crop_infos = None
        second_branch_mode = self.background_mode
        second_branch_base_debug_shapes = None
        mask_context_base_x = (
            self._apply_background_mode(base_x, fused_mask_tensor)
            if (
                self.return_mask_context_image
                and (
                    self.mask_context_view_mode == "full"
                    or not self.use_crop
                )
            )
            else None
        )
        mask_context_x = None

        if self.use_crop:
            if self.multi_view_crop:
                second_branch_base_views = []
                second_branch_base_masks = []
                mask_context_base_views = []
                crop_infos = []

                for crop_policy in self.multi_view_crop_policy_choices:
                    view_x, view_mask, view_info = self._build_crop_branch_image(
                        base_x=base_x,
                        fused_mask_tensor=fused_mask_tensor,
                        crop_threshold=crop_policy["crop_threshold"],
                        crop_margin_ratio=crop_policy["crop_margin_ratio"],
                        crop_policy_index=crop_policy["policy_index"],
                        crop_mode=crop_policy.get("crop_mode", "fixed_margin"),
                        target_bbox_ratio=crop_policy.get("target_bbox_ratio", -1.0),
                        target_mask_area_ratio=crop_policy.get("target_mask_area_ratio", -1.0),
                        adaptive_scale_factor=crop_policy.get("adaptive_scale_factor", 1.0),
                        adaptive_crop_shape=crop_policy.get("adaptive_crop_shape", "square"),
                    )
                    second_branch_base_views.append(view_x)
                    second_branch_base_masks.append(view_mask)
                    if (
                        self.return_mask_context_image
                        and self.mask_context_view_mode == "crop_aligned"
                    ):
                        mask_context_base_views.append(
                            self._apply_background_mode(view_x, view_mask)
                        )
                    crop_infos.append(view_info)

                crop_info = self._aggregate_crop_infos(crop_infos)
                second_branch_mode = "multi_view_crop"
                second_branch_base_debug_shapes = [
                    tuple(view_x.shape) for view_x in second_branch_base_views
                ]

                preprocess_views = list(second_branch_base_views)
                if mask_context_base_x is not None:
                    preprocess_views.append(mask_context_base_x)
                elif mask_context_base_views:
                    preprocess_views.extend(mask_context_base_views)

                orig_x, processed_views_x, jitter_params = self._final_preprocess_many(
                    orig_x=base_x,
                    second_branch_views=preprocess_views,
                    hflip=do_hflip,
                    vflip=do_vflip,
                    angle=angle,
                    apply_geometric_after_resize=True,
                )
                second_branch_x = processed_views_x[:len(second_branch_base_views)]
                if mask_context_base_x is not None:
                    mask_context_x = processed_views_x[len(second_branch_base_views)]
                elif mask_context_base_views:
                    start = len(second_branch_base_views)
                    stop = start + len(mask_context_base_views)
                    mask_context_x = processed_views_x[start:stop]

                second_branch_mask = self._final_preprocess_masks_many(
                    mask_views=second_branch_base_masks,
                    hflip=do_hflip,
                    vflip=do_vflip,
                    angle=angle,
                    apply_geometric_after_resize=True,
                )

            else:
                crop_policy = self._sample_crop_policy()

                second_branch_base_x, second_branch_base_mask, crop_info = self._build_crop_branch_image(
                    base_x=base_x,
                    fused_mask_tensor=fused_mask_tensor,
                    crop_threshold=crop_policy["crop_threshold"],
                    crop_margin_ratio=crop_policy["crop_margin_ratio"],
                    crop_policy_index=crop_policy["policy_index"],
                    crop_mode=crop_policy.get("crop_mode", "fixed_margin"),
                    target_bbox_ratio=crop_policy.get("target_bbox_ratio", -1.0),
                    target_mask_area_ratio=crop_policy.get("target_mask_area_ratio", -1.0),
                    adaptive_scale_factor=crop_policy.get("adaptive_scale_factor", 1.0),
                    adaptive_crop_shape=crop_policy.get("adaptive_crop_shape", "square"),
                )
                second_branch_mode = "crop"
                second_branch_base_debug_shapes = tuple(second_branch_base_x.shape)

                mask_context_base_crop_x = None
                if (
                    self.return_mask_context_image
                    and self.mask_context_view_mode == "crop_aligned"
                ):
                    mask_context_base_crop_x = self._apply_background_mode(
                        second_branch_base_x,
                        second_branch_base_mask,
                    )

                preprocess_views = [second_branch_base_x]
                if mask_context_base_x is not None:
                    preprocess_views.append(mask_context_base_x)
                elif mask_context_base_crop_x is not None:
                    preprocess_views.append(mask_context_base_crop_x)

                orig_x, processed_views_x, jitter_params = self._final_preprocess_many(
                    orig_x=base_x,
                    second_branch_views=preprocess_views,
                    hflip=do_hflip,
                    vflip=do_vflip,
                    angle=angle,
                    apply_geometric_after_resize=True,
                )
                second_branch_x = processed_views_x[0]
                if mask_context_base_x is not None:
                    mask_context_x = processed_views_x[1]
                elif mask_context_base_crop_x is not None:
                    mask_context_x = processed_views_x[1]

                second_branch_mask = self._final_preprocess_masks_many(
                    mask_views=[second_branch_base_mask],
                    hflip=do_hflip,
                    vflip=do_vflip,
                    angle=angle,
                    apply_geometric_after_resize=True,
                )[0]
        else:
            second_branch_base_x = self._apply_background_mode(
                base_x,
                fused_mask_tensor,
            )
            second_branch_mode = self.background_mode
            second_branch_base_debug_shapes = tuple(second_branch_base_x.shape)

            orig_x, second_branch_x, jitter_params = self._final_preprocess_pair(
                orig_x=base_x,
                second_branch_x=second_branch_base_x,
            )
            if self.return_mask_context_image:
                mask_context_x = second_branch_x

            second_branch_mask = self._final_preprocess_masks_many(
                mask_views=[fused_mask_tensor],
                apply_geometric_after_resize=False,
            )[0]

        meta_vec = self.meta_features[row_idx].astype(np.float32, copy=False)
        meta_vec = torch.from_numpy(meta_vec)

        label_idx = torch.tensor(self.label2idx[row_label], dtype=torch.long)

        extras = self._extract_mask_metadata(
            row=row,
            image_id=image_id,
            mask_available=mask_available,
            fused_mask_tensor=fused_mask_tensor,
            crop_info=crop_info,
            crop_infos=crop_infos,
        )

        # Mask aligned to the exact tensor used as model input.
        # This is intentionally stored inside extras so existing training code
        # remains unchanged when use_mask_attention=False.
        extras["second_branch_mask"] = second_branch_mask.to(torch.float32).clamp(0.0, 1.0)
        extras["attention_mask"] = extras["second_branch_mask"]
        if self.return_mask_context_image:
            if mask_context_x is None:
                raise RuntimeError(
                    "return_mask_context_image=True but mask_context_x was not built."
                )
            extras["mask_context_image"] = mask_context_x.to(torch.float32)

        if self._should_debug_sample():
            self._debug_seen_count += 1

            mask_min = float(fused_mask_tensor.min().item())
            mask_max = float(fused_mask_tensor.max().item())
            mask_mean = float(fused_mask_tensor.mean().item())

            if second_branch_x.ndim == 3 and orig_x.ndim == 3:
                final_delta = (second_branch_x - orig_x).abs()
                final_delta_mean = float(final_delta.mean().item())
                final_delta_max = float(final_delta.max().item())
            elif second_branch_x.ndim == 4 and orig_x.ndim == 3:
                orig_view = orig_x.unsqueeze(0).expand_as(second_branch_x)
                final_delta = (second_branch_x - orig_view).abs()
                final_delta_mean = float(final_delta.mean().item())
                final_delta_max = float(final_delta.max().item())
            else:
                final_delta_mean = float("nan")
                final_delta_max = float("nan")

            print("-" * 80)
            print(f"[DEBUG][Dataset::__getitem__] idx={idx}")
            print(f"[DEBUG][Dataset::__getitem__] image_id={image_id}")
            print(f"[DEBUG][Dataset::__getitem__] row_idx={row_idx}")
            print(f"[DEBUG][Dataset::__getitem__] label_raw={row_label}")
            print(f"[DEBUG][Dataset::__getitem__] label_idx={int(label_idx.item())}")
            print(f"[DEBUG][Dataset::__getitem__] base_x.shape={tuple(base_x.shape)}")
            print(
                "[DEBUG][Dataset::__getitem__] "
                f"second_branch_base_debug_shapes={second_branch_base_debug_shapes}"
            )
            print(f"[DEBUG][Dataset::__getitem__] orig_x.shape={tuple(orig_x.shape)}")
            print(
                f"[DEBUG][Dataset::__getitem__] "
                f"second_branch_x.shape={tuple(second_branch_x.shape)}"
            )
            print(f"[DEBUG][Dataset::__getitem__] meta_vec.shape={tuple(meta_vec.shape)}")
            print(f"[DEBUG][Dataset::__getitem__] second_branch_mask.shape={tuple(second_branch_mask.shape)}")
            if mask_context_x is not None:
                print(
                    f"[DEBUG][Dataset::__getitem__] "
                    f"mask_context_image.shape={tuple(mask_context_x.shape)}"
                )
            print(f"[DEBUG][Dataset::__getitem__] use_crop={self.use_crop}")
            print(f"[DEBUG][Dataset::__getitem__] multi_view_crop={self.multi_view_crop}")
            print(f"[DEBUG][Dataset::__getitem__] second_branch_mode={second_branch_mode}")
            print(f"[DEBUG][Dataset::__getitem__] background_mode={self.background_mode}")
            print(f"[DEBUG][Dataset::__getitem__] mask_source={mask_source}")
            print(f"[DEBUG][Dataset::__getitem__] loader_has_image={loader_has_image}")
            print(f"[DEBUG][Dataset::__getitem__] raw_mask_was_found={raw_mask_was_found}")
            print(f"[DEBUG][Dataset::__getitem__] mask_available={mask_available}")
            print(
                f"[DEBUG][Dataset::__getitem__] aug sampled: "
                f"hflip={do_hflip}, vflip={do_vflip}, angle={angle:.4f}, "
                f"applied_after_crop={self.use_crop}"
            )
            print(f"[DEBUG][Dataset::__getitem__] jitter_params={jitter_params}")
            print(
                f"[DEBUG][Dataset::__getitem__] mask stats: "
                f"shape={tuple(fused_mask_tensor.shape)}, "
                f"min={mask_min:.6f}, max={mask_max:.6f}, mean={mask_mean:.6f}"
            )

            if crop_infos is None:
                print(
                    f"[DEBUG][Dataset::__getitem__] effective crop policy: "
                    f"threshold={crop_info.get('effective_crop_threshold')}, "
                    f"margin={crop_info.get('effective_crop_margin_ratio')}, "
                    f"policy_index={crop_info.get('crop_policy_index')}"
                )
                print(
                    f"[DEBUG][Dataset::__getitem__] crop: "
                    f"bbox_found={crop_info.get('crop_bbox_found')}, "
                    f"fallback_used={crop_info.get('crop_fallback_used')}, "
                    f"bbox=({crop_info.get('crop_bbox_x1')}, "
                    f"{crop_info.get('crop_bbox_y1')}, "
                    f"{crop_info.get('crop_bbox_x2')}, "
                    f"{crop_info.get('crop_bbox_y2')}), "
                    f"area_ratio={crop_info.get('crop_area_ratio'):.6f}"
                )
            else:
                print("[DEBUG][Dataset::__getitem__] multi-view crop infos:")
                for view_i, info_i in enumerate(crop_infos):
                    print(
                        f"  view={view_i} | "
                        f"mode={info_i.get('crop_mode')} | "
                        f"thr={info_i.get('effective_crop_threshold')} | "
                        f"margin={info_i.get('effective_crop_margin_ratio')} | "
                        f"target={info_i.get('target_bbox_ratio')} | "
                        f"achieved={info_i.get('achieved_bbox_ratio_in_crop')} | "
                        f"raw_bbox_area={info_i.get('raw_bbox_area_ratio')} | "
                        f"scale={info_i.get('adaptive_scale_factor')} | "
                        f"adaptive_shape={info_i.get('adaptive_crop_shape')} | "
                        f"policy_index={info_i.get('crop_policy_index')} | "
                        f"bbox_found={info_i.get('crop_bbox_found')} | "
                        f"fallback_used={info_i.get('crop_fallback_used')} | "
                        f"bbox=({info_i.get('crop_bbox_x1')}, "
                        f"{info_i.get('crop_bbox_y1')}, "
                        f"{info_i.get('crop_bbox_x2')}, "
                        f"{info_i.get('crop_bbox_y2')}) | "
                        f"area_ratio={info_i.get('crop_area_ratio'):.6f}"
                    )

            print(
                f"[DEBUG][Dataset::__getitem__] final delta stats after preprocess: "
                f"mean_abs_diff={final_delta_mean:.6f}, max_abs_diff={final_delta_max:.6f}"
            )
            print(
                "[DEBUG][Dataset::__getitem__] extras: "
                f"has_majority_flag={float(extras['has_majority_flag'].item()):.4f}, "
                f"mean_teacher_confidence={float(extras['mean_teacher_confidence'].item()):.4f}, "
                f"has_majority_flag_missing={float(extras['has_majority_flag_missing'].item()):.4f}, "
                f"mean_teacher_confidence_missing={float(extras['mean_teacher_confidence_missing'].item()):.4f}, "
                f"mask_available={float(extras['mask_available'].item()):.4f}, "
                f"mask_missing={float(extras['mask_missing'].item()):.4f}, "
                f"mask_mean={float(extras['mask_mean'].item()):.4f}, "
                f"mask_is_all_zeros={float(extras['mask_is_all_zeros'].item()):.4f}, "
                f"mask_is_all_ones={float(extras['mask_is_all_ones'].item()):.4f}, "
                f"use_crop={float(extras['use_crop'].item()):.4f}, "
                f"multi_view_crop={float(extras['multi_view_crop'].item()):.4f}, "
                f"num_crop_views={float(extras['num_crop_views'].item()):.0f}, "
                f"crop_fallback_used={float(extras['crop_fallback_used'].item()):.4f}, "
                f"crop_bbox_found={float(extras['crop_bbox_found'].item()):.4f}, "
                f"crop_area_ratio={float(extras['crop_area_ratio'].item()):.4f}, "
                f"effective_crop_threshold={float(extras['effective_crop_threshold'].item()):.4f}, "
                f"effective_crop_margin_ratio={float(extras['effective_crop_margin_ratio'].item()):.4f}, "
                f"crop_policy_index={float(extras['crop_policy_index'].item()):.0f}, "
                f"target_bbox_ratio={float(extras['target_bbox_ratio'].item()):.4f}, "
                f"adaptive_scale_factor={float(extras['adaptive_scale_factor'].item()):.4f}, "
                f"achieved_bbox_ratio={float(extras['achieved_bbox_ratio_in_crop'].item()):.4f}, "
                f"raw_bbox_area_ratio={float(extras['raw_bbox_area_ratio'].item()):.4f}"
            )

            if self.multi_view_crop:
                print(
                    "[DEBUG][Dataset::__getitem__] view extras: "
                    f"view_thresholds={extras['view_effective_crop_thresholds'].tolist()}, "
                    f"view_margins={extras['view_effective_crop_margin_ratios'].tolist()}, "
                    f"view_policy_indices={extras['view_crop_policy_indices'].tolist()}, "
                    f"view_area_ratios={extras['view_crop_area_ratios'].tolist()}, "
                    f"view_target_bbox_ratios={extras['view_target_bbox_ratios'].tolist()}, "
                    f"view_adaptive_scale_factors={extras['view_adaptive_scale_factors'].tolist()}, "
                    f"view_achieved_bbox_ratios={extras['view_achieved_bbox_ratios'].tolist()}, "
                    f"view_raw_bbox_area_ratios={extras['view_raw_bbox_area_ratios'].tolist()}"
                )

            if loader_has_image and self.validate_loader_alignment:
                print(f"[DEBUG][Dataset::__getitem__] loader_row_idx={loader_row_idx}")
                print(f"[DEBUG][Dataset::__getitem__] loader_label={loader_label}")

            print("-" * 80)

        return orig_x, second_branch_x, meta_vec, label_idx, extras
