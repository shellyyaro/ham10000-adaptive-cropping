from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AdaptiveCropResult:
    target_ratio: float
    bbox: tuple[int, int, int, int]
    mode: str
    target_possible: bool
    mask_area: int
    target_crop_area: float
    requested_crop_w: float
    requested_crop_h: float
    actual_crop_w: int
    actual_crop_h: int
    achieved_ratio: float
    distortion_log_abs: float
    resize_scale_x: float
    resize_scale_y: float


def _as_mask_2d(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    # The fused masks are already normalized by FusedMaskLoader. Avoid
    # np.clip here because it allocates another full-size float32 array for
    # every adaptive crop view, which can trigger memory pressure during
    # DataLoader iteration.
    return mask


def compute_binary_mask_stats(
    mask: np.ndarray,
    *,
    threshold: float = 0.60,
    min_area: int = 1,
) -> dict:
    mask = _as_mask_2d(mask)
    binary = mask >= float(threshold)
    area = int(np.count_nonzero(binary))
    if area < int(min_area):
        raise ValueError(
            f"Mask area below min_area at threshold={threshold}: "
            f"{area} < {int(min_area)}"
        )

    # Avoid np.where(binary), which materializes all foreground coordinates
    # as int64 arrays. During multi-view training this can fail under memory
    # pressure even for moderate image sizes. Axis projections are enough for
    # area, bbox, and a stable crop center.
    ys = np.flatnonzero(binary.any(axis=1))
    xs = np.flatnonzero(binary.any(axis=0))

    x1, x2 = int(xs[0]), int(xs[-1]) + 1
    y1, y2 = int(ys[0]), int(ys[-1]) + 1

    return {
        "binary": binary,
        "mask_area": area,
        "bbox": (x1, y1, x2, y2),
        "bbox_w": x2 - x1,
        "bbox_h": y2 - y1,
        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
    }


def place_crop(
    center_x: float,
    center_y: float,
    crop_w: float,
    crop_h: float,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    crop_w_i = int(math.ceil(float(crop_w)))
    crop_h_i = int(math.ceil(float(crop_h)))
    crop_w_i = max(1, min(crop_w_i, int(image_w)))
    crop_h_i = max(1, min(crop_h_i, int(image_h)))

    left = int(round(float(center_x) - crop_w_i / 2.0))
    top = int(round(float(center_y) - crop_h_i / 2.0))
    right = left + crop_w_i
    bottom = top + crop_h_i

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_w:
        left -= right - image_w
        right = image_w
    if bottom > image_h:
        top -= bottom - image_h
        bottom = image_h

    left = max(0, left)
    top = max(0, top)
    right = min(int(image_w), right)
    bottom = min(int(image_h), bottom)
    return (left, top, right, bottom)


def bbox_inside(
    crop_bbox: tuple[int, int, int, int],
    lesion_bbox: tuple[int, int, int, int],
) -> bool:
    cx1, cy1, cx2, cy2 = crop_bbox
    lx1, ly1, lx2, ly2 = lesion_bbox
    return cx1 <= lx1 and cy1 <= ly1 and cx2 >= lx2 and cy2 >= ly2


def _expand_to_contain_bbox(
    crop_bbox: tuple[int, int, int, int],
    lesion_bbox: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = crop_bbox
    lx1, ly1, lx2, ly2 = lesion_bbox

    x1 = min(x1, lx1)
    y1 = min(y1, ly1)
    x2 = max(x2, lx2)
    y2 = max(y2, ly2)

    crop_w = min(int(image_w), x2 - x1)
    crop_h = min(int(image_h), y2 - y1)
    cx = (max(x1, lx1) + min(x2, lx2)) / 2.0
    cy = (max(y1, ly1) + min(y2, ly2)) / 2.0
    return place_crop(cx, cy, crop_w, crop_h, image_w, image_h)


def adaptive_crop_for_target_ratio(
    mask: np.ndarray,
    *,
    target_ratio: float,
    image_size: tuple[int, int] | None = None,
    mask_threshold: float = 0.60,
    final_size: int = 224,
    min_mask_area: int = 1,
) -> AdaptiveCropResult:
    if not (0.0 < float(target_ratio) <= 1.0):
        raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")

    mask = _as_mask_2d(mask)
    mask_h, mask_w = mask.shape
    if image_size is None:
        image_w, image_h = mask_w, mask_h
    else:
        image_w, image_h = int(image_size[0]), int(image_size[1])
        if (image_w, image_h) != (mask_w, mask_h):
            raise ValueError(
                "mask shape must match image_size. "
                f"mask={(mask_w, mask_h)}, image_size={(image_w, image_h)}"
            )

    stats = compute_binary_mask_stats(
        mask,
        threshold=mask_threshold,
        min_area=min_mask_area,
    )

    mask_area = float(stats["mask_area"])
    bbox_w = float(stats["bbox_w"])
    bbox_h = float(stats["bbox_h"])
    lesion_bbox = stats["bbox"]
    center_x, center_y = stats["center"]

    target_ratio = float(target_ratio)
    target_crop_area = mask_area / target_ratio
    square_side = math.sqrt(target_crop_area)

    candidates: list[dict] = []

    if square_side >= bbox_w and square_side >= bbox_h:
        candidates.append(
            {
                "mode": "square_exact",
                "crop_w": square_side,
                "crop_h": square_side,
                "target_possible": True,
            }
        )

    if target_crop_area >= bbox_w * bbox_h:
        crop_h = bbox_h
        crop_w = target_crop_area / crop_h
        if crop_w >= bbox_w:
            candidates.append(
                {
                    "mode": "rect_height_forced",
                    "crop_w": crop_w,
                    "crop_h": crop_h,
                    "target_possible": True,
                }
            )

        crop_w = bbox_w
        crop_h = target_crop_area / crop_w
        if crop_h >= bbox_h:
            candidates.append(
                {
                    "mode": "rect_width_forced",
                    "crop_w": crop_w,
                    "crop_h": crop_h,
                    "target_possible": True,
                }
            )

    if not candidates:
        candidates.append(
            {
                "mode": "fallback_bbox_impossible_target",
                "crop_w": bbox_w,
                "crop_h": bbox_h,
                "target_possible": False,
            }
        )

    def distortion(candidate: dict) -> float:
        w = max(float(candidate["crop_w"]), 1e-6)
        h = max(float(candidate["crop_h"]), 1e-6)
        return abs(math.log(w / h))

    candidates.sort(key=distortion)

    chosen = None
    for candidate in candidates:
        crop_bbox = place_crop(
            center_x,
            center_y,
            float(candidate["crop_w"]),
            float(candidate["crop_h"]),
            image_w,
            image_h,
        )
        if not bbox_inside(crop_bbox, lesion_bbox):
            crop_bbox = _expand_to_contain_bbox(
                crop_bbox,
                lesion_bbox,
                image_w,
                image_h,
            )
        if bbox_inside(crop_bbox, lesion_bbox):
            chosen = dict(candidate)
            chosen["bbox"] = crop_bbox
            break

    if chosen is None:
        chosen = {
            "mode": "fallback_full_image_boundary",
            "crop_w": float(image_w),
            "crop_h": float(image_h),
            "target_possible": False,
            "bbox": (0, 0, int(image_w), int(image_h)),
        }

    x1, y1, x2, y2 = chosen["bbox"]
    actual_w = int(x2 - x1)
    actual_h = int(y2 - y1)
    actual_area = float(actual_w * actual_h)
    achieved_ratio = mask_area / max(1.0, float(actual_w * actual_h))
    target_possible = bool(chosen["target_possible"]) and (
        actual_area + 1e-6 >= float(target_crop_area)
    )

    return AdaptiveCropResult(
        target_ratio=target_ratio,
        bbox=chosen["bbox"],
        mode=str(chosen["mode"]),
        target_possible=target_possible,
        mask_area=int(mask_area),
        target_crop_area=float(target_crop_area),
        requested_crop_w=float(chosen["crop_w"]),
        requested_crop_h=float(chosen["crop_h"]),
        actual_crop_w=actual_w,
        actual_crop_h=actual_h,
        achieved_ratio=float(achieved_ratio),
        distortion_log_abs=abs(math.log(max(actual_w, 1) / max(actual_h, 1))),
        resize_scale_x=float(final_size) / max(1, actual_w),
        resize_scale_y=float(final_size) / max(1, actual_h),
    )


def adaptive_crops_for_target_ratios(
    mask: np.ndarray,
    target_ratios: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    *,
    image_size: tuple[int, int] | None = None,
    mask_threshold: float = 0.60,
    final_size: int = 224,
    min_mask_area: int = 1,
) -> list[AdaptiveCropResult]:
    return [
        adaptive_crop_for_target_ratio(
            mask,
            target_ratio=float(r),
            image_size=image_size,
            mask_threshold=mask_threshold,
            final_size=final_size,
            min_mask_area=min_mask_area,
        )
        for r in target_ratios
    ]


def crop_array_by_bbox(
    arr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.asarray(arr)[int(y1):int(y2), int(x1):int(x2)].copy()


def resize_crop_to_square(
    crop: np.ndarray,
    *,
    final_size: int = 224,
    is_mask: bool = False,
) -> np.ndarray:
    crop = np.asarray(crop)
    resample = Image.NEAREST if is_mask else Image.BILINEAR

    if is_mask:
        mask = _as_mask_2d(crop)
        img = Image.fromarray((mask * 255.0).astype(np.uint8))
        out = img.resize((int(final_size), int(final_size)), resample=resample)
        return np.asarray(out).astype(np.float32) / 255.0

    img = Image.fromarray(crop)
    out = img.resize((int(final_size), int(final_size)), resample=resample)
    return np.asarray(out)


def crop_and_resize_image(
    image: np.ndarray,
    crop_result: AdaptiveCropResult,
    *,
    final_size: int = 224,
) -> np.ndarray:
    crop = crop_array_by_bbox(image, crop_result.bbox)
    return resize_crop_to_square(crop, final_size=final_size, is_mask=False)


def crop_and_resize_many(
    image: np.ndarray,
    crop_results: Iterable[AdaptiveCropResult],
    *,
    final_size: int = 224,
) -> list[np.ndarray]:
    return [
        crop_and_resize_image(image, result, final_size=final_size)
        for result in crop_results
    ]
