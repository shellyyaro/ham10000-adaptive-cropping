##resnet_vit_pretrained_comparison/data/mask_fusion_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def _normalize_mask_array(mask: np.ndarray) -> np.ndarray:
    """
    Normalize a single mask into a 2D float32 array in [0,1].

    Supports:
    - bool
    - uint8 / int
    - float
    - shapes: (H,W), (1,H,W), (H,W,1)
    """
    if mask is None:
        return None

    mask = np.asarray(mask)

    if mask.size == 0:
        return None

    mask = np.squeeze(mask)

    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            # fallback: average channels
            mask = mask.mean(axis=-1)

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeeze/reduction, got shape={mask.shape}")

    if mask.dtype == np.bool_:
        mask = mask.astype(np.float32)
    else:
        mask = mask.astype(np.float32, copy=False)

    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)

    vmax = float(mask.max()) if mask.size > 0 else 0.0
    if vmax <= 0.0:
        return np.zeros_like(mask, dtype=np.float32)

    if vmax > 1.0:
        mask = mask / vmax

    mask = np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)
    return mask


def _safe_preview_array(arr: np.ndarray, max_items: int = 8) -> str:
    arr = np.asarray(arr)
    if arr.size == 0:
        return "[]"
    flat = arr.reshape(-1)
    preview = flat[:max_items]
    return np.array2string(preview, precision=4, separator=", ")


def _print_basic_array_stats(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    print(
        f"[DEBUG] {name}: "
        f"shape={arr.shape}, dtype={arr.dtype}, size={arr.size}"
    )


def _print_numeric_stats(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)

    if arr.size == 0:
        print(f"[DEBUG] {name}: empty array")
        return

    arr_num = arr.astype(np.float32, copy=False)
    arr_num = np.nan_to_num(arr_num, nan=0.0, posinf=1.0, neginf=0.0)

    print(
        f"[DEBUG] {name}: "
        f"min={float(arr_num.min()):.6f}, "
        f"max={float(arr_num.max()):.6f}, "
        f"mean={float(arr_num.mean()):.6f}, "
        f"std={float(arr_num.std()):.6f}"
    )


def _print_unique_stats(name: str, arr: np.ndarray, max_unique_to_show: int = 20) -> None:
    arr = np.asarray(arr)

    if arr.size == 0:
        print(f"[DEBUG] {name}: empty array")
        return

    try:
        unique_vals = np.unique(arr)
        shown = unique_vals[:max_unique_to_show]
        print(
            f"[DEBUG] {name}: unique_count={len(unique_vals)}, "
            f"preview={np.array2string(shown, separator=', ')}"
        )
    except Exception as e:
        print(f"[DEBUG] {name}: failed to compute unique values: {e}")


class FusedMaskLoader:
    """
    Loads pre-computed fused masks from NPZ files.

    Required fields:
        - masks
        - image_ids
        - labels
        - row_idx

    Optional fields:
        - has_majority_flag
        - mean_teacher_confidence
        - any future per-image arrays aligned with image_ids

    Behavior:
    - Works with old NPZ files that only contain the required fields
    - If optional fields exist, they are loaded and exposed via helper methods
    - Unknown extra arrays are kept in `extra_fields`
    """

    REQUIRED_KEYS = {"masks", "image_ids", "labels", "row_idx"}
    KNOWN_OPTIONAL_KEYS = {
        "has_majority_flag",
        "mean_teacher_confidence",
    }

    def __init__(
        self,
        masks: np.ndarray,
        image_ids: np.ndarray,
        labels: np.ndarray,
        row_idx: np.ndarray,
        *,
        has_majority_flag: Optional[np.ndarray] = None,
        mean_teacher_confidence: Optional[np.ndarray] = None,
        extra_fields: Optional[dict[str, np.ndarray]] = None,
        debug: bool = False,
        source_path: Optional[str | Path] = None,
    ):
        self.masks = masks
        self.image_ids = image_ids
        self.labels = labels
        self.row_idx = row_idx

        self.has_majority_flag = has_majority_flag
        self.mean_teacher_confidence = mean_teacher_confidence
        self.extra_fields = extra_fields or {}

        self.debug = bool(debug)
        self.source_path = str(source_path) if source_path is not None else None

        self._id2idx = {str(img_id): i for i, img_id in enumerate(image_ids)}

        n = len(self.image_ids)

        if len(self.masks) != n:
            raise ValueError(
                f"Inconsistent sizes: masks={len(self.masks)}, image_ids={n}"
            )
        if len(self.labels) != n:
            raise ValueError(
                f"Inconsistent sizes: labels={len(self.labels)}, image_ids={n}"
            )
        if len(self.row_idx) != n:
            raise ValueError(
                f"Inconsistent sizes: row_idx={len(self.row_idx)}, image_ids={n}"
            )

        if self.has_majority_flag is not None and len(self.has_majority_flag) != n:
            raise ValueError(
                f"Inconsistent sizes: has_majority_flag={len(self.has_majority_flag)}, image_ids={n}"
            )

        if self.mean_teacher_confidence is not None and len(self.mean_teacher_confidence) != n:
            raise ValueError(
                f"Inconsistent sizes: mean_teacher_confidence={len(self.mean_teacher_confidence)}, image_ids={n}"
            )

        for key, arr in self.extra_fields.items():
            if len(arr) != n:
                raise ValueError(
                    f"Inconsistent sizes: extra field '{key}' has len={len(arr)}, image_ids={n}"
                )

        if self.debug:
            self._print_loader_summary()

    @classmethod
    def from_npz(
        cls,
        npz_path: str | Path,
        *,
        debug: bool = False,
        strict_optional_fields: bool = False,
        load_extra_fields: bool = True,
    ) -> "FusedMaskLoader":
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        with np.load(npz_path, allow_pickle=True) as data:
            keys = set(data.keys())

            if debug:
                print("\n" + "=" * 80)
                print(f"[DEBUG] Loading fused mask NPZ: {npz_path}")
                print(f"[DEBUG] Available keys: {list(data.keys())}")
                print("=" * 80)

            missing = cls.REQUIRED_KEYS - keys
            if missing:
                raise KeyError(
                    f"NPZ file is missing required keys: {missing}. "
                    f"Available keys: {list(data.keys())}"
                )

            if strict_optional_fields:
                required_optional = {"has_majority_flag", "mean_teacher_confidence"}
                missing_optional = required_optional - keys
                if missing_optional:
                    raise KeyError(
                        f"NPZ file is missing expected optional keys in strict mode: "
                        f"{missing_optional}. Available keys: {list(data.keys())}"
                    )

            masks = data["masks"]
            image_ids = data["image_ids"]
            labels = data["labels"]
            row_idx = data["row_idx"]

            has_majority_flag = data["has_majority_flag"] if "has_majority_flag" in data else None
            mean_teacher_confidence = (
                data["mean_teacher_confidence"] if "mean_teacher_confidence" in data else None
            )

            known_keys = cls.REQUIRED_KEYS | cls.KNOWN_OPTIONAL_KEYS
            if load_extra_fields:
                extra_fields = {
                    key: data[key]
                    for key in data.keys()
                    if key not in known_keys
                }
            else:
                extra_fields = {}

            if debug:
                _print_basic_array_stats("masks", masks)
                _print_basic_array_stats("image_ids", image_ids)
                _print_basic_array_stats("labels", labels)
                _print_basic_array_stats("row_idx", row_idx)

                try:
                    print(f"[DEBUG] image_ids preview: {_safe_preview_array(image_ids)}")
                except Exception as e:
                    print(f"[DEBUG] Failed to preview image_ids: {e}")

                try:
                    print(f"[DEBUG] labels preview: {_safe_preview_array(labels)}")
                except Exception as e:
                    print(f"[DEBUG] Failed to preview labels: {e}")

                try:
                    print(f"[DEBUG] row_idx preview: {_safe_preview_array(row_idx)}")
                except Exception as e:
                    print(f"[DEBUG] Failed to preview row_idx: {e}")

                if masks is not None and len(masks) > 0:
                    try:
                        sample_mask = _normalize_mask_array(masks[0])
                        if sample_mask is None:
                            print("[DEBUG] sample normalized mask[0]: None")
                        else:
                            print(
                                "[DEBUG] sample normalized mask[0]: "
                                f"shape={sample_mask.shape}, "
                                f"dtype={sample_mask.dtype}, "
                                f"min={float(sample_mask.min()):.6f}, "
                                f"max={float(sample_mask.max()):.6f}, "
                                f"mean={float(sample_mask.mean()):.6f}"
                            )
                    except Exception as e:
                        print(f"[DEBUG] Failed to inspect sample mask[0]: {e}")

                if has_majority_flag is not None:
                    _print_basic_array_stats("has_majority_flag", has_majority_flag)
                    _print_unique_stats("has_majority_flag", has_majority_flag)
                    _print_numeric_stats("has_majority_flag", has_majority_flag)
                else:
                    print("[DEBUG] has_majority_flag is MISSING")

                if mean_teacher_confidence is not None:
                    _print_basic_array_stats("mean_teacher_confidence", mean_teacher_confidence)
                    _print_numeric_stats("mean_teacher_confidence", mean_teacher_confidence)
                    print(
                        "[DEBUG] mean_teacher_confidence preview: "
                        f"{_safe_preview_array(mean_teacher_confidence)}"
                    )
                else:
                    print("[DEBUG] mean_teacher_confidence is MISSING")

                if extra_fields:
                    print(f"[DEBUG] extra optional fields found: {sorted(extra_fields.keys())}")
                    for key, arr in extra_fields.items():
                        _print_basic_array_stats(f"extra_fields['{key}']", arr)
                else:
                    print("[DEBUG] No extra optional fields found")

        return cls(
            masks=masks,
            image_ids=image_ids,
            labels=labels,
            row_idx=row_idx,
            has_majority_flag=has_majority_flag,
            mean_teacher_confidence=mean_teacher_confidence,
            extra_fields=extra_fields,
            debug=debug,
            source_path=npz_path,
        )

    def _print_loader_summary(self) -> None:
        print("\n" + "-" * 80)
        print("[DEBUG] FusedMaskLoader summary")
        if self.source_path is not None:
            print(f"[DEBUG] source_path: {self.source_path}")
        print(f"[DEBUG] n_samples: {len(self)}")
        print(f"[DEBUG] masks shape: {getattr(self.masks, 'shape', 'unknown')}")
        print(f"[DEBUG] masks dtype: {getattr(self.masks, 'dtype', 'unknown')}")
        print(f"[DEBUG] optional_fields: {self.get_available_optional_fields()}")
        print("-" * 80)

        sample_n = min(3, len(self))
        for i in range(sample_n):
            img_id = str(self.image_ids[i])
            try:
                mask = self.get_mask(img_id)
                mask_mean = None if mask is None else float(mask.mean())
                has_majority = self.get_has_majority_flag(img_id, default=-999)
                teacher_conf = self.get_mean_teacher_confidence(img_id, default=-999.0)

                print(
                    f"[DEBUG] sample[{i}] "
                    f"image_id={img_id}, "
                    f"label={self.labels[i]}, "
                    f"row_idx={self.row_idx[i]}, "
                    f"mask_mean={mask_mean}, "
                    f"has_majority_flag={has_majority}, "
                    f"mean_teacher_confidence={teacher_conf}"
                )
            except Exception as e:
                print(f"[DEBUG] Failed to inspect sample[{i}] ({img_id}): {e}")

        print("-" * 80 + "\n")

    def _get_idx(self, image_id: str) -> Optional[int]:
        return self._id2idx.get(str(image_id))

    def get_mask(self, image_id: str) -> Optional[np.ndarray]:
        idx = self._get_idx(image_id)
        if idx is None:
            if self.debug:
                print(f"[DEBUG] get_mask: image_id='{image_id}' not found")
            return None

        mask = self.masks[idx]
        norm_mask = _normalize_mask_array(mask)

        if self.debug and idx < 3:
            if norm_mask is None:
                print(f"[DEBUG] get_mask: image_id='{image_id}', normalized mask=None")
            else:
                print(
                    f"[DEBUG] get_mask: image_id='{image_id}', "
                    f"shape={norm_mask.shape}, "
                    f"min={float(norm_mask.min()):.6f}, "
                    f"max={float(norm_mask.max()):.6f}, "
                    f"mean={float(norm_mask.mean()):.6f}"
                )

        return norm_mask

    def has_mask(self, image_id: str) -> bool:
        return str(image_id) in self._id2idx

    def get_label(self, image_id: str) -> Optional[Any]:
        idx = self._get_idx(image_id)
        if idx is None:
            if self.debug:
                print(f"[DEBUG] get_label: image_id='{image_id}' not found")
            return None
        return self.labels[idx]

    def get_row_idx(self, image_id: str) -> Optional[int]:
        idx = self._get_idx(image_id)
        if idx is None:
            if self.debug:
                print(f"[DEBUG] get_row_idx: image_id='{image_id}' not found")
            return None
        return int(self.row_idx[idx])

    def get_has_majority_flag(self, image_id: str, default: int = 0) -> int:
        """
        Returns:
            0 or 1 if field exists and image exists
            default if field missing or image missing
        """
        idx = self._get_idx(image_id)
        if idx is None:
            if self.debug:
                print(f"[DEBUG] get_has_majority_flag: image_id='{image_id}' not found, default={default}")
            return int(default)

        if self.has_majority_flag is None:
            if self.debug:
                print(f"[DEBUG] get_has_majority_flag: field missing, image_id='{image_id}', default={default}")
            return int(default)

        value = int(self.has_majority_flag[idx])

        if self.debug and idx < 3:
            print(f"[DEBUG] get_has_majority_flag: image_id='{image_id}', value={value}")

        return value

    def get_mean_teacher_confidence(self, image_id: str, default: float = 0.0) -> float:
        """
        Returns:
            mean teacher confidence if field exists and image exists
            default if field missing or image missing
        """
        idx = self._get_idx(image_id)
        if idx is None:
            if self.debug:
                print(
                    f"[DEBUG] get_mean_teacher_confidence: image_id='{image_id}' not found, default={default}"
                )
            return float(default)

        if self.mean_teacher_confidence is None:
            if self.debug:
                print(
                    f"[DEBUG] get_mean_teacher_confidence: field missing, image_id='{image_id}', default={default}"
                )
            return float(default)

        value = float(self.mean_teacher_confidence[idx])

        if self.debug and idx < 3:
            print(f"[DEBUG] get_mean_teacher_confidence: image_id='{image_id}', value={value:.6f}")

        return value

    def has_field(self, field_name: str) -> bool:
        if field_name in {"has_majority_flag", "mean_teacher_confidence"}:
            return getattr(self, field_name) is not None
        return field_name in self.extra_fields

    def get_field(self, image_id: str, field_name: str, default=None):
        """
        Generic accessor for optional per-image fields.
        """
        idx = self._get_idx(image_id)
        if idx is None:
            if self.debug:
                print(f"[DEBUG] get_field('{field_name}'): image_id='{image_id}' not found")
            return default

        if field_name in {"has_majority_flag", "mean_teacher_confidence"}:
            arr = getattr(self, field_name)
            if arr is None:
                if self.debug:
                    print(f"[DEBUG] get_field('{field_name}'): field missing, default={default}")
                return default
            return arr[idx]

        arr = self.extra_fields.get(field_name)
        if arr is None:
            if self.debug:
                print(f"[DEBUG] get_field('{field_name}'): field missing in extra_fields, default={default}")
            return default
        return arr[idx]

    def get_available_optional_fields(self) -> list[str]:
        fields = []
        if self.has_majority_flag is not None:
            fields.append("has_majority_flag")
        if self.mean_teacher_confidence is not None:
            fields.append("mean_teacher_confidence")
        fields.extend(sorted(self.extra_fields.keys()))
        return fields

    def get_all_image_ids(self) -> list[str]:
        return [str(img_id) for img_id in self.image_ids]

    def __len__(self) -> int:
        return len(self.masks)

    def __repr__(self) -> str:
        optional_fields = self.get_available_optional_fields()
        return (
            f"FusedMaskLoader(n={len(self)}, "
            f"shape={getattr(self.masks, 'shape', 'unknown')}, "
            f"dtype={getattr(self.masks, 'dtype', 'unknown')}, "
            f"optional_fields={optional_fields}, "
            f"source_path={self.source_path})"
        )
