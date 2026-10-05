from __future__ import annotations
"""
generate_pred_label_fusion_masks_combined_resnet_sources.py

Creates predicted-label Grad-CAM fusion masks for HAM10000 using THREE ResNet
source roots:

1) regular baseline ablation rerun
2) mid-level fusion rerun
3) filter-size-7 configs rerun

Main behavior:
  - Loads one best checkpoint per configuration folder from each of the three roots.
  - For every image, each model predicts its own label.
  - Computes Grad-CAM for each model's own predicted label.
  - Finds dynamic strict majority: floor(num_models / 2) + 1.
  - If majority exists: weights CAMs by p_majority and boosts models that agree with the majority.
  - If no majority: uses mean of predicted-label CAMs.
  - Saves fused_masks_train.npz, fused_masks_val.npz, fused_masks_test.npz.
  - Saves fusion_config.json and model_sources.csv for verification.
  - Fails fast if not all three expected source labels are used.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import ast
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label as scipy_label, binary_dilation
from torchvision import transforms

# ============================================================
# Project imports
# ============================================================

from resnet_vit_pretrained_comparison.data.data_loader import load_ham10000_metadata
from resnet_vit_pretrained_comparison.data.meta_features import build_meta_features_leakage_free
from resnet_vit_pretrained_comparison.scripts.splits import make_or_load_splits

from resnet_vit_pretrained_comparison.models.models_cnn_resnet import (
    ResNetWithMeta,
    ResNetWithMetaLegacy,
)
from resnet_vit_pretrained_comparison.models.DenseNet import DenseNetWithMeta
from resnet_vit_pretrained_comparison.models.ConvNeXt import ConvNeXtWithMeta


# ============================================================
# Constants
# ============================================================

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Output folder for the new combined masks.
# New folder name is intentionally different from the old 2026-05-06 folder
# so you do not accidentally inspect stale masks.
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("HAM10000_MASK_ROOT", str(REPO_ROOT / "data" / "generated_masks"))
)

# Source 1: current source already used in this file.
REGULAR_BASELINE_ABLATION_RESNET_ROOT = Path(
    REPO_ROOT / "checkpoints" / "resnet_ensemble" / "regular_baseline"
)

# Source 2: mid-level fusion rerun.
MIDLEVEL_FUSION_RESNET_ROOT = Path(
    REPO_ROOT / "checkpoints" / "resnet_ensemble" / "midlevel_fusion"
)

# Source 3: filter size 7 configs rerun.
FILTER_SIZE_7_CONFIGS_RESNET_ROOT = Path(
    REPO_ROOT / "checkpoints" / "resnet_ensemble" / "filter_size"
)

# Keep this True so the script uses EXACTLY the three sources in DEFAULT_MODEL_SOURCES.
# This is safer than passing roots through the CLI, because it prevents accidentally
# running only one source due to missing/ignored arguments.
FORCE_ONLY_DEFAULT_MODEL_SOURCES = True

PREFERRED_CHECKPOINT_NAMES = {
    "best_model.pt",
    "best.pt",
    "model_best.pt",
    "best_checkpoint.pt",
    "checkpoint_best.pt",
    "best_balanced_accuracy.pt",
    "best_macro_f1.pt",
}

MODEL_DEFAULTS = {
    "resnet": {
        "arch": "resnet18",
        "meta_hidden_dim": 128,
        "clf_hidden_dim": 256,
        "dropout": 0.3,
        "pretrained": True,
        "img_emb_dim": 512,
        "use_meta_bn": True,

        # stem / pooling
        "stem_kernel_size": 7,
        "stem_stride": 2,
        "stem_use_maxpool": True,
        "pool_type": "avg",
        "gem_p": 3.0,
        "gem_trainable": False,

        # multiscale
        "use_multiscale_stem": False,
        "stem_branch_kernels": [3, 3],
        "stem_branch_dilations": [1, 2],
        "use_multiscale_block": False,
        "multiscale_block_position": "none",
        "multiscale_kernels": [3, 3],
        "multiscale_dilations": [1, 2],

        # mid-level fusion
        "use_midlevel_fusion": False,
        "midlevel_source": "layer2",
        "midlevel_pool_type": "avg",
        "midlevel_emb_dim": 128,
    },
    "densenet": {
        "arch": "densenet121",
        "meta_hidden_dim": 128,
        "clf_hidden_dim": 256,
        "dropout": 0.3,
        "pretrained": True,
        "img_emb_dim": 1024,
        "use_meta_bn": True,
    },
    "convnext": {
        "arch": "convnext_tiny",
        "meta_hidden_dim": 128,
        "clf_hidden_dim": 256,
        "dropout": 0.3,
        "pretrained": True,
        "img_emb_dim": 768,
        "use_meta_bn": True,
    },
}


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class SourceSpec:
    path: Path
    arch: str
    label: str


@dataclass
class HamDataBundle:
    meta: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    meta_features: np.ndarray
    meta_feature_dim: int
    label2idx: dict
    idx2label: dict
    eval_transform: transforms.Compose


DEFAULT_MODEL_SOURCES = [
    SourceSpec(
        path=REGULAR_BASELINE_ABLATION_RESNET_ROOT,
        arch="resnet",
        label="resnet_regular_baseline_ablation_rerun_full_config_v1",
    ),
    SourceSpec(
        path=MIDLEVEL_FUSION_RESNET_ROOT,
        arch="resnet",
        label="resnet_midlevel_fusion_rerun_v1",
    ),
    SourceSpec(
        path=FILTER_SIZE_7_CONFIGS_RESNET_ROOT,
        arch="resnet",
        label="resnet_filter_size_7_configs_rerun_v1",
    ),
]

EXPECTED_SOURCE_LABELS = {
    "resnet_regular_baseline_ablation_rerun_full_config_v1",
    "resnet_midlevel_fusion_rerun_v1",
    "resnet_filter_size_7_configs_rerun_v1",
}

EXPECTED_MODEL_COUNTS_BY_SOURCE = {
    "resnet_regular_baseline_ablation_rerun_full_config_v1": 9,
    "resnet_midlevel_fusion_rerun_v1": 8,
    "resnet_filter_size_7_configs_rerun_v1": 7,
}
EXPECTED_ENSEMBLE_SIZE = sum(EXPECTED_MODEL_COUNTS_BY_SOURCE.values())


# ============================================================
# Grad-CAM
# ============================================================

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        gout = grad_out[0] if isinstance(grad_out, (tuple, list)) else grad_out
        self._gradients = gout.detach()

    def __call__(
        self,
        image: torch.Tensor,
        meta_vecs: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        self.model.eval()

        logits = self.model(image, meta_vecs) if meta_vecs is not None else self.model(image)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()

        if self._activations is None or self._gradients is None:
            h, w = image.shape[2], image.shape[3]
            return np.ones((h, w), dtype=np.float32)

        if self._activations.ndim != 4 or self._gradients.ndim != 4:
            h, w = image.shape[2], image.shape[3]
            return np.ones((h, w), dtype=np.float32)

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        return normalize_01(cam)

    def remove_hooks(self):
        try:
            self._fwd_hook.remove()
        except Exception:
            pass
        try:
            self._bwd_hook.remove()
        except Exception:
            pass


# ============================================================
# Helpers
# ============================================================

def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.ones_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def denormalize(tensor: torch.Tensor, mean=MEAN, std=STD) -> np.ndarray:
    img = tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def get_last_conv_layer(module: nn.Module) -> nn.Module:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Could not find any Conv2d layer in the provided module")
    return last


def get_cnn_backbone_for_gradcam(model: nn.Module) -> nn.Module:
    candidate_names = [
        "img_backbone",
        "backbone",
        "convnext",
        "densenet",
        "cnn",
        "features",
        "model",
    ]

    for name in candidate_names:
        if hasattr(model, name):
            candidate = getattr(model, name)
            if isinstance(candidate, nn.Module):
                has_conv = any(isinstance(m, nn.Conv2d) for m in candidate.modules())
                if has_conv:
                    return candidate

    has_conv = any(isinstance(m, nn.Conv2d) for m in model.modules())
    if has_conv:
        return model

    raise RuntimeError(
        f"Could not find a CNN backbone with Conv2d layers for model of type {type(model).__name__}"
    )


def get_resnet_cam_target_layer(model: nn.Module, model_cfg: dict) -> tuple[nn.Module, str]:
    if hasattr(model, "img_backbone") and hasattr(model.img_backbone, "layer4"):
        return model.img_backbone.layer4, "img_backbone.layer4"

    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4, "backbone.layer4"

    if hasattr(model, "layer4"):
        return model.layer4, "layer4"

    try:
        backbone = get_cnn_backbone_for_gradcam(model)
        target_layer = get_last_conv_layer(backbone)
        return target_layer, f"{type(backbone).__name__}.last_conv"
    except Exception as e:
        raise RuntimeError(
            f"Could not find target layer for ResNet Grad-CAM. "
            f"Model type={type(model).__name__}, model_cfg_name={model_cfg.get('name', 'N/A')}, error={e}"
        )


def show_overlay(ax, rgb_img: np.ndarray, cam01: np.ndarray, max_alpha: float = 0.55):
    ax.imshow(rgb_img)
    alpha = np.clip(cam01, 0, 1) * max_alpha
    ax.imshow(cam01, cmap="jet", alpha=alpha)
    ax.axis("off")


def save_mask_png(mask01: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(mask01, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def pick_contour_threshold(mask01: np.ndarray, q: float = 0.80) -> float:
    thr = float(np.quantile(mask01, q))
    return max(0.35, min(thr, 0.85))


def keep_largest_region(mask01: np.ndarray, thr: float, dilation_radius: int = 15) -> np.ndarray:
    binary = (mask01 > thr).astype(np.uint8)
    labeled, num_features = scipy_label(binary)

    if num_features == 0:
        return mask01.copy()

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(sizes.argmax())

    binary_largest = (labeled == largest_label)

    if dilation_radius > 0:
        r = dilation_radius
        struct = np.ones((r * 2 + 1, r * 2 + 1), dtype=bool)
        binary_largest = binary_dilation(binary_largest, structure=struct)

    result = mask01.copy()
    result[~binary_largest] = 0.0
    return result


def parse_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not parse json: {path} | error={e}")
        return {}


def load_saved_run_config(pt_file: Path) -> dict:
    """
    Preference order:
      1) best_params_used.json
      2) config.json
      3) a JSON configuration dictionary

    Looks both next to the checkpoint and one/two levels above it.
    """
    candidates = []
    for parent in [pt_file.parent, pt_file.parent.parent, pt_file.parent.parent.parent]:
        candidates.extend([
            parent / "best_params_used.json",
            parent / "config.json",
        ])

    out = {
        "config_source_path": None,
        "config_source_type": None,
        "params": {},
        "extras": {},
    }

    seen = set()
    for p in candidates:
        p = Path(p)
        if p in seen:
            continue
        seen.add(p)

        if not p.exists():
            continue

        if p.suffix.lower() == ".json":
            payload = parse_json_file(p)
            if not payload:
                continue

            if "best_params" in payload:
                out["config_source_path"] = str(p)
                out["config_source_type"] = "best_params_used.json"
                out["params"] = payload.get("best_params", {}) or {}
                out["extras"] = payload.get("extras", {}) or {}
                return out

            if "config" in payload:
                out["config_source_path"] = str(p)
                out["config_source_type"] = "config.json"
                out["params"] = payload.get("config", {}) or {}
                out["extras"] = payload.get("extras", {}) or {}
                return out

            # Fallback: sometimes the config JSON itself is the params dict.
            out["config_source_path"] = str(p)
            out["config_source_type"] = p.name
            out["params"] = payload
            out["extras"] = {}
            return out

    return out


def safe_int(val, fallback: int) -> int:
    try:
        if val is None or val == "":
            return fallback
        return int(val)
    except Exception:
        return fallback


def safe_float(val, fallback: float) -> float:
    try:
        if val is None or val == "":
            return fallback
        return float(val)
    except Exception:
        return fallback


def safe_bool(val, fallback: bool) -> bool:
    try:
        if isinstance(val, bool):
            return val
        if val is None or val == "":
            return fallback
        s = str(val).strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        return fallback
    except Exception:
        return fallback


def safe_literal(val, fallback):
    try:
        if val is None or val == "":
            return fallback
        if isinstance(val, (list, tuple, dict)):
            return val
        return ast.literal_eval(str(val))
    except Exception:
        return fallback


def infer_uses_meta_from_folder_name(config_folder: str) -> Optional[bool]:
    cf = config_folder.lower()
    if "_nm1" in cf or "__nometa__" in cf or "nometa" in cf:
        return False
    if "_nm0" in cf or "__meta__" in cf:
        return True
    return None


def infer_resnet_variants_from_folder_name(config_folder: str) -> dict:
    cf = config_folder.lower()

    out = {
        "stem_kernel_size": None,
        "stem_stride": None,
        "stem_use_maxpool": None,
        "pool_type": None,
        "gem_p": None,
        "gem_trainable": None,

        "use_multiscale_stem": False,
        "stem_branch_kernels": None,
        "stem_branch_dilations": None,

        "use_multiscale_block": False,
        "multiscale_block_position": None,
        "multiscale_kernels": None,
        "multiscale_dilations": None,

        "use_midlevel_fusion": False,
        "midlevel_source": None,
        "midlevel_pool_type": None,
        "midlevel_emb_dim": None,
    }

    if "midfusion" in cf:
        out["use_midlevel_fusion"] = True

        m_source = re.search(r"midfusion[_\-]l([123])", cf)
        if m_source:
            out["midlevel_source"] = f"layer{m_source.group(1)}"

        m_pool = re.search(r"midfusion[_\-]l[123][_\-](avg|max|gem)", cf)
        if m_pool:
            out["midlevel_pool_type"] = m_pool.group(1)

        m_dim = re.search(r"midfusion[_\-]l[123][_\-](?:avg|max|gem)[_\-](\d+)", cf)
        if m_dim:
            out["midlevel_emb_dim"] = int(m_dim.group(1))

    if re.search(r"stem3x3", cf):
        out["stem_kernel_size"] = 3
        out["stem_stride"] = 2
        out["stem_use_maxpool"] = True

    if re.search(r"stem5x5", cf):
        out["stem_kernel_size"] = 5
        out["stem_stride"] = 2
        out["stem_use_maxpool"] = True

    if "multiscalestem" in cf:
        out["use_multiscale_stem"] = True

        m_kernels = re.search(r"multiscalestem[_\-](\d)x\d[_\-](\d)x\d", cf)
        if m_kernels:
            k1 = int(m_kernels.group(1))
            k2 = int(m_kernels.group(2))
            out["stem_branch_kernels"] = [k1, k2]
            out["stem_branch_dilations"] = [1, 1]

        m_dil = re.search(r"multiscalestem[_\-](\d)x\d[_\-]dil(\d+)", cf)
        if m_dil:
            k = int(m_dil.group(1))
            d = int(m_dil.group(2))
            out["stem_branch_kernels"] = [k, k]
            out["stem_branch_dilations"] = [1, d]

    if "msblock" in cf:
        out["use_multiscale_block"] = True

        m_pos = re.search(r"msblock[_\-]afterlayer([1234])", cf)
        if m_pos:
            out["multiscale_block_position"] = f"after_layer{m_pos.group(1)}"

        m_block = re.search(r"msblock[_\-]afterlayer[1234][_\-](\d)x\d[_\-]dil(\d+)", cf)
        if m_block:
            k = int(m_block.group(1))
            d = int(m_block.group(2))
            out["multiscale_kernels"] = [k, k]
            out["multiscale_dilations"] = [1, d]

    if re.search(r"[_\-]gem([_\-]|$)", cf) or cf.endswith("gem"):
        out["pool_type"] = "gem"
    elif re.search(r"[_\-]avg([_\-]|$)", cf) or cf.endswith("avg"):
        out["pool_type"] = "avg"
    elif re.search(r"[_\-]max([_\-]|$)", cf) or cf.endswith("max"):
        out["pool_type"] = "max"

    return out


def maybe_empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def weighted_label_fusion(cams: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(cams) == 0:
        raise ValueError("cams is empty")
    if len(cams) != len(weights):
        raise ValueError("len(cams) != len(weights)")

    cams = [np.asarray(c, dtype=np.float32) for c in cams]
    weights_arr = np.asarray(weights, dtype=np.float32)

    w_sum = float(weights_arr.sum())
    if w_sum < 1e-12:
        fused = np.mean(np.stack(cams, axis=0), axis=0)
    else:
        w_norm = weights_arr / (w_sum + 1e-8)
        fused = np.zeros_like(cams[0], dtype=np.float32)
        for wi, ci in zip(w_norm, cams):
            fused += float(wi) * ci

    return normalize_01(fused)


def get_majority_info(pred_indices: list[int]) -> tuple[int, int, bool, int]:
    if len(pred_indices) == 0:
        raise ValueError("pred_indices is empty")

    counts = Counter(pred_indices)
    majority_idx, majority_count = counts.most_common(1)[0]
    n_models = len(pred_indices)
    required_votes = (n_models // 2) + 1
    has_majority = majority_count >= required_votes

    return int(majority_idx), int(majority_count), bool(has_majority), int(required_votes)


# ============================================================
# Data preparation
# ============================================================

def prepare_ham_data(split_seed: int = 0) -> HamDataBundle:
    meta, _ = load_ham10000_metadata()

    # Use exactly the same persisted preliminary split as the 24-model ResNet
    # training stage.  This file is one package level deeper than the other
    # entry points, so recomputing the repository root with parents[3] would
    # incorrectly point at ``src/`` and silently create a second split there.
    splits_dir = Path(
        os.environ.get(
            "SPLITS_ROOT",
            str(REPO_ROOT / "splits" / "mask_generation_fixed"),
        )
    )

    splits = make_or_load_splits(
        meta=meta,
        splits_dir=splits_dir,
        split_seed=split_seed,
        stratify_col="dx",
        overwrite=False,
        prefix="ham10000_mask_generation",
    )
    validate_fixed_split_manifest(splits)

    train_row_idxs = splits.train_df["row_idx"].values
    meta_features, _, _, meta_feature_dim = build_meta_features_leakage_free(meta, train_row_idxs)

    train_df = splits.train_df.reset_index(drop=True)
    val_df = splits.val_df.reset_index(drop=True)
    test_df = splits.test_df.reset_index(drop=True)

    labels = sorted(meta["dx"].unique())
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i: l for l, i in label2idx.items()}

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    return HamDataBundle(
        meta=meta,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        meta_features=meta_features,
        meta_feature_dim=meta_feature_dim,
        label2idx=label2idx,
        idx2label=idx2label,
        eval_transform=eval_transform,
    )


# ============================================================
# Source parsing
# ============================================================

def parse_source_arg(s: str) -> SourceSpec:
    parts = s.split("|")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --source format: {s}\n"
            f"Expected: path|arch|label"
        )

    path_str, arch, label = parts
    arch = arch.strip().lower()

    if arch not in {"resnet", "densenet", "convnext"}:
        raise ValueError(f"Unsupported arch '{arch}' in source '{s}'")

    return SourceSpec(path=Path(path_str.strip()), arch=arch, label=label.strip())


def build_sources_from_roots(
    *,
    resnet_roots: list[str] | None = None,
    densenet_roots: list[str] | None = None,
    convnext_roots: list[str] | None = None,
) -> list[SourceSpec]:
    sources: list[SourceSpec] = []

    for i, p in enumerate(resnet_roots or []):
        sources.append(SourceSpec(path=Path(p).expanduser(), arch="resnet", label=f"resnet_root_{i + 1}"))

    for i, p in enumerate(densenet_roots or []):
        sources.append(SourceSpec(path=Path(p).expanduser(), arch="densenet", label=f"densenet_root_{i + 1}"))

    for i, p in enumerate(convnext_roots or []):
        sources.append(SourceSpec(path=Path(p).expanduser(), arch="convnext", label=f"convnext_root_{i + 1}"))

    return sources


# ============================================================
# Checkpoint discovery
# ============================================================

def get_run_folder_for_checkpoint(pt_file: Path, source_root: Path) -> Path:
    """
    Returns the most likely run/config folder for a checkpoint.

    Handles common structures:
      config_folder/best_model.pt
      config_folder/checkpoints/best_model.pt
      config_folder/models/best_model.pt
    """
    parent = pt_file.parent
    if parent.name.lower() in {"checkpoints", "checkpoint", "models", "weights"}:
        return parent.parent
    return parent


def checkpoint_priority(pt_file: Path) -> tuple[int, int, float, str]:
    """
    Lower tuple is better.
    Prefer known best checkpoint names, then names containing best, then newest modified time.
    """
    name = pt_file.name.lower()
    if name in PREFERRED_CHECKPOINT_NAMES:
        group = 0
    elif "best" in name:
        group = 1
    elif "last" in name or "final" in name:
        group = 2
    else:
        group = 3

    try:
        mtime = pt_file.stat().st_mtime
    except Exception:
        mtime = 0.0

    # Use negative mtime because newer should come first.
    return group, len(name), -mtime, name


def select_one_checkpoint_per_config(source: SourceSpec) -> list[Path]:
    all_pt = sorted(source.path.rglob("*.pt"))
    grouped: dict[Path, list[Path]] = defaultdict(list)

    for pt in all_pt:
        # Avoid obvious non-model state files if they exist.
        lower_name = pt.name.lower()
        if any(x in lower_name for x in ["optimizer", "scheduler", "scaler"]):
            continue

        run_folder = get_run_folder_for_checkpoint(pt, source.path)
        grouped[run_folder].append(pt)

    selected = []
    for run_folder, ckpts in sorted(grouped.items(), key=lambda x: str(x[0]).lower()):
        best = sorted(ckpts, key=checkpoint_priority)[0]
        if len(ckpts) > 1:
            print(f"[discover][select] run={run_folder.name} selected={best.name} from {len(ckpts)} .pt files")
        selected.append(best)

    return selected


def discover_checkpoints_for_source(
    source: SourceSpec,
    max_models: Optional[int] = None,
) -> list[dict]:
    selected_pt = select_one_checkpoint_per_config(source)
    configs = []

    print(f"[discover] source={source.label} arch={source.arch}")
    print(f"[discover] root={source.path}")
    print(f"[discover] selected {len(selected_pt)} checkpoint files, one per config folder")

    defaults = MODEL_DEFAULTS[source.arch]

    for pt_file in selected_pt:
        saved_cfg = load_saved_run_config(pt_file)
        cfg_params = saved_cfg["params"]
        cfg_extras = saved_cfg["extras"]
        config_source_path = saved_cfg["config_source_path"]
        config_source_type = saved_cfg["config_source_type"]

        config_folder_path = get_run_folder_for_checkpoint(pt_file, source.path)
        config_folder = config_folder_path.name

        if "use_meta" in cfg_params:
            uses_meta = safe_bool(cfg_params.get("use_meta"), True)
        else:
            no_meta_val = None
            if "no_meta" in cfg_extras:
                no_meta_val = cfg_extras.get("no_meta")
            elif "no_meta" in cfg_params:
                no_meta_val = cfg_params.get("no_meta")

            if no_meta_val is not None:
                uses_meta = not safe_bool(no_meta_val, False)
            else:
                inferred = infer_uses_meta_from_folder_name(config_folder)
                uses_meta = True if inferred is None else inferred

        folder_infer = infer_resnet_variants_from_folder_name(config_folder)

        item = {
            "name": config_folder,
            "folder": config_folder,
            "run_tag": str(cfg_params.get("run_tag", config_folder)),
            "run_folder": str(config_folder_path),
            "path": pt_file,
            "source_label": source.label,
            "arch": source.arch,
            "config_source_path": config_source_path,
            "config_source_type": config_source_type,
            "uses_meta": uses_meta,
            "arch_name": cfg_params.get("arch", defaults["arch"]),
            "meta_hidden_dim": safe_int(cfg_params.get("meta_hidden_dim", ""), defaults["meta_hidden_dim"]),
            "clf_hidden_dim": safe_int(cfg_params.get("clf_hidden_dim", ""), defaults["clf_hidden_dim"]),
            "dropout": safe_float(cfg_params.get("dropout", ""), defaults["dropout"]),
            "pretrained": safe_bool(cfg_params.get("pretrained", defaults["pretrained"]), defaults["pretrained"]),
            "img_emb_dim": safe_int(cfg_params.get("img_emb_dim", ""), defaults["img_emb_dim"]),
            "use_meta_bn": safe_bool(cfg_params.get("use_meta_bn", defaults["use_meta_bn"]), defaults["use_meta_bn"]),

            "stem_kernel_size": safe_int(
                cfg_params.get("stem_kernel_size", ""),
                folder_infer["stem_kernel_size"] if folder_infer["stem_kernel_size"] is not None else defaults.get("stem_kernel_size", 7),
            ),
            "stem_stride": safe_int(
                cfg_params.get("stem_stride", ""),
                folder_infer["stem_stride"] if folder_infer["stem_stride"] is not None else defaults.get("stem_stride", 2),
            ),
            "stem_use_maxpool": safe_bool(
                cfg_params.get(
                    "stem_use_maxpool",
                    folder_infer["stem_use_maxpool"] if folder_infer["stem_use_maxpool"] is not None else defaults.get("stem_use_maxpool", True),
                ),
                folder_infer["stem_use_maxpool"] if folder_infer["stem_use_maxpool"] is not None else defaults.get("stem_use_maxpool", True),
            ),
            "pool_type": str(
                cfg_params.get(
                    "pool_type",
                    folder_infer["pool_type"] if folder_infer["pool_type"] is not None else defaults.get("pool_type", "avg"),
                )
            ).lower(),
            "gem_p": safe_float(cfg_params.get("gem_p", ""), defaults.get("gem_p", 3.0)),
            "gem_trainable": safe_bool(
                cfg_params.get("gem_trainable", defaults.get("gem_trainable", False)),
                defaults.get("gem_trainable", False),
            ),

            "use_multiscale_stem": safe_bool(
                cfg_params.get("use_multiscale_stem", folder_infer["use_multiscale_stem"]),
                folder_infer["use_multiscale_stem"] if folder_infer["use_multiscale_stem"] is not None else defaults.get("use_multiscale_stem", False),
            ),
            "stem_branch_kernels": safe_literal(
                cfg_params.get("stem_branch_kernels", ""),
                folder_infer["stem_branch_kernels"] if folder_infer["stem_branch_kernels"] is not None else defaults.get("stem_branch_kernels", [3, 3]),
            ),
            "stem_branch_dilations": safe_literal(
                cfg_params.get("stem_branch_dilations", ""),
                folder_infer["stem_branch_dilations"] if folder_infer["stem_branch_dilations"] is not None else defaults.get("stem_branch_dilations", [1, 2]),
            ),

            "use_multiscale_block": safe_bool(
                cfg_params.get("use_multiscale_block", folder_infer["use_multiscale_block"]),
                folder_infer["use_multiscale_block"] if folder_infer["use_multiscale_block"] is not None else defaults.get("use_multiscale_block", False),
            ),
            "multiscale_block_position": str(
                cfg_params.get(
                    "multiscale_block_position",
                    folder_infer["multiscale_block_position"] if folder_infer["multiscale_block_position"] is not None else defaults.get("multiscale_block_position", "none"),
                )
            ).lower(),
            "multiscale_kernels": safe_literal(
                cfg_params.get("multiscale_kernels", ""),
                folder_infer["multiscale_kernels"] if folder_infer["multiscale_kernels"] is not None else defaults.get("multiscale_kernels", [3, 3]),
            ),
            "multiscale_dilations": safe_literal(
                cfg_params.get("multiscale_dilations", ""),
                folder_infer["multiscale_dilations"] if folder_infer["multiscale_dilations"] is not None else defaults.get("multiscale_dilations", [1, 2]),
            ),

            "use_midlevel_fusion": safe_bool(
                cfg_params.get("use_midlevel_fusion", folder_infer["use_midlevel_fusion"]),
                folder_infer["use_midlevel_fusion"] if folder_infer["use_midlevel_fusion"] is not None else defaults.get("use_midlevel_fusion", False),
            ),
            "midlevel_source": str(
                cfg_params.get(
                    "midlevel_source",
                    folder_infer["midlevel_source"] if folder_infer["midlevel_source"] is not None else defaults.get("midlevel_source", "layer2"),
                )
            ).lower(),
            "midlevel_pool_type": str(
                cfg_params.get(
                    "midlevel_pool_type",
                    folder_infer["midlevel_pool_type"] if folder_infer["midlevel_pool_type"] is not None else defaults.get("midlevel_pool_type", "avg"),
                )
            ).lower(),
            "midlevel_emb_dim": safe_int(
                cfg_params.get("midlevel_emb_dim", ""),
                folder_infer["midlevel_emb_dim"] if folder_infer["midlevel_emb_dim"] is not None else defaults.get("midlevel_emb_dim", 128),
            ),
        }

        configs.append(item)

        if source.arch == "resnet":
            print(
                "[discover][resnet_infer]",
                {
                    "folder": config_folder,
                    "checkpoint": pt_file.name,
                    "config_source_type": item["config_source_type"],
                    "config_source_path": item["config_source_path"],
                    "uses_meta": item["uses_meta"],
                    "stem_kernel_size": item["stem_kernel_size"],
                    "pool_type": item["pool_type"],
                    "use_multiscale_stem": item["use_multiscale_stem"],
                    "stem_branch_kernels": item["stem_branch_kernels"],
                    "stem_branch_dilations": item["stem_branch_dilations"],
                    "use_multiscale_block": item["use_multiscale_block"],
                    "multiscale_block_position": item["multiscale_block_position"],
                    "multiscale_kernels": item["multiscale_kernels"],
                    "multiscale_dilations": item["multiscale_dilations"],
                    "use_midlevel_fusion": item["use_midlevel_fusion"],
                    "midlevel_source": item["midlevel_source"],
                    "midlevel_pool_type": item["midlevel_pool_type"],
                    "midlevel_emb_dim": item["midlevel_emb_dim"],
                }
            )

    if max_models is not None:
        configs = configs[:max_models]

    print(f"[discover] returning {len(configs)} checkpoints for {source.label}")
    return configs


# ============================================================
# Model loading
# ============================================================

def load_resnet_model(
    checkpoint_path: Path,
    num_classes: int,
    meta_feature_dim: int,
    device: torch.device,
    uses_meta: bool,
    arch_name: str = "resnet18",
    img_emb_dim: int = 512,
    meta_hidden_dim: int = 128,
    clf_hidden_dim: int = 256,
    dropout: float = 0.3,
    use_meta_bn: bool = True,
    pretrained: bool = True,

    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    stem_use_maxpool: bool = True,
    pool_type: str = "avg",
    gem_p: float = 3.0,
    gem_trainable: bool = False,

    use_multiscale_stem: bool = False,
    stem_branch_kernels=None,
    stem_branch_dilations=None,
    use_multiscale_block: bool = False,
    multiscale_block_position: str = "none",
    multiscale_kernels=None,
    multiscale_dilations=None,

    use_midlevel_fusion: bool = False,
    midlevel_source: str = "layer2",
    midlevel_pool_type: str = "avg",
    midlevel_emb_dim: int = 128,

    strict_load: bool = False,
) -> nn.Module:
    cfg = {
        "arch": arch_name,
        "pretrained": pretrained,
        "img_emb_dim": img_emb_dim,
        "meta_hidden_dim": meta_hidden_dim,
        "use_meta_bn": use_meta_bn,
        "clf_hidden_dim": clf_hidden_dim,
        "dropout": dropout,
        "use_meta": uses_meta,

        "stem_kernel_size": stem_kernel_size,
        "stem_stride": stem_stride,
        "stem_use_maxpool": stem_use_maxpool,
        "pool_type": pool_type,
        "gem_p": gem_p,
        "gem_trainable": gem_trainable,

        "use_multiscale_stem": use_multiscale_stem,
        "stem_branch_kernels": stem_branch_kernels if stem_branch_kernels is not None else [3, 3],
        "stem_branch_dilations": stem_branch_dilations if stem_branch_dilations is not None else [1, 2],
        "use_multiscale_block": use_multiscale_block,
        "multiscale_block_position": multiscale_block_position,
        "multiscale_kernels": multiscale_kernels if multiscale_kernels is not None else [3, 3],
        "multiscale_dilations": multiscale_dilations if multiscale_dilations is not None else [1, 2],

        "use_midlevel_fusion": use_midlevel_fusion,
        "midlevel_source": str(midlevel_source).lower(),
        "midlevel_pool_type": str(midlevel_pool_type).lower(),
        "midlevel_emb_dim": midlevel_emb_dim,
    }

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    fixed = {}
    for k, v in state.items():
        nk = k
        nk = nk.replace("module.", "")
        nk = nk.replace("img_backbone.features.", "img_backbone.")
        nk = nk.replace("meta_mlp.", "meta_net.")
        fixed[nk] = v

    is_legacy = "img_emb.0.weight" not in fixed

    if is_legacy:
        model = ResNetWithMetaLegacy(num_classes=num_classes, meta_dim=meta_feature_dim, cfg=cfg)
    else:
        model = ResNetWithMeta(num_classes=num_classes, meta_dim=meta_feature_dim, cfg=cfg)

    missing, unexpected = model.load_state_dict(fixed, strict=False)

    if missing:
        print(f"[WARN] Missing keys in {checkpoint_path.name}: {missing[:15]}")
    if unexpected:
        print(f"[WARN] Unexpected keys in {checkpoint_path.name}: {unexpected[:15]}")

    suspicious_keys = [
        k for k in list(missing) + list(unexpected)
        if any(token in k.lower() for token in ["mid", "multiscale", "stem", "img_emb", "classifier", "fc"])
    ]
    if suspicious_keys:
        print(f"[WARN] Possible architecture mismatch in {checkpoint_path.name}: {suspicious_keys[:20]}")

    if strict_load and (missing or unexpected):
        raise RuntimeError(
            f"Checkpoint/model mismatch in {checkpoint_path}\n"
            f"missing={missing[:50]}\n"
            f"unexpected={unexpected[:50]}\n"
            f"cfg={json.dumps(cfg, indent=2, default=str)}"
        )

    model.to(device).eval()
    return model


def load_densenet_model(
    checkpoint_path: Path,
    num_classes: int,
    meta_feature_dim: int,
    device: torch.device,
    uses_meta: bool,
    arch_name: str = "densenet121",
    img_emb_dim: int = 1024,
    meta_hidden_dim: int = 128,
    clf_hidden_dim: int = 256,
    dropout: float = 0.3,
    use_meta_bn: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    fixed = {}
    for k, v in state.items():
        nk = k
        nk = nk.replace("module.", "")
        nk = nk.replace("img_backbone.features.", "img_backbone.")
        nk = nk.replace("meta_mlp.", "meta_net.")
        fixed[nk] = v

    model = DenseNetWithMeta(
        num_classes=num_classes,
        arch=str(arch_name),
        pretrained=bool(pretrained),
        meta_feature_dim=int(meta_feature_dim) if uses_meta else 0,
        use_meta=bool(uses_meta),
        meta_hidden_dim=int(meta_hidden_dim),
        clf_hidden_dim=int(clf_hidden_dim),
        dropout=float(dropout),
        in_channels=3,
    )

    missing, unexpected = model.load_state_dict(fixed, strict=False)

    if missing:
        print(f"[WARN] Missing keys in {checkpoint_path.name}: {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys in {checkpoint_path.name}: {unexpected[:10]}")

    model.to(device).eval()
    return model


def load_convnext_model(
    checkpoint_path: Path,
    num_classes: int,
    meta_feature_dim: int,
    device: torch.device,
    uses_meta: bool,
    arch_name: str = "convnext_tiny",
    img_emb_dim: int = 768,
    meta_hidden_dim: int = 128,
    clf_hidden_dim: int = 256,
    dropout: float = 0.3,
    use_meta_bn: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    fixed = {}
    for k, v in state.items():
        nk = k
        nk = nk.replace("module.", "")
        nk = nk.replace("img_backbone.features.", "img_backbone.")
        nk = nk.replace("backbone.features.", "backbone.")
        nk = nk.replace("meta_mlp.", "meta_net.")
        fixed[nk] = v

    model = ConvNeXtWithMeta(
        num_classes=num_classes,
        arch=str(arch_name),
        pretrained=bool(pretrained),
        meta_feature_dim=int(meta_feature_dim) if uses_meta else 0,
        use_meta=bool(uses_meta),
        meta_hidden_dim=int(meta_hidden_dim),
        clf_hidden_dim=int(clf_hidden_dim),
        dropout=float(dropout),
        in_channels=3,
    )

    missing, unexpected = model.load_state_dict(fixed, strict=False)

    if missing:
        print(f"[WARN] Missing keys in {checkpoint_path.name}: {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys in {checkpoint_path.name}: {unexpected[:10]}")

    model.to(device).eval()
    return model


def load_model_from_config(
    *,
    config: dict,
    num_classes: int,
    meta_feature_dim: int,
    device: torch.device,
    strict_load: bool = False,
) -> nn.Module:
    arch = config["arch"].lower()

    if arch == "resnet":
        return load_resnet_model(
            checkpoint_path=config["path"],
            num_classes=num_classes,
            meta_feature_dim=meta_feature_dim,
            device=device,
            uses_meta=config["uses_meta"],
            arch_name=config["arch_name"],
            img_emb_dim=config["img_emb_dim"],
            meta_hidden_dim=config["meta_hidden_dim"],
            clf_hidden_dim=config["clf_hidden_dim"],
            dropout=config["dropout"],
            use_meta_bn=config["use_meta_bn"],
            pretrained=config["pretrained"],

            stem_kernel_size=config.get("stem_kernel_size", 7),
            stem_stride=config.get("stem_stride", 2),
            stem_use_maxpool=config.get("stem_use_maxpool", True),
            pool_type=config.get("pool_type", "avg"),
            gem_p=config.get("gem_p", 3.0),
            gem_trainable=config.get("gem_trainable", False),

            use_multiscale_stem=config.get("use_multiscale_stem", False),
            stem_branch_kernels=config.get("stem_branch_kernels", [3, 3]),
            stem_branch_dilations=config.get("stem_branch_dilations", [1, 2]),
            use_multiscale_block=config.get("use_multiscale_block", False),
            multiscale_block_position=config.get("multiscale_block_position", "none"),
            multiscale_kernels=config.get("multiscale_kernels", [3, 3]),
            multiscale_dilations=config.get("multiscale_dilations", [1, 2]),

            use_midlevel_fusion=config.get("use_midlevel_fusion", False),
            midlevel_source=config.get("midlevel_source", "layer2"),
            midlevel_pool_type=config.get("midlevel_pool_type", "avg"),
            midlevel_emb_dim=config.get("midlevel_emb_dim", 128),
            strict_load=strict_load,
        )

    if arch == "densenet":
        return load_densenet_model(
            checkpoint_path=config["path"],
            num_classes=num_classes,
            meta_feature_dim=meta_feature_dim,
            device=device,
            uses_meta=config["uses_meta"],
            arch_name=config["arch_name"],
            img_emb_dim=config["img_emb_dim"],
            meta_hidden_dim=config["meta_hidden_dim"],
            clf_hidden_dim=config["clf_hidden_dim"],
            dropout=config["dropout"],
            use_meta_bn=config["use_meta_bn"],
            pretrained=config["pretrained"],
        )

    if arch == "convnext":
        return load_convnext_model(
            checkpoint_path=config["path"],
            num_classes=num_classes,
            meta_feature_dim=meta_feature_dim,
            device=device,
            uses_meta=config["uses_meta"],
            arch_name=config["arch_name"],
            img_emb_dim=config["img_emb_dim"],
            meta_hidden_dim=config["meta_hidden_dim"],
            clf_hidden_dim=config["clf_hidden_dim"],
            dropout=config["dropout"],
            use_meta_bn=config["use_meta_bn"],
            pretrained=config["pretrained"],
        )

    raise ValueError(f"Unsupported arch: {arch}")


# ============================================================
# Build ensemble
# ============================================================

def build_models_and_cams(
    *,
    sources: list[SourceSpec],
    num_classes: int,
    meta_feature_dim: int,
    device: torch.device,
    max_models_per_source: Optional[int] = None,
    strict_load: bool = False,
) -> list[dict]:
    models_and_cams = []
    seen_ckpts = set()

    for source in sources:
        if not source.path.exists():
            print(f"[WARN] source path does not exist: {source.path}")
            continue

        configs = discover_checkpoints_for_source(source, max_models=max_models_per_source)

        for c in configs:
            ckpt_resolved = str(Path(c["path"]).resolve())
            if ckpt_resolved in seen_ckpts:
                print(f"[SKIP] duplicate checkpoint: {ckpt_resolved}")
                continue
            seen_ckpts.add(ckpt_resolved)

            print(
                f"[load] {c['source_label']} | {c['arch']} | {c['name']} | "
                f"uses_meta={c['uses_meta']} | cfg_source={c.get('config_source_type')}"
            )

            model = load_model_from_config(
                config=c,
                num_classes=num_classes,
                meta_feature_dim=meta_feature_dim,
                device=device,
                strict_load=strict_load,
            )

            if c["arch"] == "resnet":
                target_layer, target_name = get_resnet_cam_target_layer(model, c)
                backbone_name = "resnet_backbone"
            else:
                backbone_for_cam = get_cnn_backbone_for_gradcam(model)
                target_layer = get_last_conv_layer(backbone_for_cam)
                target_name = type(target_layer).__name__
                backbone_name = type(backbone_for_cam).__name__

            print(
                f"[gradcam] model={c['name']} "
                f"backbone={backbone_name} "
                f"target_name={target_name} "
                f"target_type={type(target_layer).__name__} "
                f"config_source_type={c.get('config_source_type')} "
                f"use_midlevel_fusion={c.get('use_midlevel_fusion')} "
                f"midlevel_source={c.get('midlevel_source')} "
                f"stem_kernel_size={c.get('stem_kernel_size')} "
                f"pool_type={c.get('pool_type')} "
                f"use_multiscale_stem={c.get('use_multiscale_stem')} "
                f"use_multiscale_block={c.get('use_multiscale_block')} "
                f"multiscale_block_position={c.get('multiscale_block_position')}"
            )

            cam = GradCAM(model, target_layer)

            models_and_cams.append({
                "name": c["name"],
                "folder": c["folder"],
                "ckpt": ckpt_resolved,
                "source_label": c["source_label"],
                "uses_meta": bool(c["uses_meta"]),
                "arch": c["arch"],
                "model_cfg": c,
                "target_layer_name": target_name,
                "model": model,
                "grad_cam": cam,
            })

    if not models_and_cams:
        raise RuntimeError("No models were loaded.")

    return models_and_cams


def cleanup_models_and_cams(models_and_cams: list[dict]):
    for mc in models_and_cams:
        try:
            mc["grad_cam"].remove_hooks()
        except Exception:
            pass


# ============================================================
# Majority-based fusion
# ============================================================

def compute_majority_fused_mask_for_row(
    *,
    row: pd.Series,
    models_and_cams: list[dict],
    meta_features: np.ndarray,
    label2idx: dict,
    idx2label: dict,
    device: torch.device,
    eval_transform,
    gamma: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-4,
    apply_quality_gate: bool = True,
    keep_largest: bool = True,
    contour_quantile: float = 0.80,
    dilation_radius: int = 15,
) -> dict:
    img_path = str(row["img_path"])
    image_id = Path(img_path).stem
    true_label = str(row["dx"])
    true_idx = int(label2idx[true_label])
    row_idx = int(row["row_idx"])

    pil_img = Image.open(img_path).convert("RGB")
    img_tensor = eval_transform(pil_img).unsqueeze(0).to(device)
    meta_vec = torch.from_numpy(
        np.asarray(meta_features[row_idx], dtype=np.float32)
    ).unsqueeze(0).to(device)
    original_rgb = denormalize(img_tensor.squeeze(0))

    # First pass: predictions/probabilities for each model.
    first_pass = []
    pred_indices = []

    for mc in models_and_cams:
        uses_meta = bool(mc["uses_meta"])
        model = mc["model"]

        with torch.no_grad():
            logits = model(img_tensor, meta_vec) if uses_meta else model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]
            probs_np = probs.detach().cpu().numpy().astype(np.float32)

            pred_idx = int(probs.argmax().item())
            pred_label = idx2label[pred_idx]
            conf_pred = float(probs[pred_idx].item())

        first_pass.append({
            "name": mc["name"],
            "arch": mc["arch"],
            "source_label": mc["source_label"],
            "uses_meta": uses_meta,
            "probs": probs_np,
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "confidence_pred": conf_pred,
        })
        pred_indices.append(pred_idx)

    majority_idx, majority_count, has_majority, required_majority_votes = get_majority_info(pred_indices)
    majority_label = idx2label[int(majority_idx)]
    has_majority_flag = np.uint8(1 if has_majority else 0)
    vote_ratio = float(majority_count / len(pred_indices))
    vote_hist = {
        idx2label[int(k)]: int(v)
        for k, v in sorted(Counter(pred_indices).items())
    }

    # Second pass: Grad-CAM for each model's own predicted label.
    per_model = []
    cams = []
    weights = []
    majority_probs_all = []
    pred_confidences = []

    for fp, mc in zip(first_pass, models_and_cams):
        uses_meta = bool(mc["uses_meta"])
        pred_idx = int(fp["pred_idx"])
        p_majority = float(fp["probs"][majority_idx])
        is_majority_match = bool(pred_idx == majority_idx)

        cam_pred = mc["grad_cam"](
            img_tensor,
            meta_vecs=(meta_vec if uses_meta else None),
            target_class=pred_idx,
        ).astype(np.float32)

        if has_majority:
            w = float((eps + p_majority) ** gamma)
            if is_majority_match:
                w *= float(1.0 + beta)
            majority_probs_all.append(p_majority)
        else:
            w = 1.0 / max(1, len(models_and_cams))
            pred_confidences.append(float(fp["confidence_pred"]))

        per_model.append({
            "name": mc["name"],
            "arch": mc["arch"],
            "source_label": mc["source_label"],
            "uses_meta": uses_meta,
            "pred_idx": fp["pred_idx"],
            "pred_label": fp["pred_label"],
            "confidence_pred": fp["confidence_pred"],
            "p_majority": p_majority,
            "is_majority_match": is_majority_match,
            "weight": float(w),
            "cam_pred": cam_pred,
            "cam_majority": cam_pred,  # Backward compatibility with old debug plotting code.
        })

        cams.append(cam_pred)
        weights.append(w)

    # Fusion of predicted-label CAMs.
    if has_majority:
        fused_raw = weighted_label_fusion(cams, weights)
        mean_teacher_confidence = float(np.mean(majority_probs_all)) if majority_probs_all else 0.0
    else:
        fused_raw = normalize_01(np.mean(np.stack(cams, axis=0), axis=0))
        mean_teacher_confidence = float(np.mean(pred_confidences)) if pred_confidences else 0.0

    if apply_quality_gate and float(fused_raw.std()) < 0.02:
        fused_post = np.ones((224, 224), dtype=np.float32)
    else:
        if keep_largest:
            thr = pick_contour_threshold(fused_raw, q=contour_quantile)
            fused_post = keep_largest_region(
                fused_raw,
                thr=thr,
                dilation_radius=dilation_radius,
            )
            fused_post = normalize_01(fused_post)
        else:
            fused_post = fused_raw.copy()

    fused_final = fused_post.copy()

    return {
        "image_id": image_id,
        "img_path": img_path,
        "true_label": true_label,
        "true_idx": true_idx,
        "row_idx": row_idx,
        "original_rgb": original_rgb,
        "has_majority": bool(has_majority),
        "has_majority_flag": has_majority_flag,
        "majority_label": majority_label,
        "majority_idx": int(majority_idx),
        "majority_count": int(majority_count),
        "required_majority_votes": int(required_majority_votes),
        "vote_ratio": vote_ratio,
        "vote_hist": vote_hist,
        "mean_teacher_confidence": mean_teacher_confidence,
        "used_identity_mask": False,
        "cam_target_mode": "predicted_label",
        "per_model": per_model,
        "fused_raw": fused_raw,
        "fused_post": fused_post,
        "fused_final": fused_final,
    }


# ============================================================
# Debug figure
# ============================================================

def create_one_image_debug_figure(
    *,
    result: dict,
    results_root: Path,
) -> dict:
    results_root = Path(results_root)

    true_label = result["true_label"]
    majority_label = result["majority_label"]
    image_id = result["image_id"]
    original_rgb = result["original_rgb"]
    per_model = result["per_model"]
    fused_raw = result["fused_raw"]
    fused_post = result["fused_post"]
    fused_final = result["fused_final"]

    out_dir = results_root / "one_image_fusion" / majority_label
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_mask_path = out_dir / f"{image_id}_fused_raw_mask.png"
    post_mask_path = out_dir / f"{image_id}_fused_post_mask.png"
    final_mask_path = out_dir / f"{image_id}_fused_final_mask.png"

    save_mask_png(fused_raw, raw_mask_path)
    save_mask_png(fused_post, post_mask_path)
    save_mask_png(fused_final, final_mask_path)

    n_models = max(1, len(per_model))
    ncols = max(1 + n_models, 4)

    fig = plt.figure(figsize=(4.8 * ncols, 14))

    ax = plt.subplot(4, ncols, 1)
    ax.imshow(original_rgb)
    ax.set_title(
        f"Original\nTrue: {true_label}\nMajority: {majority_label} ({result['majority_count']}/{len(per_model)})",
        fontweight="bold",
    )
    ax.axis("off")

    if len(per_model) > 0:
        for i, pm in enumerate(per_model):
            ax = plt.subplot(4, ncols, i + 2)
            show_overlay(ax, original_rgb, pm["cam_majority"], max_alpha=0.55)
            mark = "✓" if pm["is_majority_match"] else "✗"
            col = "green" if pm["is_majority_match"] else "red"
            meta_tag = "meta" if pm["uses_meta"] else "no-meta"

            ax.set_title(
                f"{pm['name']} [{pm['arch']}|{meta_tag}]\n"
                f"pred={pm['pred_label']} ({pm['confidence_pred']:.1%}) {mark}\n"
                f"p_majority={pm['p_majority']:.1%}  w={pm['weight']:.4f}",
                fontsize=9,
                color=col,
            )
            ax.axis("off")

        ax = plt.subplot(4, ncols, 1 + ncols)
        ax.axis("off")
        ax.set_title("Raw predicted-label Grad-CAM maps", fontweight="bold")

        for i, pm in enumerate(per_model):
            ax = plt.subplot(4, ncols, 2 + ncols + i)
            ax.imshow(pm["cam_majority"], cmap="jet")
            ax.set_title(f"{pm['name']}\n[{pm['arch']}]", fontsize=9)
            ax.axis("off")

    ax = plt.subplot(4, ncols, 1 + 2 * ncols)
    show_overlay(ax, original_rgb, fused_raw, max_alpha=0.55)
    ax.set_title("FUSED RAW overlay", fontweight="bold")
    ax.axis("off")

    ax = plt.subplot(4, ncols, 2 + 2 * ncols)
    show_overlay(ax, original_rgb, fused_post, max_alpha=0.55)
    ax.set_title("FUSED POST overlay", fontweight="bold")
    ax.axis("off")

    ax = plt.subplot(4, ncols, 3 + 2 * ncols)
    show_overlay(ax, original_rgb, fused_final, max_alpha=0.55)
    ax.set_title("FINAL overlay", fontweight="bold")
    ax.axis("off")

    ax = plt.subplot(4, ncols, 4 + 2 * ncols)
    ax.imshow(fused_final, cmap="jet")
    ax.set_title("FINAL raw mask", fontweight="bold")
    ax.axis("off")

    ax = plt.subplot(4, ncols, 1 + 3 * ncols)
    ax.axis("off")
    summary = (
        f"has_majority={result['has_majority']}\n"
        f"has_majority_flag={int(result['has_majority_flag'])}\n"
        f"majority_label={result['majority_label']}\n"
        f"majority_count={result['majority_count']}\n"
        f"required_majority_votes={result['required_majority_votes']}\n"
        f"vote_ratio={result['vote_ratio']:.3f}\n"
        f"mean_teacher_confidence={result['mean_teacher_confidence']:.3f}\n"
        f"used_identity_mask={result['used_identity_mask']}\n"
        f"vote_hist={result['vote_hist']}"
    )
    ax.text(0.0, 1.0, summary, va="top", ha="left", fontsize=10)

    plt.tight_layout()

    fig_path = out_dir / f"{image_id}_pred_label_fusion_debug.png"
    plt.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_path": str(fig_path),
        "raw_mask_path": str(raw_mask_path),
        "post_mask_path": str(post_mask_path),
        "final_mask_path": str(final_mask_path),
    }


# ============================================================
# NPZ generation
# ============================================================

def generate_fused_masks_for_split(
    *,
    df: pd.DataFrame,
    models_and_cams: list[dict],
    meta_features: np.ndarray,
    label2idx: dict,
    idx2label: dict,
    device: torch.device,
    eval_transform,
    results_root: Path,
    out_name: str,
    gamma: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-4,
    limit: Optional[int] = None,
    apply_quality_gate: bool = True,
    keep_largest: bool = True,
    contour_quantile: float = 0.80,
    dilation_radius: int = 15,
    log_every: int = 200,
) -> Path:
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    n = len(df) if limit is None else min(limit, len(df))
    print(f"Generating {n} fused masks -> {out_name}")

    masks = np.zeros((n, 224, 224), dtype=np.float32)
    fused_raw_masks = np.zeros((n, 224, 224), dtype=np.float32)
    image_ids = np.empty((n,), dtype=object)
    labels = np.empty((n,), dtype=object)
    row_idxs = np.zeros((n,), dtype=np.int32)

    fused_post_masks = np.zeros((n, 224, 224), dtype=np.float32)
    true_labels = np.empty((n,), dtype=object)
    majority_labels = np.empty((n,), dtype=object)
    majority_counts = np.zeros((n,), dtype=np.int32)
    vote_ratios = np.zeros((n,), dtype=np.float32)
    used_identity_mask = np.zeros((n,), dtype=np.uint8)

    has_majority_flags = np.zeros((n,), dtype=np.uint8)
    mean_teacher_confidence = np.zeros((n,), dtype=np.float32)
    required_majority_votes_arr = np.zeros((n,), dtype=np.int32)

    for i in range(n):
        if i % log_every == 0:
            print(f"  processing {i}/{n}")

        row = df.iloc[i]

        res = compute_majority_fused_mask_for_row(
            row=row,
            models_and_cams=models_and_cams,
            meta_features=meta_features,
            label2idx=label2idx,
            idx2label=idx2label,
            device=device,
            eval_transform=eval_transform,
            gamma=gamma,
            beta=beta,
            eps=eps,
            apply_quality_gate=apply_quality_gate,
            keep_largest=keep_largest,
            contour_quantile=contour_quantile,
            dilation_radius=dilation_radius,
        )

        masks[i] = res["fused_final"]
        fused_raw_masks[i] = res["fused_raw"]
        image_ids[i] = res["image_id"]
        labels[i] = res["true_label"]
        row_idxs[i] = res["row_idx"]

        fused_post_masks[i] = res["fused_post"]
        true_labels[i] = res["true_label"]
        majority_labels[i] = res["majority_label"]
        majority_counts[i] = res["majority_count"]
        vote_ratios[i] = res["vote_ratio"]
        used_identity_mask[i] = int(res["used_identity_mask"])

        has_majority_flags[i] = np.uint8(res["has_majority_flag"])
        mean_teacher_confidence[i] = np.float32(res["mean_teacher_confidence"])
        required_majority_votes_arr[i] = int(res["required_majority_votes"])

        if i % 25 == 0:
            maybe_empty_cuda_cache(device)

    out_path = results_root / out_name
    np.savez_compressed(
        out_path,
        masks=masks,
        fused_raw_masks=fused_raw_masks,
        image_ids=image_ids,
        labels=labels,
        row_idx=row_idxs,
        fused_post_masks=fused_post_masks,
        true_labels=true_labels,
        majority_labels=majority_labels,
        majority_count=majority_counts,
        required_majority_votes=required_majority_votes_arr,
        vote_ratio=vote_ratios,
        used_identity_mask=used_identity_mask,
        has_majority_flag=has_majority_flags,
        mean_teacher_confidence=mean_teacher_confidence,
    )

    print(f"Saved: {out_path}")
    return out_path


def generate_all_splits(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_and_cams: list[dict],
    meta_features: np.ndarray,
    label2idx: dict,
    idx2label: dict,
    device: torch.device,
    eval_transform,
    results_root: Path,
    gamma: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-4,
    limit_per_split: Optional[int] = None,
    apply_quality_gate: bool = True,
    keep_largest: bool = True,
    contour_quantile: float = 0.80,
    dilation_radius: int = 15,
    log_every: int = 200,
) -> dict[str, str]:
    split_map = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    npz_paths = {}

    for split_name, split_df in split_map.items():
        print(f"\n{'=' * 60}")
        print(f"Split: {split_name.upper()} ({len(split_df)} images)")
        print(f"{'=' * 60}")

        out_path = generate_fused_masks_for_split(
            df=split_df,
            models_and_cams=models_and_cams,
            meta_features=meta_features,
            label2idx=label2idx,
            idx2label=idx2label,
            device=device,
            eval_transform=eval_transform,
            results_root=results_root,
            out_name=f"fused_masks_{split_name}.npz",
            gamma=gamma,
            beta=beta,
            eps=eps,
            limit=limit_per_split,
            apply_quality_gate=apply_quality_gate,
            keep_largest=keep_largest,
            contour_quantile=contour_quantile,
            dilation_radius=dilation_radius,
            log_every=log_every,
        )
        npz_paths[split_name] = str(out_path)

    return npz_paths


# ============================================================
# Config writer
# ============================================================

def save_model_sources_csv(*, results_root: Path, models_and_cams: list[dict]) -> Path:
    rows = []
    for mc in models_and_cams:
        model_cfg = mc.get("model_cfg", {}) or {}
        rows.append({
            "name": mc.get("name"),
            "run_tag": model_cfg.get("run_tag"),
            "arch": mc.get("arch"),
            "source_label": mc.get("source_label"),
            "uses_meta": mc.get("uses_meta"),
            "checkpoint_path": mc.get("ckpt"),
            "run_folder": model_cfg.get("run_folder"),
            "target_layer_name": mc.get("target_layer_name"),
            "config_source_type": model_cfg.get("config_source_type"),
            "config_source_path": model_cfg.get("config_source_path"),
            "arch_name": model_cfg.get("arch_name"),
            "use_midlevel_fusion": model_cfg.get("use_midlevel_fusion"),
            "midlevel_source": model_cfg.get("midlevel_source"),
            "midlevel_pool_type": model_cfg.get("midlevel_pool_type"),
            "midlevel_emb_dim": model_cfg.get("midlevel_emb_dim"),
            "pool_type": model_cfg.get("pool_type"),
            "stem_kernel_size": model_cfg.get("stem_kernel_size"),
            "stem_stride": model_cfg.get("stem_stride"),
            "stem_use_maxpool": model_cfg.get("stem_use_maxpool"),
            "use_multiscale_stem": model_cfg.get("use_multiscale_stem"),
            "stem_branch_kernels": model_cfg.get("stem_branch_kernels"),
            "stem_branch_dilations": model_cfg.get("stem_branch_dilations"),
            "use_multiscale_block": model_cfg.get("use_multiscale_block"),
            "multiscale_block_position": model_cfg.get("multiscale_block_position"),
            "multiscale_kernels": model_cfg.get("multiscale_kernels"),
            "multiscale_dilations": model_cfg.get("multiscale_dilations"),
        })

    out_path = Path(results_root) / "model_sources.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved model sources: {out_path}")
    return out_path


def save_fusion_config(
    *,
    results_root: Path,
    sources: list[SourceSpec],
    models_and_cams: list[dict],
    gamma: float,
    beta: float,
    eps: float,
    apply_quality_gate: bool,
    keep_largest: bool,
    contour_quantile: float,
    dilation_radius: int,
    npz_paths: dict,
) -> Path:
    config = {
        "fusion_name": results_root.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "predicted_label_cam_weighted_fusion_v1_dynamic_majority_metadata",
        "forced_only_default_model_sources": FORCE_ONLY_DEFAULT_MODEL_SOURCES,
        "formula": {
            "majority_rule": "dynamic strict majority: majority_count >= floor(num_models/2) + 1",
            "raw_weight": "w_k = (eps + p_majority_k)^gamma ; if pred_k == majority_label => w_k *= (1 + beta)",
            "normalize_weights": "w_k = w_k / sum_j w_j",
            "fused_map": "Fused(x,y) = sum_k w_k * CAM_k(predicted_label_k, x, y)",
            "post_processing": "optional keep_largest / quality_gate after weighted fusion",
            "final_mask": "Final(x,y) = FusedPost(x,y)",
            "fallback": "If no majority -> Final(x,y) = mean of predicted-label CAMs",
            "has_majority_flag": "1 if a dynamic strict majority exists else 0",
            "mean_teacher_confidence": (
                "if has majority -> mean probability of majority label across all models; "
                "else -> mean of each model's max probability for its own predicted label"
            ),
        },
        "gamma": gamma,
        "beta": beta,
        "eps": eps,
        "apply_quality_gate": apply_quality_gate,
        "keep_largest": keep_largest,
        "contour_quantile": contour_quantile,
        "dilation_radius": dilation_radius,
        "num_models": len(models_and_cams),
        "sources": [
            {"path": str(s.path), "arch": s.arch, "label": s.label}
            for s in sources
        ],
        "models": [
            {
                "name": mc["name"],
                "run_tag": mc.get("model_cfg", {}).get("run_tag"),
                "arch": mc["arch"],
                "source_label": mc["source_label"],
                "uses_meta": mc["uses_meta"],
                "ckpt": mc["ckpt"],
                "target_layer_name": mc.get("target_layer_name"),
                "config_source_type": mc.get("model_cfg", {}).get("config_source_type"),
                "config_source_path": mc.get("model_cfg", {}).get("config_source_path"),
                "run_folder": mc.get("model_cfg", {}).get("run_folder"),
                "model_cfg": {
                    "arch_name": mc.get("model_cfg", {}).get("arch_name"),
                    "use_midlevel_fusion": mc.get("model_cfg", {}).get("use_midlevel_fusion"),
                    "midlevel_source": mc.get("model_cfg", {}).get("midlevel_source"),
                    "midlevel_pool_type": mc.get("model_cfg", {}).get("midlevel_pool_type"),
                    "midlevel_emb_dim": mc.get("model_cfg", {}).get("midlevel_emb_dim"),
                    "pool_type": mc.get("model_cfg", {}).get("pool_type"),
                    "stem_kernel_size": mc.get("model_cfg", {}).get("stem_kernel_size"),
                    "stem_stride": mc.get("model_cfg", {}).get("stem_stride"),
                    "stem_use_maxpool": mc.get("model_cfg", {}).get("stem_use_maxpool"),
                    "use_multiscale_stem": mc.get("model_cfg", {}).get("use_multiscale_stem"),
                    "stem_branch_kernels": mc.get("model_cfg", {}).get("stem_branch_kernels"),
                    "stem_branch_dilations": mc.get("model_cfg", {}).get("stem_branch_dilations"),
                    "use_multiscale_block": mc.get("model_cfg", {}).get("use_multiscale_block"),
                    "multiscale_block_position": mc.get("model_cfg", {}).get("multiscale_block_position"),
                    "multiscale_kernels": mc.get("model_cfg", {}).get("multiscale_kernels"),
                    "multiscale_dilations": mc.get("model_cfg", {}).get("multiscale_dilations"),
                },
            }
            for mc in models_and_cams
        ],
        "outputs": npz_paths,
    }

    out_path = Path(results_root) / "fusion_config.json"
    out_path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"Saved config: {out_path}")
    return out_path




# ============================================================
# Verification helpers
# ============================================================

def validate_fixed_split_manifest(splits: object) -> None:
    """Require mask generation to use the versioned lesion-level split."""
    manifest_path = REPO_ROOT / "splits" / "mask_generation_fixed_split.csv"
    manifest = pd.read_csv(manifest_path, dtype={"image_id": str, "lesion_id": str})

    for partition in ("train", "val", "test"):
        expected = set(
            map(
                tuple,
                manifest.loc[
                    manifest["split"] == partition,
                    ["image_id", "lesion_id"],
                ].to_numpy(),
            )
        )
        frame = getattr(splits, f"{partition}_df")
        actual = set(
            map(
                tuple,
                frame[["image_id", "lesion_id"]].astype(str).to_numpy(),
            )
        )
        if actual != expected:
            raise RuntimeError(
                f"Generated {partition} split does not match {manifest_path}"
            )


def expected_run_tags_by_source() -> dict[str, set[str]]:
    """Load the exact 9/8/7 run tags from the versioned JSON configs."""
    series_by_label = {
        "resnet_regular_baseline_ablation_rerun_full_config_v1": "regular_baseline",
        "resnet_midlevel_fusion_rerun_v1": "midlevel_fusion",
        "resnet_filter_size_7_configs_rerun_v1": "filter_size",
    }
    expected: dict[str, set[str]] = {}
    config_root = REPO_ROOT / "configs" / "resnet_ensemble"
    for label, series in series_by_label.items():
        tags: set[str] = set()
        for path in sorted((config_root / series).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            run_tag = str(payload.get("config", {}).get("run_tag", ""))
            if not run_tag:
                raise ValueError(f"Missing run_tag in {path}")
            if run_tag in tags:
                raise ValueError(f"Duplicate run_tag {run_tag!r} in {series}")
            tags.add(run_tag)
        expected[label] = tags
    return expected

def validate_selected_sources(sources: list[SourceSpec]) -> None:
    """
    Verifies that the code-level source list contains all three expected source labels
    and that every configured source path exists before model loading starts.
    """
    labels = {s.label for s in sources}
    missing_labels = EXPECTED_SOURCE_LABELS - labels
    unexpected_labels = labels - EXPECTED_SOURCE_LABELS

    if missing_labels:
        raise RuntimeError(
            "Missing expected source labels in sources: "
            f"{sorted(missing_labels)}\n"
            "Current sources are:\n"
            + "\n".join(f"  - {s.label}: {s.path}" for s in sources)
        )

    if unexpected_labels:
        print(f"[WARN] Unexpected source labels were configured: {sorted(unexpected_labels)}")

    missing_paths = [s for s in sources if not Path(s.path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            "One or more configured source roots do not exist:\n"
            + "\n".join(f"  - {s.label}: {s.path}" for s in missing_paths)
        )


def validate_loaded_model_sources(
    models_and_cams: list[dict],
    *,
    require_report_ensemble: bool = True,
) -> None:
    """
    Verifies after checkpoint loading that at least one model was loaded from each
    expected source label. This catches the exact bug where the folder name says
    combined but only one source actually contributed models.
    """
    source_counts = Counter(str(mc.get("source_label")) for mc in models_and_cams)
    loaded_labels = set(source_counts.keys())
    missing_loaded_labels = EXPECTED_SOURCE_LABELS - loaded_labels

    print("\nLoaded model counts by source:")
    for label, count in sorted(source_counts.items()):
        print(f"  - {label}: {count}")

    if missing_loaded_labels:
        raise RuntimeError(
            "Models were not loaded from all expected sources. Missing: "
            f"{sorted(missing_loaded_labels)}\n"
            "Loaded counts by source:\n"
            + "\n".join(f"  - {k}: {v}" for k, v in sorted(source_counts.items()))
        )

    if require_report_ensemble:
        mismatches = {
            label: {
                "expected": expected,
                "actual": int(source_counts.get(label, 0)),
            }
            for label, expected in EXPECTED_MODEL_COUNTS_BY_SOURCE.items()
            if int(source_counts.get(label, 0)) != expected
        }
        if len(models_and_cams) != EXPECTED_ENSEMBLE_SIZE or mismatches:
            raise RuntimeError(
                "The report mask ensemble must contain exactly 24 checkpoints "
                "with source counts 9/8/7. "
                f"total={len(models_and_cams)}, mismatches={mismatches}"
            )

        expected_tags = expected_run_tags_by_source()
        actual_tags: dict[str, list[str]] = defaultdict(list)
        for model_and_cam in models_and_cams:
            label = str(model_and_cam.get("source_label"))
            model_cfg = model_and_cam.get("model_cfg", {})
            actual_tags[label].append(str(model_cfg.get("run_tag", "")))

        tag_mismatches = {}
        for label, expected in expected_tags.items():
            actual_list = actual_tags.get(label, [])
            actual = set(actual_list)
            if actual != expected or len(actual_list) != len(actual):
                tag_mismatches[label] = {
                    "missing": sorted(expected - actual),
                    "unexpected": sorted(actual - expected),
                    "duplicates": sorted(
                        tag
                        for tag, count in Counter(actual_list).items()
                        if count > 1
                    ),
                }
        if tag_mismatches:
            raise RuntimeError(
                "Loaded checkpoints do not match the exact 24 versioned ResNet "
                f"configurations: {tag_mismatches}"
            )

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create predicted-label Grad-CAM fusion masks with dynamic majority metadata."
    )

    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Format: path|arch|label  (can be repeated). Ignored when FORCE_ONLY_DEFAULT_MODEL_SOURCES=True.",
    )
    parser.add_argument(
        "--resnet-root",
        action="append",
        default=[],
        help="Root folder containing ResNet run subfolders. Ignored when FORCE_ONLY_DEFAULT_MODEL_SOURCES=True.",
    )
    parser.add_argument(
        "--densenet-root",
        action="append",
        default=[],
        help="Root folder containing DenseNet run subfolders. Ignored when FORCE_ONLY_DEFAULT_MODEL_SOURCES=True.",
    )
    parser.add_argument(
        "--convnext-root",
        action="append",
        default=[],
        help="Root folder containing ConvNeXt run subfolders. Ignored when FORCE_ONLY_DEFAULT_MODEL_SOURCES=True.",
    )

    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help=f"Output folder. Default: {DEFAULT_RESULTS_ROOT}",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-seed", type=int, default=0)

    parser.add_argument("--image-id", default=None)
    parser.add_argument("--max-models-per-source", type=int, default=None)
    parser.add_argument("--strict-load", action="store_true")

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-4)

    parser.add_argument("--contour-quantile", type=float, default=0.80)
    parser.add_argument("--dilation-radius", type=int, default=15)

    parser.add_argument("--no-quality-gate", action="store_true")
    parser.add_argument("--no-keep-largest", action="store_true")

    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--npz-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--npz-limit", type=int, default=None)
    parser.add_argument("--npz-out", default=None)

    parser.add_argument("--save-all-splits", action="store_true")
    parser.add_argument("--log-every", type=int, default=200)

    args = parser.parse_args()

    # ========================================================
    # CODE-LEVEL DEFAULTS FOR RUNNING FULL MASK GENERATION
    # WHEN LAUNCHING DIRECTLY FROM AN IDE WITHOUT CLI FLAGS.
    # ========================================================
    args.save_all_splits = True
    args.save_npz = False
    args.npz_split = "test"
    args.npz_limit = None
    args.strict_load = True

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # This is the important part: use the code-level DEFAULT_MODEL_SOURCES list.
    # In this updated version, DEFAULT_MODEL_SOURCES contains all three ResNet roots.
    if FORCE_ONLY_DEFAULT_MODEL_SOURCES:
        sources = [
            SourceSpec(path=s.path, arch=s.arch, label=s.label)
            for s in DEFAULT_MODEL_SOURCES
        ]
        print("[INFO] FORCE_ONLY_DEFAULT_MODEL_SOURCES=True")
        print("[INFO] Ignoring all CLI sources and using only DEFAULT_MODEL_SOURCES.")
    else:
        sources: list[SourceSpec] = []
        if args.source:
            sources.extend(parse_source_arg(s) for s in args.source)

        sources.extend(
            build_sources_from_roots(
                resnet_roots=args.resnet_root,
                densenet_roots=args.densenet_root,
                convnext_roots=args.convnext_root,
            )
        )

        if not sources:
            sources = [
                SourceSpec(path=s.path, arch=s.arch, label=s.label)
                for s in DEFAULT_MODEL_SOURCES
            ]
            print("[INFO] No sources were provided via CLI. Using DEFAULT_MODEL_SOURCES.")

    print("\nSources:")
    for s in sources:
        print(f"  - {s.label}: arch={s.arch} path={s.path}")

    validate_selected_sources(sources)

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    print(f"\nResults root: {results_root}")

    print("\nLoading HAM10000 metadata and splits.")
    data = prepare_ham_data(split_seed=args.split_seed)

    print(f"Classes ({len(data.label2idx)}): {data.label2idx}")

    print("\nLoading models.")
    models_and_cams = build_models_and_cams(
        sources=sources,
        num_classes=len(data.label2idx),
        meta_feature_dim=data.meta_feature_dim,
        device=device,
        max_models_per_source=args.max_models_per_source,
        strict_load=args.strict_load,
    )

    print(f"\nTotal models loaded: {len(models_and_cams)}")
    validate_loaded_model_sources(
        models_and_cams,
        require_report_ensemble=args.max_models_per_source is None,
    )

    split_map = {
        "train": data.train_df,
        "val": data.val_df,
        "test": data.test_df,
    }

    npz_paths = {}

    try:
        if not args.save_all_splits and not args.save_npz:
            debug_df = data.test_df
            if args.image_id is None:
                image_id = Path(debug_df.iloc[0]["img_path"]).stem
            else:
                image_id = args.image_id

            row_matches = debug_df[debug_df["img_path"].apply(lambda p: Path(str(p)).stem) == image_id]
            if row_matches.empty:
                raise ValueError(f"image_id not found in test split: {image_id}")

            row = row_matches.iloc[0]
            res = compute_majority_fused_mask_for_row(
                row=row,
                models_and_cams=models_and_cams,
                meta_features=data.meta_features,
                label2idx=data.label2idx,
                idx2label=data.idx2label,
                device=device,
                eval_transform=data.eval_transform,
                gamma=args.gamma,
                beta=args.beta,
                eps=args.eps,
                apply_quality_gate=not args.no_quality_gate,
                keep_largest=not args.no_keep_largest,
                contour_quantile=args.contour_quantile,
                dilation_radius=args.dilation_radius,
            )
            debug_paths = create_one_image_debug_figure(result=res, results_root=results_root)
            print("\nDebug paths:")
            for k, v in debug_paths.items():
                print(f"  {k}: {v}")

        if args.save_npz:
            split_name = args.npz_split
            out_name = args.npz_out or f"fused_masks_{split_name}.npz"
            out_path = generate_fused_masks_for_split(
                df=split_map[split_name],
                models_and_cams=models_and_cams,
                meta_features=data.meta_features,
                label2idx=data.label2idx,
                idx2label=data.idx2label,
                device=device,
                eval_transform=data.eval_transform,
                results_root=results_root,
                out_name=out_name,
                gamma=args.gamma,
                beta=args.beta,
                eps=args.eps,
                limit=args.npz_limit,
                apply_quality_gate=not args.no_quality_gate,
                keep_largest=not args.no_keep_largest,
                contour_quantile=args.contour_quantile,
                dilation_radius=args.dilation_radius,
                log_every=args.log_every,
            )
            npz_paths[split_name] = str(out_path)

        if args.save_all_splits:
            npz_paths = generate_all_splits(
                train_df=data.train_df,
                val_df=data.val_df,
                test_df=data.test_df,
                models_and_cams=models_and_cams,
                meta_features=data.meta_features,
                label2idx=data.label2idx,
                idx2label=data.idx2label,
                device=device,
                eval_transform=data.eval_transform,
                results_root=results_root,
                gamma=args.gamma,
                beta=args.beta,
                eps=args.eps,
                limit_per_split=args.npz_limit,
                apply_quality_gate=not args.no_quality_gate,
                keep_largest=not args.no_keep_largest,
                contour_quantile=args.contour_quantile,
                dilation_radius=args.dilation_radius,
                log_every=args.log_every,
            )

        save_model_sources_csv(results_root=results_root, models_and_cams=models_and_cams)
        save_fusion_config(
            results_root=results_root,
            sources=sources,
            models_and_cams=models_and_cams,
            gamma=args.gamma,
            beta=args.beta,
            eps=args.eps,
            apply_quality_gate=not args.no_quality_gate,
            keep_largest=not args.no_keep_largest,
            contour_quantile=args.contour_quantile,
            dilation_radius=args.dilation_radius,
            npz_paths=npz_paths,
        )

        print("\nDone.")
        print(f"Output folder: {results_root}")
        print("Check model_sources.csv and fusion_config.json to verify that all three ResNet sources were used.")

    finally:
        cleanup_models_and_cams(models_and_cams)
        maybe_empty_cuda_cache(device)


if __name__ == "__main__":
    main()
