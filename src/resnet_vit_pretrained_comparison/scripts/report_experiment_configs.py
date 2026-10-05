"""Authoritative code-defined configurations for the six report experiments.

This module is intentionally self-contained.  The final experiment runner must
not reconstruct experiments from old console dumps, TXT result records, or an
external project directory.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _merged(*parts: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in parts:
        result.update(deepcopy(part))
    return result


FIXED_TRAIN_POLICIES = [
    {"crop_mode": "fixed_margin", "crop_threshold": 0.20, "crop_margin_ratio": 0.00, "policy_index": 0},
    {"crop_mode": "fixed_margin", "crop_threshold": 0.30, "crop_margin_ratio": 0.05, "policy_index": 1},
    {"crop_mode": "fixed_margin", "crop_threshold": 0.40, "crop_margin_ratio": 0.10, "policy_index": 2},
    {"crop_mode": "fixed_margin", "crop_threshold": 0.20, "crop_margin_ratio": 0.05, "policy_index": 3},
    {"crop_mode": "fixed_margin", "crop_threshold": 0.30, "crop_margin_ratio": 0.10, "policy_index": 4},
    {"crop_mode": "fixed_margin", "crop_threshold": 0.40, "crop_margin_ratio": 0.15, "policy_index": 5},
]

FIXED_EVAL_POLICIES = deepcopy(FIXED_TRAIN_POLICIES[:3])

ADAPTIVE_POLICIES = [
    {
        "crop_mode": "adaptive_mask_area",
        "crop_threshold": 0.60,
        "target_mask_area_ratio": ratio,
        "policy_index": index,
    }
    for index, ratio in enumerate((0.10, 0.20, 0.30, 0.40, 0.50, 0.60))
]


BASE_CONFIG: dict[str, Any] = {
    # Model and metadata branch.
    "arch": "efficientnet_b1",
    "pretrained": True,
    "use_meta": True,
    "no_meta": False,
    "meta_hidden_dim": 128,
    "clf_hidden_dim": 256,
    "dropout": 0.30,
    "freeze_backbone_epochs": 0,
    # Optimisation and imbalance handling.
    "batch_size": 16,
    "train_microbatch_size": 8,
    "eval_batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_epochs": 50,
    "patience": 7,
    "loss_name": "ce",
    "class_weighting": "inverse_freq",
    "label_smoothing": 0.05,
    "logit_adjustment": "none",
    "tau": 1.0,
    "sampler": "none",
    "focal_gamma": 2.0,
    "focal_alpha_mode": "none",
    # RGB input and masks used only to calculate crop coordinates.
    "img_size": 224,
    "use_fused_masks": False,
    "mask_mode": "none",
    "mask_npz_train": None,
    "mask_npz_val": None,
    "mask_npz_test": None,
    "use_mask_attention": False,
    "mask_attention_alpha": 0.20,
    "mask_attention_base_weight": 1.0,
    "use_mask_context_branch": False,
    "mask_context_view_mode": "full",
    "mask_context_branch_mode": "independent",
    "mask_power": 1.0,
    "missing_mask_mode": "keep_original",
    "background_mode": "blur",
    "blur_kernel_size": 11,
    "blur_sigma": 4.0,
    "softmultiply_alpha": 0.5,
    "agcwd_preset": "none",
    # Crop construction defaults.
    "crop_min_area": 10,
    "crop_fallback_mode": "keep_original",
    "crop_resize_mode": "stretch",
    "adaptive_crop_shape": "closest_rectangle",
    "random_crop_policy": False,
    "crop_policy_choices": None,
    # Common augmentation used by the first four configurations.
    "rotation_deg": 20.0,
    "hflip": True,
    "hflip_p": 0.5,
    "vflip": True,
    "vflip_p": 0.5,
    "use_color_jitter": True,
    "color_jitter_brightness": 0.15,
    "color_jitter_contrast": 0.15,
    "color_jitter_saturation": 0.15,
    "color_jitter_hue": 0.03,
    "per_view_color_jitter_brightness": True,
    "per_view_color_jitter_contrast": True,
    "per_view_color_jitter_saturation": True,
    "per_view_color_jitter_hue": False,
    # Explicit inactive loss components.  Each experiment overrides only what
    # it actually uses, making the executed objective visible in code.
    "main_loss_type": "ce",
    "main_loss_weight": 1.0,
    "main_margin_target": 0.10,
    "main_margin_temperature": 0.10,
    "consistency_weight": 0.0,
    "consistency_loss_type": "mse_probs",
    "consistency_gate_mode": "none",
    "consistency_gate_temperature": 0.05,
    "consistency_gate_margin_threshold": 0.0,
    "view_ce_weight": 0.0,
    "worst_view_ce_weight": 0.0,
    "consensus_kl_weight": 0.0,
    "consensus_kl_temperature": 1.0,
    "wrong_crop_count_weight": 0.0,
    "wrong_crop_count_temperature": 0.15,
    "wrong_crop_count_reduction": "mean",
    "feature_consistency_weight": 0.0,
    "feature_consistency_loss_type": "cosine",
    "drop_last_train": False,
    "pin_memory": False,
    "debug_mask_pipeline": False,
    "debug_stop_after_model_build": False,
}

FULL_IMAGE = {
    "run_tag": "full_image_efficientnet_baseline",
    "input_source": "original",
    "use_crop_as_input": False,
    "strict_mask_loading": False,
    "crop_threshold": 0.30,
    "crop_margin_ratio": 0.05,
    "multi_view_crop": False,
    "multi_view_crop_policy_choices": None,
    "eval_crop_mode": "fixed",
    "eval_crop_policy_choices": None,
    "multi_view_loss_mode": "avg_logits",
}

FIXED_CROP_BASELINE = {
    "run_tag": "fixed_crop_viewce030_msecons005_baseline",
    "input_source": "crop",
    "use_crop_as_input": True,
    "strict_mask_loading": True,
    "crop_threshold": 0.30,
    "crop_margin_ratio": 0.05,
    "multi_view_crop": True,
    "multi_view_crop_policy_choices": FIXED_TRAIN_POLICIES,
    "eval_crop_mode": "tta_probs_avg",
    "eval_crop_policy_choices": FIXED_EVAL_POLICIES,
    "multi_view_loss_mode": "avg_probs_consistency",
    "view_ce_weight": 0.30,
    "consistency_weight": 0.05,
    "consistency_loss_type": "mse_probs",
}

ADAPTIVE_INPUT = {
    "input_source": "crop",
    "use_crop_as_input": True,
    "strict_mask_loading": True,
    "crop_threshold": 0.60,
    "crop_margin_ratio": 0.0,
    "multi_view_crop": True,
    "multi_view_crop_policy_choices": ADAPTIVE_POLICIES,
    "eval_crop_mode": "tta_probs_avg",
    "eval_crop_policy_choices": ADAPTIVE_POLICIES,
    "multi_view_loss_mode": "avg_probs_consistency",
}

ADAPTIVE_BASELINE_OBJECTIVE = {
    "main_loss_type": "ce",
    "view_ce_weight": 0.30,
    "consistency_weight": 0.05,
    "consistency_loss_type": "mse_probs",
    "consistency_gate_mode": "none",
}

ALTERNATIVE_UNGATED_OBJECTIVE = {
    "main_loss_type": "avg_margin_hinge",
    "main_margin_target": 0.10,
    "wrong_crop_count_weight": 0.30,
    "wrong_crop_count_temperature": 0.15,
    "wrong_crop_count_reduction": "mean",
    "consistency_weight": 0.20,
    "consistency_loss_type": "kl_probs",
    "consistency_gate_mode": "none",
    "view_ce_weight": 0.0,
    "worst_view_ce_weight": 0.0,
}

MILD_AUGMENTATION = {
    "rotation_deg": 0.0,
    "hflip": False,
    "hflip_p": 0.0,
    "vflip": False,
    "vflip_p": 0.0,
    "color_jitter_brightness": 0.10,
    "color_jitter_contrast": 0.10,
    "color_jitter_saturation": 0.10,
    "color_jitter_hue": 0.03,
    "per_view_color_jitter_brightness": True,
    "per_view_color_jitter_contrast": False,
    "per_view_color_jitter_saturation": False,
    "per_view_color_jitter_hue": False,
    "eval_batch_size": 16,
}

GATED_KL = {
    "consistency_gate_mode": "true_margin",
    "consistency_gate_temperature": 0.05,
    "consistency_gate_margin_threshold": 0.0,
}


REPORT_EXPERIMENT_CONFIGS: dict[str, dict[str, Any]] = {
    "full_image_efficientnet_baseline": _merged(BASE_CONFIG, FULL_IMAGE),
    "fixed_crop_viewce030_msecons005_baseline": _merged(
        BASE_CONFIG, FIXED_CROP_BASELINE
    ),
    "adaptive_crop_viewce030_msecons005_baseline": _merged(
        BASE_CONFIG,
        ADAPTIVE_INPUT,
        ADAPTIVE_BASELINE_OBJECTIVE,
        {"run_tag": "adaptive_crop_viewce030_msecons005_baseline"},
    ),
    "adaptive_margin_softwrong_ungated_kl020": _merged(
        BASE_CONFIG,
        ADAPTIVE_INPUT,
        ALTERNATIVE_UNGATED_OBJECTIVE,
        {"run_tag": "adaptive_margin_softwrong_ungated_kl020"},
    ),
    "adaptive_ungated_kl_mild_aug": _merged(
        BASE_CONFIG,
        ADAPTIVE_INPUT,
        ALTERNATIVE_UNGATED_OBJECTIVE,
        MILD_AUGMENTATION,
        {"run_tag": "adaptive_ungated_kl_mild_aug"},
    ),
    "adaptive_gated_kl_mild_aug": _merged(
        BASE_CONFIG,
        ADAPTIVE_INPUT,
        ALTERNATIVE_UNGATED_OBJECTIVE,
        MILD_AUGMENTATION,
        GATED_KL,
        {"run_tag": "adaptive_gated_kl_mild_aug"},
    ),
}


def get_report_experiment_config(slug: str) -> dict[str, Any]:
    """Return an isolated copy so fold-specific edits cannot mutate constants."""
    try:
        return deepcopy(REPORT_EXPERIMENT_CONFIGS[slug])
    except KeyError as exc:
        raise KeyError(f"Unknown report experiment configuration: {slug}") from exc
