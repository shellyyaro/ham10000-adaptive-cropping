"""Evaluate every checkpoint represented in the report's final results table.

The six table rows were produced by three experiment roots.  This script keeps
that mapping explicit, loads the five lesion-level fold checkpoints for every
configuration, reproduces the original deterministic evaluation pipeline, and
saves pooled out-of-fold probabilities and publication-ready diagnostic plots.

Default output layout::

    final_table_diagnostic_plots/
      <configuration>/
        fold_0/probabilities.csv
        ...
        pooled_oof_probabilities.csv
        auc_metrics.csv
        summary_metrics.json
        confusion_matrix_counts.png
        confusion_matrix_normalized.png
        roc_curves_ovr.png
        precision_recall_curves_ovr.png

Run ``python evaluate_final_table_checkpoints.py --dry-run`` first, then run
without ``--dry-run`` to perform inference.  No training is performed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resnet_vit_pretrained_comparison.data.AGCWD import build_agcwd_from_cfg
from resnet_vit_pretrained_comparison.data.data_loader import load_ham10000_metadata
from resnet_vit_pretrained_comparison.data.meta_features import (
    build_meta_features_leakage_free,
)
from resnet_vit_pretrained_comparison.models.EfficientNet import (
    EfficientNetWithMeta,
    efficientnet_transforms,
)
from resnet_vit_pretrained_comparison.models.models_cnn_resnet import (
    HAMMultiInputDataset,
)
from resnet_vit_pretrained_comparison.scripts.run_efficientnet_with_meta import (
    _build_crop_eval_dataset,
    _forward_model_optional_attention,
    _load_required_fused_mask_loader,
    _normalize_crop_policy_choices,
    unpack_batch_4,
    unpack_batch_crop_only,
)
from resnet_vit_pretrained_comparison.scripts.run_vit_with_meta import (
    SimpleImageOnlyDataset,
)
from resnet_vit_pretrained_comparison.scripts.splits import (
    make_or_load_repeated_train_val_test_splits,
)
from resnet_vit_pretrained_comparison.training.common_imbalance import (
    build_loss_and_logits_transform,
)
from resnet_vit_pretrained_comparison.training.common_training import (
    compute_class_counts,
    forward_flexible,
)


CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
N_FOLDS = 5
SPLIT_SEED = 42
VAL_FRACTION = 0.20
SPLIT_PREFIX = "ham10000_lesion_5way_tvt"
SPLITS_DIR = REPO_ROOT / "splits" / "repeated_tvt"


@dataclass(frozen=True)
class Configuration:
    slug: str
    display_name: str
    result_dir: Path


RUN_ROOT = Path(
    os.environ.get(
        "HAM10000_RESULTS_ROOT",
        str(REPO_ROOT / "results" / "runs" / "report_experiments"),
    )
)

# Exactly the six rows in Table 1 of "דוח פרויקט גמר 23.8.26.docx".
CONFIGURATIONS = [
    Configuration(
        "full_image",
        "Full image",
        RUN_ROOT / "full_image_efficientnet_baseline",
    ),
    Configuration(
        "fixed_crop",
        "Fixed crop",
        RUN_ROOT / "fixed_crop_viewce030_msecons005_baseline",
    ),
    Configuration(
        "adaptive_baseline",
        "Adaptive baseline",
        RUN_ROOT / "adaptive_crop_viewce030_msecons005_baseline",
    ),
    Configuration(
        "adaptive_ungated_kl_common_aug",
        "Adaptive + alternative objective (ungated KL, common aug.)",
        RUN_ROOT / "adaptive_margin_softwrong_ungated_kl020",
    ),
    Configuration(
        "adaptive_ungated_kl_mild_aug",
        "Adaptive + alternative objective (ungated KL, mild aug.)",
        RUN_ROOT / "adaptive_ungated_kl_mild_aug",
    ),
    Configuration(
        "adaptive_gated_kl_mild_aug",
        "Adaptive + alternative objective (gated KL, mild aug.)",
        RUN_ROOT / "adaptive_gated_kl_mild_aug",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrices and multiclass ROC/PR curves "
        "for every configuration in the report's final table."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "checkpoint_diagnostics",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[item.slug for item in CONFIGURATIONS],
        help="Configuration slugs to evaluate (default: all six).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(N_FOLDS)),
        choices=range(N_FOLDS),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete evaluation plan without requiring local checkpoints.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute folds even when their probability CSV already exists.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def select_configurations(slugs: Iterable[str]) -> list[Configuration]:
    by_slug = {item.slug: item for item in CONFIGURATIONS}
    unknown = sorted(set(slugs) - set(by_slug))
    if unknown:
        raise ValueError(f"Unknown configuration slugs: {unknown}")
    return [by_slug[slug] for slug in slugs]


def find_checkpoint(fold_dir: Path) -> Path:
    checkpoints = sorted(fold_dir.glob("*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint in {fold_dir}, found {checkpoints}"
        )
    return checkpoints[0]


def validate_plan(configurations: list[Configuration], folds: list[int]) -> None:
    errors: list[str] = []
    for item in configurations:
        if not item.result_dir.is_dir():
            errors.append(f"Missing configuration directory: {item.result_dir}")
            continue
        for fold_idx in folds:
            fold_dir = item.result_dir / f"fold_{fold_idx}"
            config_path = fold_dir / "effective_config.json"
            if not config_path.is_file():
                errors.append(f"Missing effective config: {config_path}")
            try:
                checkpoint = find_checkpoint(fold_dir)
            except RuntimeError as exc:
                errors.append(str(exc))
                checkpoint = None
            if checkpoint is not None:
                print(f"[OK] {item.slug} fold_{fold_idx}: {checkpoint.name}")
    if errors:
        raise FileNotFoundError("\n".join(errors))


def load_splits_and_metadata():
    meta, _ = load_ham10000_metadata()
    folds = make_or_load_repeated_train_val_test_splits(
        meta=meta,
        splits_dir=SPLITS_DIR,
        split_seed=SPLIT_SEED,
        n_splits=N_FOLDS,
        val_fraction=VAL_FRACTION,
        stratify_col="dx",
        overwrite=False,
        prefix=SPLIT_PREFIX,
    )
    if len(folds) != N_FOLDS:
        raise RuntimeError(f"Expected {N_FOLDS} folds, received {len(folds)}")
    return meta, folds


def load_model(
    config: dict[str, Any],
    checkpoint: Path,
    meta_feature_dim: int,
    device: torch.device,
) -> EfficientNetWithMeta:
    use_meta = not bool(config.get("no_meta", False)) and bool(
        config.get("use_meta", True)
    )
    in_channels = 4 if str(config.get("mask_mode", "none")) == "concat" else 3

    # pretrained=False prevents any attempted download.  The full trained state
    # is loaded immediately afterwards, so the resulting model is identical.
    model = EfficientNetWithMeta(
        num_classes=len(CLASS_ORDER),
        arch=str(config.get("arch", "efficientnet_b1")),
        pretrained=False,
        meta_feature_dim=int(meta_feature_dim) if use_meta else 0,
        use_meta=use_meta,
        meta_hidden_dim=int(config.get("meta_hidden_dim", 128)),
        clf_hidden_dim=int(config.get("clf_hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        in_channels=in_channels,
        use_mask_attention=bool(config.get("use_mask_attention", False)),
        mask_attention_alpha=float(config.get("mask_attention_alpha", 0.2)),
        mask_attention_base_weight=float(
            config.get("mask_attention_base_weight", 1.0)
        ),
        use_mask_context_branch=bool(config.get("use_mask_context_branch", False)),
        mask_context_branch_mode=str(
            config.get("mask_context_branch_mode", "independent")
        ),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint content in {checkpoint}: {type(state)}")
    state = {
        (key.removeprefix("module.")): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_logits_transform(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    label2idx: dict[str, int],
    device: torch.device,
):
    class_counts = compute_class_counts(train_df, label2idx, label_col="dx")
    _, logits_transform, _ = build_loss_and_logits_transform(
        loss_name=str(config.get("loss_name", "ce")),
        class_weighting=str(config.get("class_weighting", "none")),
        label_smoothing=float(config.get("label_smoothing", 0.0)),
        focal_gamma=float(config.get("focal_gamma", 2.0)),
        focal_alpha_mode=str(config.get("focal_alpha_mode", "none")),
        logit_adjustment=str(config.get("logit_adjustment", "none")),
        tau=float(config.get("tau", 1.0)),
        class_counts_np=class_counts,
        device=device,
    )
    return logits_transform


def forward_crop_batch(
    model,
    batch,
    device,
    config,
    logits_transform,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_meta = not bool(config.get("no_meta", False)) and bool(
        config.get("use_meta", True)
    )
    images, meta_vecs, labels, extras = unpack_batch_crop_only(
        batch, device, use_meta=use_meta
    )
    attention_masks = extras.get("attention_mask") if isinstance(extras, dict) else None
    context_images = (
        extras.get("mask_context_image") if isinstance(extras, dict) else None
    )
    logits = _forward_model_optional_attention(
        model=model,
        images=images,
        meta_vecs=meta_vecs,
        use_meta=use_meta,
        attention_masks=attention_masks,
        use_mask_attention=bool(config.get("use_mask_attention", False)),
        mask_context_images=context_images,
        use_mask_context_branch=bool(config.get("use_mask_context_branch", False)),
    )
    return logits_transform(logits), labels


@torch.inference_mode()
def infer_fold_probabilities(
    *,
    model: EfficientNetWithMeta,
    config: dict[str, Any],
    split_df: pd.DataFrame,
    train_df: pd.DataFrame,
    meta_features: np.ndarray,
    label2idx: dict[str, int],
    device: torch.device,
    batch_size_override: int | None,
) -> pd.DataFrame:
    df_reset = split_df.reset_index(drop=True)
    batch_size = int(
        batch_size_override
        or config.get("eval_batch_size", config.get("batch_size", 16))
    )
    input_source = str(config.get("input_source", "original")).lower()
    eval_crop_mode = str(config.get("eval_crop_mode", "fixed")).lower()
    logits_transform = build_logits_transform(config, train_df, label2idx, device)
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    if input_source == "crop":
        mask_loader = _load_required_fused_mask_loader(
            npz_path=config.get("mask_npz_test"),
            split_name="test",
            split_df=df_reset,
            debug=False,
            img_col_name="img_path",
        )
        agcwd_transform = build_agcwd_from_cfg(config)
        common = dict(
            split_df=df_reset,
            meta_features=meta_features,
            label2idx=label2idx,
            fused_mask_loader=mask_loader,
            agcwd_transform=agcwd_transform,
            img_size=int(config.get("img_size", 224)),
            crop_resize_mode=str(config.get("crop_resize_mode", "stretch")),
            adaptive_crop_shape=str(
                config.get("adaptive_crop_shape", "closest_rectangle")
            ),
            crop_threshold=float(config.get("crop_threshold", 0.6)),
            crop_margin_ratio=float(config.get("crop_margin_ratio", 0.0)),
            crop_min_area=int(config.get("crop_min_area", 10)),
            crop_fallback_mode=str(config.get("crop_fallback_mode", "keep_original")),
            background_mode=str(config.get("background_mode", "blur")),
            blur_kernel_size=int(config.get("blur_kernel_size", 11)),
            blur_sigma=float(config.get("blur_sigma", 4.0)),
            softmultiply_alpha=float(config.get("softmultiply_alpha", 0.5)),
            mask_power=float(config.get("mask_power", 1.0)),
            missing_mask_mode=str(config.get("missing_mask_mode", "keep_original")),
            debug_mask_pipeline=False,
            debug_max_samples=0,
            return_mask_context_image=bool(config.get("use_mask_context_branch", False)),
            mask_context_view_mode=str(config.get("mask_context_view_mode", "full")),
        )

        if eval_crop_mode == "fixed":
            dataset = _build_crop_eval_dataset(**common)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            for batch in loader:
                logits, labels = forward_crop_batch(
                    model, batch, device, config, logits_transform
                )
                all_probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        else:
            policies = _normalize_crop_policy_choices(
                config.get("eval_crop_policy_choices"),
                fallback_threshold=float(config.get("crop_threshold", 0.6)),
                fallback_margin_ratio=float(config.get("crop_margin_ratio", 0.0)),
            )
            loaders = []
            for policy in policies:
                policy_dataset_args = dict(common)
                policy_dataset_args["adaptive_crop_shape"] = str(
                    policy.get(
                        "adaptive_crop_shape",
                        config.get("adaptive_crop_shape", "closest_rectangle"),
                    )
                )
                dataset = _build_crop_eval_dataset(
                    **policy_dataset_args,
                    single_crop_policy=policy,
                )
                loaders.append(
                    DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                    )
                )

            for batches in zip(*loaders):
                logits_by_policy = []
                labels_ref = None
                for batch in batches:
                    logits, labels = forward_crop_batch(
                        model, batch, device, config, logits_transform
                    )
                    if labels_ref is None:
                        labels_ref = labels
                    elif not torch.equal(labels_ref, labels):
                        raise RuntimeError("Crop-policy loaders are not aligned.")
                    logits_by_policy.append(logits)
                stacked = torch.stack(logits_by_policy, dim=0)
                if eval_crop_mode == "tta_probs_avg":
                    probabilities = torch.softmax(stacked.float(), dim=-1).mean(dim=0)
                elif eval_crop_mode == "tta_logits_avg":
                    probabilities = torch.softmax(stacked.mean(dim=0).float(), dim=-1)
                else:
                    raise ValueError(f"Unsupported eval_crop_mode: {eval_crop_mode}")
                all_probabilities.append(probabilities.cpu().numpy())
                all_labels.append(labels_ref.cpu().numpy())
    else:
        _, eval_transform = efficientnet_transforms(
            pretrained=bool(config.get("pretrained", True)),
            img_size=int(config.get("img_size", 224)),
            rotation_deg=0.0,
            vflip=False,
            use_color_jitter=bool(config.get("use_color_jitter", False)),
            color_jitter_brightness=float(config.get("color_jitter_brightness", 0.0)),
            color_jitter_contrast=float(config.get("color_jitter_contrast", 0.0)),
            color_jitter_saturation=float(config.get("color_jitter_saturation", 0.0)),
            color_jitter_hue=float(config.get("color_jitter_hue", 0.0)),
            hflip=False,
        )
        use_meta = not bool(config.get("no_meta", False)) and bool(
            config.get("use_meta", True)
        )
        if use_meta:
            dataset = HAMMultiInputDataset(
                df=df_reset,
                meta_features=meta_features,
                hog_lbp_features=None,
                transform=eval_transform,
                label_col="dx",
                label2idx=label2idx,
                use_masked_image=False,
                use_hog_lbp=False,
                mask_col_name="img_path_masked",
                img_col_name="img_path",
                fused_mask_loader=None,
                fused_mask_mode="none",
                rotation_deg=0.0,
                hflip_p=0.0,
                vflip_p=0.0,
                agcwd_transform=build_agcwd_from_cfg(config),
            )
        else:
            dataset = SimpleImageOnlyDataset(
                df=df_reset,
                transform=eval_transform,
                label_col="dx",
                label2idx=label2idx,
                img_col_name="img_path",
                fused_mask_loader=None,
                fused_mask_mode="none",
                rotation_deg=0.0,
                hflip_p=0.0,
                vflip_p=0.0,
                agcwd_transform=build_agcwd_from_cfg(config),
            )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            images, meta_vecs, labels, masks = unpack_batch_4(batch, device)
            logits = forward_flexible(
                model,
                images,
                meta_vecs,
                use_meta=use_meta,
                masks=masks,
                use_mask=False,
                mask_mode="none",
            )
            logits = logits_transform(logits)
            all_probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probabilities = np.concatenate(all_probabilities, axis=0)
    true_indices = np.concatenate(all_labels, axis=0).astype(int)
    if len(df_reset) != len(probabilities):
        raise RuntimeError(
            f"Prediction count mismatch: dataframe={len(df_reset)}, "
            f"probabilities={len(probabilities)}"
        )

    output = df_reset.copy()
    output["y_true_idx"] = true_indices
    output["true_dx"] = [CLASS_ORDER[index] for index in true_indices]
    predicted = probabilities.argmax(axis=1)
    output["y_pred_idx"] = predicted
    output["pred_dx"] = [CLASS_ORDER[index] for index in predicted]
    output["is_correct"] = (true_indices == predicted).astype(int)
    for class_idx, class_name in enumerate(CLASS_ORDER):
        output[f"prob_{class_name}"] = probabilities[:, class_idx]
    return output


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    title: str,
) -> None:
    counts = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_ORDER)))
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )
    pd.DataFrame(counts, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        output_dir / "confusion_matrix_counts.csv"
    )
    pd.DataFrame(normalized, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        output_dir / "confusion_matrix_normalized.csv"
    )
    for matrix, filename, value_format, vmax, color_threshold in [
        (counts, "confusion_matrix_counts.png", "d", None, counts.max() / 2),
        (normalized, "confusion_matrix_normalized.png", ".2f", 1.0, 0.5),
    ]:
        fig, axis = plt.subplots(figsize=(8.2, 7.2))
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        axis.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=45, ha="right")
        axis.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
        axis.set_xlabel("Predicted category")
        axis.set_ylabel("True category")
        axis.set_title(title)
        for row in range(len(CLASS_ORDER)):
            for column in range(len(CLASS_ORDER)):
                value = matrix[row, column]
                text_value = format(int(value), value_format) if value_format == "d" else format(value, value_format)
                axis.text(
                    column,
                    row,
                    text_value,
                    ha="center",
                    va="center",
                    color="white" if value >= color_threshold else "black",
                    fontsize=9,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_curves_and_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
    title: str,
) -> pd.DataFrame:
    binary = label_binarize(y_true, classes=range(len(CLASS_ORDER)))
    rows: list[dict[str, Any]] = []

    fig_roc, ax_roc = plt.subplots(figsize=(8.4, 7.2))
    fig_pr, ax_pr = plt.subplots(figsize=(8.4, 7.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_ORDER)))
    for class_idx, (class_name, color) in enumerate(zip(CLASS_ORDER, colors)):
        fpr, tpr, _ = roc_curve(binary[:, class_idx], probabilities[:, class_idx])
        precision, recall, _ = precision_recall_curve(
            binary[:, class_idx], probabilities[:, class_idx]
        )
        roc_auc = auc(fpr, tpr)
        average_precision = average_precision_score(
            binary[:, class_idx], probabilities[:, class_idx]
        )
        prevalence = float(binary[:, class_idx].mean())
        rows.append(
            {
                "class": class_name,
                "roc_auc_ovr": roc_auc,
                "average_precision": average_precision,
                "prevalence": prevalence,
            }
        )
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{class_name} (AUC={roc_auc:.3f})")
        ax_pr.plot(
            recall,
            precision,
            color=color,
            lw=2,
            label=f"{class_name} (AP={average_precision:.3f})",
        )

    micro_fpr, micro_tpr, _ = roc_curve(binary.ravel(), probabilities.ravel())
    micro_precision, micro_recall, _ = precision_recall_curve(
        binary.ravel(), probabilities.ravel()
    )
    micro_roc_auc = auc(micro_fpr, micro_tpr)
    micro_ap = average_precision_score(binary, probabilities, average="micro")
    macro_roc_auc = roc_auc_score(binary, probabilities, average="macro")
    weighted_roc_auc = roc_auc_score(binary, probabilities, average="weighted")
    macro_ap = average_precision_score(binary, probabilities, average="macro")
    weighted_ap = average_precision_score(binary, probabilities, average="weighted")

    rows.extend(
        [
            {"class": "micro-average", "roc_auc_ovr": micro_roc_auc, "average_precision": micro_ap, "prevalence": np.nan},
            {"class": "macro-average", "roc_auc_ovr": macro_roc_auc, "average_precision": macro_ap, "prevalence": np.nan},
            {"class": "weighted-average", "roc_auc_ovr": weighted_roc_auc, "average_precision": weighted_ap, "prevalence": np.nan},
        ]
    )

    ax_roc.plot(micro_fpr, micro_tpr, "k--", lw=2.3, label=f"micro-average (AUC={micro_roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.5)
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", title=title)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.grid(alpha=0.2)
    ax_roc.legend(loc="lower right", fontsize=8)
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / "roc_curves_ovr.png", dpi=300, bbox_inches="tight")
    plt.close(fig_roc)

    ax_pr.plot(
        micro_recall,
        micro_precision,
        "k--",
        lw=2.3,
        label=f"micro-average (AP={micro_ap:.3f})",
    )
    ax_pr.set(xlabel="Recall", ylabel="Precision", title=title)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.grid(alpha=0.2)
    ax_pr.legend(loc="lower left", fontsize=8)
    fig_pr.tight_layout()
    fig_pr.savefig(output_dir / "precision_recall_curves_ovr.png", dpi=300, bbox_inches="tight")
    plt.close(fig_pr)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "auc_metrics.csv", index=False)
    return metrics


def save_all_diagnostics(predictions: pd.DataFrame, output_dir: Path, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = predictions["y_true_idx"].to_numpy(dtype=int)
    y_pred = predictions["y_pred_idx"].to_numpy(dtype=int)
    probabilities = predictions[[f"prob_{name}" for name in CLASS_ORDER]].to_numpy()
    plot_confusion_matrices(y_true, y_pred, output_dir, title)
    auc_metrics = plot_curves_and_metrics(y_true, probabilities, output_dir, title)
    summary = {
        "n_images": int(len(predictions)),
        "n_lesions": int(predictions["lesion_id"].nunique()) if "lesion_id" in predictions else None,
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_roc_auc_ovr": float(
            auc_metrics.loc[auc_metrics["class"] == "macro-average", "roc_auc_ovr"].iloc[0]
        ),
        "micro_roc_auc_ovr": float(
            auc_metrics.loc[auc_metrics["class"] == "micro-average", "roc_auc_ovr"].iloc[0]
        ),
        "macro_average_precision": float(
            auc_metrics.loc[auc_metrics["class"] == "macro-average", "average_precision"].iloc[0]
        ),
        "micro_average_precision": float(
            auc_metrics.loc[auc_metrics["class"] == "micro-average", "average_precision"].iloc[0]
        ),
    }
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def verify_against_original(
    predictions: pd.DataFrame,
    fold_input: Path,
    fold_output: Path,
) -> None:
    original_path = fold_input / "test_predictions_for_summary.csv"
    if not original_path.is_file():
        raise FileNotFoundError(f"Missing original predictions: {original_path}")
    original = pd.read_csv(original_path)
    required = {"row_idx", "y_true_idx", "y_pred_idx"}
    if not required.issubset(original.columns):
        raise ValueError(
            f"Original prediction file lacks {sorted(required - set(original.columns))}: "
            f"{original_path}"
        )
    comparison = predictions[list(required)].merge(
        original[list(required)],
        on="row_idx",
        suffixes=("_new", "_original"),
        validate="one_to_one",
    )
    true_matches = comparison["y_true_idx_new"].eq(
        comparison["y_true_idx_original"]
    )
    prediction_matches = comparison["y_pred_idx_new"].eq(
        comparison["y_pred_idx_original"]
    )
    verification = {
        "n_rows_new": int(len(predictions)),
        "n_rows_original": int(len(original)),
        "n_rows_matched": int(len(comparison)),
        "true_label_matches": int(true_matches.sum()),
        "hard_prediction_matches": int(prediction_matches.sum()),
        "exact_match": bool(
            len(comparison) == len(predictions) == len(original)
            and true_matches.all()
            and prediction_matches.all()
        ),
        "original_predictions": str(original_path.resolve()),
    }
    (fold_output / "reproduction_check.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    if not verification["exact_match"]:
        raise RuntimeError(
            "Recomputed predictions do not exactly reproduce the original fold. "
            f"See {fold_output / 'reproduction_check.json'}"
        )


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    configurations = select_configurations(args.configs)
    print(f"Output root: {args.output_root.resolve()}")
    if args.dry_run:
        for item in configurations:
            print(f"[CONFIG] {item.slug}: {item.result_dir}")
            for fold_idx in args.folds:
                print(f"  fold_{fold_idx}")
        print(
            "Dry run completed: the six-config/five-fold evaluation plan is valid; "
            "local checkpoints are checked only during a real evaluation run."
        )
        return

    validate_plan(configurations, args.folds)

    device = choose_device(args.device)
    print(f"Device: {device}")
    meta, splits = load_splits_and_metadata()
    label2idx = {name: index for index, name in enumerate(CLASS_ORDER)}
    args.output_root.mkdir(parents=True, exist_ok=True)

    for item in configurations:
        print(f"\n=== {item.display_name} ===")
        config_output = args.output_root / item.slug
        pooled: list[pd.DataFrame] = []
        for fold_idx in args.folds:
            fold = splits[fold_idx]
            fold_input = item.result_dir / f"fold_{fold_idx}"
            fold_output = config_output / f"fold_{fold_idx}"
            probability_path = fold_output / "probabilities.csv"
            if probability_path.exists() and not args.overwrite:
                print(f"[CACHE] {item.slug} fold_{fold_idx}")
                fold_predictions = pd.read_csv(probability_path)
            else:
                print(f"[INFER] {item.slug} fold_{fold_idx}")
                fold_output.mkdir(parents=True, exist_ok=True)
                config = json.loads(
                    (fold_input / "effective_config.json").read_text(encoding="utf-8")
                )
                meta_features, _, _, meta_feature_dim = build_meta_features_leakage_free(
                    meta, fold.train_df["row_idx"].values
                )
                model = load_model(
                    config,
                    find_checkpoint(fold_input),
                    meta_feature_dim,
                    device,
                )
                fold_predictions = infer_fold_probabilities(
                    model=model,
                    config=config,
                    split_df=fold.test_df,
                    train_df=fold.train_df,
                    meta_features=meta_features,
                    label2idx=label2idx,
                    device=device,
                    batch_size_override=args.batch_size,
                )
                fold_predictions.insert(0, "fold", fold_idx)
                fold_predictions.insert(0, "configuration", item.slug)
                fold_predictions.to_csv(probability_path, index=False)
                save_all_diagnostics(
                    fold_predictions,
                    fold_output,
                    f"{item.display_name} — fold {fold_idx}",
                )
                del model
                cleanup()
            verify_against_original(fold_predictions, fold_input, fold_output)
            pooled.append(fold_predictions)

        pooled_predictions = pd.concat(pooled, ignore_index=True)
        pooled_predictions.to_csv(
            config_output / "pooled_oof_probabilities.csv", index=False
        )
        save_all_diagnostics(
            pooled_predictions,
            config_output,
            f"{item.display_name} — pooled out-of-fold",
        )
        print(f"[DONE] {config_output.resolve()}")

    print("\nAll requested diagnostic folders were generated successfully.")


if __name__ == "__main__":
    main()
