from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import openpyxl  # noqa: F401 - fail early if Excel export support is unavailable
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resnet_vit_pretrained_comparison.data.data_loader import (  # noqa: E402
    load_ham10000_metadata,
)
from resnet_vit_pretrained_comparison.data.meta_features import (  # noqa: E402
    build_meta_features_leakage_free,
)
from resnet_vit_pretrained_comparison.scripts.run_efficientnet_with_meta import (  # noqa: E402
    run_efficientnet_experiment,
)
from resnet_vit_pretrained_comparison.scripts.report_experiment_configs import (  # noqa: E402
    ADAPTIVE_POLICIES,
    FIXED_EVAL_POLICIES,
    FIXED_TRAIN_POLICIES,
    get_report_experiment_config,
)
from resnet_vit_pretrained_comparison.scripts.splits import (  # noqa: E402
    make_or_load_repeated_train_val_test_splits,
)


N_FOLDS = 5
SPLIT_SEED = 42
MODEL_SEED = 0
VAL_FRACTION = 0.20
SPLIT_SUBDIR = "repeated_tvt"
SPLIT_PREFIX = "ham10000_lesion_5way_tvt"
CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

MASK_ROOT = Path(
    os.environ.get("HAM10000_MASK_ROOT", str(REPO_ROOT / "data" / "generated_masks"))
)
MASK_SOURCE_PATHS = [
    MASK_ROOT / "fused_masks_train.npz",
    MASK_ROOT / "fused_masks_val.npz",
    MASK_ROOT / "fused_masks_test.npz",
]

DEFAULT_RESULTS_ROOT = Path(
    os.environ.get(
        "HAM10000_RESULTS_ROOT",
        str(REPO_ROOT / "results" / "runs" / "report_experiments"),
    )
)

@dataclass(frozen=True)
class ConfigurationSpec:
    slug: str
    display_name: str
    preprocessing_mode: str


CONFIGURATIONS = [
    ConfigurationSpec(
        slug="full_image_efficientnet_baseline",
        display_name="Full-image EfficientNet-B1 baseline",
        preprocessing_mode="full_image",
    ),
    ConfigurationSpec(
        slug="fixed_crop_viewce030_msecons005_baseline",
        display_name="Classical fixed-crop baseline",
        preprocessing_mode="fixed_crop",
    ),
    ConfigurationSpec(
        slug="adaptive_crop_viewce030_msecons005_baseline",
        display_name="Adaptive crops with the fixed-baseline objective",
        preprocessing_mode="adaptive_crop",
    ),
    ConfigurationSpec(
        slug="adaptive_margin_softwrong_ungated_kl020",
        display_name="Adaptive crops with ungated KL loss",
        preprocessing_mode="adaptive_crop",
    ),
    ConfigurationSpec(
        slug="adaptive_ungated_kl_mild_aug",
        display_name="Adaptive alternative objective with mild augmentation and ungated KL",
        preprocessing_mode="adaptive_crop",
    ),
    ConfigurationSpec(
        slug="adaptive_gated_kl_mild_aug",
        display_name="Adaptive alternative objective with mild augmentation and gated KL",
        preprocessing_mode="adaptive_crop",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run six EfficientNet configurations on five lesion-level folds "
            "and create fold-level and pooled result workbooks."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory for all training and summary outputs.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(N_FOLDS)),
        help="Subset of folds to run, for example: --folds 0 1.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[spec.slug for spec in CONFIGURATIONS],
        help="Subset of configuration slugs to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/configs/splits and print the plan without training.",
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Incomplete folds are now archived and "
            "restarted automatically; completed folds are always skipped."
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training without CUDA. Disabled by default to prevent accidental long CPU runs.",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_preprocessing(spec: ConfigurationSpec, config: dict[str, Any]) -> None:
    required = {
        "arch",
        "run_tag",
        "batch_size",
        "num_epochs",
        "patience",
        "input_source",
        "multi_view_crop",
        "multi_view_loss_mode",
        "main_loss_type",
        "consistency_loss_type",
        "consistency_gate_mode",
    }
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(f"Code configuration {spec.slug} is missing keys: {missing}")

    if spec.preprocessing_mode == "full_image":
        if not (
            config["input_source"] == "original"
            and config["use_crop_as_input"] is False
            and config["multi_view_crop"] is False
            and config["multi_view_crop_policy_choices"] is None
            and config["eval_crop_policy_choices"] is None
        ):
            raise ValueError(f"{spec.slug} is not a single-view full-image configuration")

    if spec.preprocessing_mode == "fixed_crop":
        train_policies = config.get("multi_view_crop_policy_choices") or []
        eval_policies = config.get("eval_crop_policy_choices") or []
        fixed_crop_ok = (
            config.get("use_crop_as_input") is True
            and config.get("input_source") == "crop"
            and config.get("multi_view_crop") is True
            and train_policies == FIXED_TRAIN_POLICIES
            and eval_policies == FIXED_EVAL_POLICIES
        )
        if not fixed_crop_ok:
            raise ValueError(
                f"{spec.slug} does not contain the six training and three evaluation fixed crops"
            )

    if spec.preprocessing_mode == "adaptive_crop":
        train_policies = config.get("multi_view_crop_policy_choices") or []
        eval_policies = config.get("eval_crop_policy_choices") or []
        adaptive_ok = (
            config.get("use_crop_as_input") is True
            and config.get("input_source") == "crop"
            and config.get("multi_view_crop") is True
            and train_policies == ADAPTIVE_POLICIES
            and eval_policies == ADAPTIVE_POLICIES
        )
        if not adaptive_ok:
            raise ValueError(
                f"{spec.slug} does not contain the six code-defined adaptive crops"
            )

    alternative_slugs = {
        "adaptive_margin_softwrong_ungated_kl020",
        "adaptive_ungated_kl_mild_aug",
        "adaptive_gated_kl_mild_aug",
    }
    if spec.slug in alternative_slugs:
        expected = {
            "main_loss_type": "avg_margin_hinge",
            "main_margin_target": 0.10,
            "wrong_crop_count_weight": 0.30,
            "wrong_crop_count_temperature": 0.15,
            "consistency_weight": 0.20,
            "consistency_loss_type": "kl_probs",
        }
        mismatches = {
            key: (value, config.get(key))
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{spec.slug} alternative objective mismatch: {mismatches}")

    if spec.slug == "adaptive_gated_kl_mild_aug":
        if not (
            config.get("consistency_gate_mode") == "true_margin"
            and config.get("consistency_gate_temperature") == 0.05
            and config.get("consistency_gate_margin_threshold") == 0.0
        ):
            raise ValueError("Gated KL must use the detached true-margin gate at T=0.05")
    elif config.get("consistency_gate_mode") != "none":
        raise ValueError(f"{spec.slug} must not enable a KL gate")


def validate_requested_items(args: argparse.Namespace) -> list[ConfigurationSpec]:
    invalid_folds = sorted(set(args.folds) - set(range(N_FOLDS)))
    if invalid_folds:
        raise ValueError(f"Folds must be between 0 and {N_FOLDS - 1}: {invalid_folds}")
    by_slug = {spec.slug: spec for spec in CONFIGURATIONS}
    unknown = [slug for slug in args.configs if slug not in by_slug]
    if unknown:
        raise ValueError(f"Unknown configuration slugs: {unknown}")
    return [by_slug[slug] for slug in args.configs]


def validate_saved_split_assignments(splits) -> None:
    manifest_path = REPO_ROOT / "splits" / "report_five_fold_assignments.csv"
    expected = pd.read_csv(manifest_path)[
        ["split_idx", "split", "image_id", "lesion_id"]
    ].copy()
    actual_frames = []
    for fold_idx, fold in enumerate(splits):
        for partition in ("train", "val", "test"):
            frame = getattr(fold, f"{partition}_df")[["image_id", "lesion_id"]].copy()
            frame.insert(0, "split", partition)
            frame.insert(0, "split_idx", fold_idx)
            actual_frames.append(frame)
    actual = pd.concat(actual_frames, ignore_index=True)
    columns = ["split_idx", "split", "image_id", "lesion_id"]
    expected = expected.sort_values(columns).reset_index(drop=True).astype(str)
    actual = actual.sort_values(columns).reset_index(drop=True).astype(str)
    if not actual.equals(expected):
        raise RuntimeError(f"Generated five-fold assignments do not match {manifest_path}")


def is_per_image_array(array: np.ndarray, n_images: int) -> bool:
    return hasattr(array, "shape") and len(array.shape) >= 1 and len(array) == n_images


def make_combined_mask_npz(output_path: Path) -> Path:
    """Combine the original train/val/test mask packs for changing CV folds."""
    if output_path.exists():
        return output_path
    for source_path in MASK_SOURCE_PATHS:
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source mask file: {source_path}")

    payloads: list[dict[str, np.ndarray]] = []
    key_sets: list[set[str]] = []
    for source_path in MASK_SOURCE_PATHS:
        with np.load(source_path, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        if "image_ids" not in payload:
            raise KeyError(f"image_ids is missing from {source_path}")
        n_images = len(payload["image_ids"])
        payloads.append(payload)
        key_sets.append(
            {key for key, array in payload.items() if is_per_image_array(array, n_images)}
        )

    allowed = {
        "masks",
        "image_ids",
        "labels",
        "row_idx",
        "has_majority_flag",
        "mean_teacher_confidence",
    }
    merge_keys = set.intersection(*key_sets) & allowed
    required = {"masks", "image_ids", "labels", "row_idx"}
    missing = sorted(required - merge_keys)
    if missing:
        raise KeyError(f"Combined mask pack cannot be built; missing keys: {missing}")

    merged = {
        key: np.concatenate([payload[key] for payload in payloads], axis=0)
        for key in sorted(merge_keys)
    }
    _, unique_indices = np.unique(merged["image_ids"].astype(str), return_index=True)
    unique_indices = np.sort(unique_indices)
    merged = {key: array[unique_indices] for key, array in merged.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)
    return output_path


def resolve_prediction_columns(predictions: pd.DataFrame) -> tuple[str, str, str]:
    true_col = "true_dx" if "true_dx" in predictions else "dx"
    pred_col = "pred_dx" if "pred_dx" in predictions else "y_pred"
    lesion_col = "lesion_id"
    missing = [
        name
        for name, column in {
            "true label": true_col,
            "predicted label": pred_col,
            "lesion identifier": lesion_col,
        }.items()
        if column not in predictions
    ]
    if missing:
        raise KeyError(f"Prediction file is missing required fields: {missing}")
    return true_col, pred_col, lesion_col


def calculate_result_tables(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    true_col, pred_col, lesion_col = resolve_prediction_columns(predictions)
    df = predictions.copy()
    df[true_col] = df[true_col].astype(str).str.lower()
    df[pred_col] = df[pred_col].astype(str).str.lower()
    labels = [label for label in CLASS_ORDER if label in set(df[true_col])]
    unexpected = sorted(set(df[true_col]) - set(labels))
    if unexpected:
        raise ValueError(f"Unexpected true diagnostic labels: {unexpected}")

    per_class = (
        df.assign(is_correct=df[true_col] == df[pred_col])
        .groupby(true_col, observed=True)
        .agg(n_images=(pred_col, "size"), n_correct=("is_correct", "sum"))
        .reindex(labels)
    )
    per_class["per_class_accuracy"] = per_class["n_correct"] / per_class["n_images"]
    per_class = per_class.reset_index().rename(columns={true_col: "class"})

    accuracy = float((df[true_col] == df[pred_col]).mean())
    balanced_accuracy = float(per_class["per_class_accuracy"].mean())

    cm_counts = pd.DataFrame(
        confusion_matrix(df[true_col], df[pred_col], labels=labels),
        index=pd.Index(labels, name="true_class"),
        columns=pd.Index(labels, name="predicted_class"),
    )
    cm_normalized = cm_counts.div(cm_counts.sum(axis=1).replace(0, np.nan), axis=0)

    lesion_rows = []
    for (lesion_id, true_class), group in df.groupby([lesion_col, true_col], observed=True):
        if len(group) < 2:
            continue
        counts = group[pred_col].value_counts()
        lesion_rows.append(
            {
                "lesion_id": lesion_id,
                "true_class": true_class,
                "n_images": len(group),
                "modal_prediction": counts.index[0],
                "modal_prediction_count": int(counts.iloc[0]),
                "consistency": float(counts.iloc[0] / len(group)),
                "all_predictions": ", ".join(group[pred_col].astype(str)),
            }
        )
    lesion_consistency = pd.DataFrame(lesion_rows)
    if lesion_consistency.empty:
        consistency_overall = np.nan
        consistency_by_class = pd.DataFrame(
            columns=["class", "n_multi_image_lesions", "lesion_consistency"]
        )
    else:
        consistency_overall = float(lesion_consistency["consistency"].mean())
        consistency_by_class = (
            lesion_consistency.groupby("true_class", observed=True)
            .agg(
                n_multi_image_lesions=("lesion_id", "size"),
                lesion_consistency=("consistency", "mean"),
            )
            .reindex(labels)
            .reset_index()
            .rename(columns={"true_class": "class"})
        )

    errors = df.loc[df[true_col] != df[pred_col]].copy()
    if errors.empty:
        common_errors = pd.DataFrame(
            columns=[
                "true_class",
                "predicted_class",
                "error_count",
                "pct_of_all_errors",
                "pct_within_true_class",
            ]
        )
    else:
        common_errors = (
            errors.groupby([true_col, pred_col], observed=True)
            .size()
            .rename("error_count")
            .reset_index()
            .rename(columns={true_col: "true_class", pred_col: "predicted_class"})
        )
        common_errors["pct_of_all_errors"] = (
            common_errors["error_count"] / len(errors)
        )
        true_counts = df[true_col].value_counts()
        common_errors["pct_within_true_class"] = common_errors.apply(
            lambda row: row["error_count"] / true_counts[row["true_class"]], axis=1
        )
        common_errors = common_errors.sort_values(
            ["error_count", "true_class", "predicted_class"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    summary_values: dict[str, Any] = {
        "n_images": len(df),
        "n_lesions": df[lesion_col].nunique(),
        "n_multi_image_lesions": len(lesion_consistency),
        "n_correct": int((df[true_col] == df[pred_col]).sum()),
        "n_errors": int((df[true_col] != df[pred_col]).sum()),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "lesion_consistency": consistency_overall,
    }
    for class_name, _, _, class_accuracy in per_class.itertuples(index=False, name=None):
        summary_values[f"accuracy_{class_name}"] = class_accuracy
    for class_name, _, class_consistency in consistency_by_class.itertuples(
        index=False, name=None
    ):
        summary_values[f"consistency_{class_name}"] = class_consistency

    summary = pd.DataFrame([summary_values])
    return {
        "summary": summary,
        "per_class_accuracy": per_class,
        "confusion_counts": cm_counts,
        "confusion_normalized": cm_normalized,
        "consistency_by_class": consistency_by_class,
        "lesion_consistency": lesion_consistency,
        "common_errors": common_errors,
    }


def save_confusion_figure(cm: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(cm.values, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(cm.columns)), labels=cm.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(cm.index)), labels=cm.index)
    axis.set_xlabel("Predicted category")
    axis.set_ylabel("True category")
    axis.set_title(title)
    for row in range(len(cm.index)):
        for column in range(len(cm.columns)):
            value = cm.iloc[row, column]
            axis.text(
                column,
                row,
                "" if pd.isna(value) else f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.5 else "black",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_prediction_and_result_artifacts(
    predictions_path: Path,
    output_dir: Path,
    title: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    predictions = pd.read_csv(predictions_path)
    tables = calculate_result_tables(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_excel = output_dir / "test_predictions.xlsx"
    predictions.to_excel(predictions_excel, index=False, engine="openpyxl")

    for table_name, table in tables.items():
        table.to_csv(output_dir / f"{table_name}.csv", index=True)

    workbook_path = output_dir / "fold_results.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        tables["summary"].to_excel(writer, sheet_name="summary", index=False)
        tables["per_class_accuracy"].to_excel(
            writer, sheet_name="per_class_accuracy", index=False
        )
        tables["confusion_counts"].to_excel(writer, sheet_name="confusion_counts")
        tables["confusion_normalized"].to_excel(
            writer, sheet_name="confusion_normalized"
        )
        tables["consistency_by_class"].to_excel(
            writer, sheet_name="consistency_by_class", index=False
        )
        tables["lesion_consistency"].to_excel(
            writer, sheet_name="lesion_consistency", index=False
        )
        tables["common_errors"].to_excel(
            writer, sheet_name="common_errors", index=False
        )

    save_confusion_figure(
        tables["confusion_normalized"],
        output_dir / "confusion_matrix_normalized.png",
        title,
    )
    return predictions, tables


def find_training_artifacts(
    fold_dir: Path, summary_path: Path
) -> tuple[Path, pd.DataFrame]:
    training_summary = pd.read_csv(summary_path)
    if training_summary.empty:
        raise RuntimeError(f"Training summary is empty: {summary_path}")
    summary_row = training_summary.iloc[0]

    candidate = summary_row.get("test_preds_path")
    if isinstance(candidate, str) and Path(candidate).exists():
        return Path(candidate), training_summary

    candidates = sorted(fold_dir.rglob("test_predictions.csv"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one test_predictions.csv under {fold_dir}, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0], training_summary


def add_training_summary_metrics(
    result_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
) -> pd.DataFrame:
    output = result_summary.copy()
    row = training_summary.iloc[0]
    for source, target in [
        ("best_metric_value", "best_validation_selection_metric"),
        ("val_accuracy", "validation_accuracy"),
        ("val_balanced_accuracy", "validation_balanced_accuracy"),
        ("val_macro_f1", "validation_macro_f1"),
        ("test_macro_f1", "test_macro_f1"),
    ]:
        if source in training_summary.columns:
            output[target] = row[source]
    return output


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def completion_marker_path(fold_dir: Path) -> Path:
    return fold_dir / "_FOLD_COMPLETED.json"


def load_completed_fold(fold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    marker_path = completion_marker_path(fold_dir)
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    predictions_path = Path(marker["predictions_csv"])
    metrics_path = Path(marker["metrics_csv"])
    if not predictions_path.exists() or not metrics_path.exists():
        return None
    return pd.read_csv(predictions_path), pd.read_csv(metrics_path)


def mark_fold_complete(
    fold_dir: Path,
    predictions_path: Path,
    metrics_path: Path,
    configuration_slug: str,
) -> None:
    marker = {
        "predictions_csv": str(predictions_path.resolve()),
        "metrics_csv": str(metrics_path.resolve()),
        "configuration_slug": configuration_slug,
        "configuration_source": "report_experiment_configs.py",
    }
    temporary = fold_dir / "_FOLD_COMPLETED.tmp"
    temporary.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    os.replace(temporary, completion_marker_path(fold_dir))


def archive_incomplete_fold(fold_dir: Path, config_dir: Path) -> Path:
    """Preserve a partial attempt, then allow only that fold to restart."""
    fold_dir = fold_dir.resolve()
    config_dir = config_dir.resolve()
    if fold_dir.parent != config_dir or not fold_dir.name.startswith("fold_"):
        raise RuntimeError(f"Refusing to move unexpected directory: {fold_dir}")
    if not fold_dir.exists():
        raise FileNotFoundError(f"Incomplete fold directory does not exist: {fold_dir}")

    archive_root = config_dir / "_incomplete_attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = archive_root / f"{fold_dir.name}_{timestamp}"
    suffix = 1
    while destination.exists():
        destination = archive_root / f"{fold_dir.name}_{timestamp}_{suffix}"
        suffix += 1
    shutil.move(str(fold_dir), str(destination))
    return destination


def format_mean_sd(mean: float, std: float, metric: str) -> str:
    if pd.isna(mean):
        return ""
    if metric.startswith(("accuracy", "balanced_accuracy", "lesion_consistency", "consistency_")):
        return f"{100 * mean:.2f}% ± {100 * std:.2f}%"
    return f"{mean:.4f} ± {std:.4f}"


def aggregate_configuration(
    spec: ConfigurationSpec,
    config_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]] | None:
    fold_metrics = []
    pooled_predictions = []
    for fold_idx in range(N_FOLDS):
        fold_dir = config_dir / f"fold_{fold_idx}"
        completed = load_completed_fold(fold_dir)
        if completed is None:
            return None
        predictions, metrics = completed
        metrics["fold"] = fold_idx
        metrics["configuration"] = spec.slug
        metrics["display_name"] = spec.display_name
        leading = ["configuration", "display_name", "fold"]
        metrics = metrics[leading + [column for column in metrics if column not in leading]]
        fold_metrics.append(metrics)
        predictions = predictions.copy()
        predictions.insert(0, "fold", fold_idx)
        predictions.insert(0, "configuration", spec.slug)
        pooled_predictions.append(predictions)

    fold_metrics_df = pd.concat(fold_metrics, ignore_index=True)
    pooled_predictions_df = pd.concat(pooled_predictions, ignore_index=True)
    if "image_id" in pooled_predictions_df:
        duplicates = pooled_predictions_df["image_id"].duplicated().sum()
        if duplicates:
            raise RuntimeError(
                f"{spec.slug}: pooled out-of-fold predictions contain "
                f"{duplicates} duplicate image IDs."
            )

    pooled_tables = calculate_result_tables(pooled_predictions_df)
    config_dir.mkdir(parents=True, exist_ok=True)
    pooled_predictions_df.to_csv(config_dir / "pooled_oof_predictions.csv", index=False)
    pooled_predictions_df.to_excel(
        config_dir / "pooled_oof_predictions.xlsx", index=False, engine="openpyxl"
    )
    with pd.ExcelWriter(config_dir / "pooled_oof_results.xlsx", engine="openpyxl") as writer:
        for name, table in pooled_tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=name.startswith("confusion_"))
    save_confusion_figure(
        pooled_tables["confusion_normalized"],
        config_dir / "pooled_confusion_matrix_normalized.png",
        f"{spec.display_name} — pooled out-of-fold confusion matrix",
    )

    numeric_metrics = [
        column
        for column in fold_metrics_df.columns
        if column not in {"configuration", "fold"}
        and pd.api.types.is_numeric_dtype(fold_metrics_df[column])
    ]
    means = fold_metrics_df[numeric_metrics].mean()
    stds = fold_metrics_df[numeric_metrics].std(ddof=1)
    mean_sd = pd.DataFrame(
        {
            metric: [format_mean_sd(means[metric], stds[metric], metric)]
            for metric in numeric_metrics
        }
    )
    mean_sd.insert(0, "display_name", spec.display_name)
    mean_sd.insert(0, "configuration", spec.slug)

    numeric_summary = pd.DataFrame(
        {
            "configuration": spec.slug,
            "display_name": spec.display_name,
            "metric": numeric_metrics,
            "mean": [means[metric] for metric in numeric_metrics],
            "sd": [stds[metric] for metric in numeric_metrics],
        }
    )
    return fold_metrics_df, pooled_predictions_df, {
        **pooled_tables,
        "mean_sd": mean_sd,
        "numeric_summary": numeric_summary,
    }


def write_final_summary(
    completed: list[
        tuple[ConfigurationSpec, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]
    ],
    results_root: Path,
) -> None:
    all_fold_metrics = pd.concat([item[1] for item in completed], ignore_index=True)
    mean_sd = pd.concat([item[3]["mean_sd"] for item in completed], ignore_index=True)
    numeric_summary = pd.concat(
        [item[3]["numeric_summary"] for item in completed], ignore_index=True
    )

    pooled_metric_rows = []
    pooled_per_class = []
    pooled_consistency = []
    pooled_errors = []
    for spec, _, _, tables in completed:
        summary = tables["summary"].copy()
        summary.insert(0, "display_name", spec.display_name)
        summary.insert(0, "configuration", spec.slug)
        pooled_metric_rows.append(summary)

        per_class = tables["per_class_accuracy"].copy()
        per_class.insert(0, "configuration", spec.slug)
        pooled_per_class.append(per_class)

        consistency = tables["consistency_by_class"].copy()
        consistency.insert(0, "configuration", spec.slug)
        pooled_consistency.append(consistency)

        errors = tables["common_errors"].copy()
        errors.insert(0, "configuration", spec.slug)
        pooled_errors.append(errors)

    pooled_metrics = pd.concat(pooled_metric_rows, ignore_index=True)
    pooled_per_class_df = pd.concat(pooled_per_class, ignore_index=True)
    pooled_consistency_df = pd.concat(pooled_consistency, ignore_index=True)
    pooled_errors_df = pd.concat(pooled_errors, ignore_index=True)

    validation_ranking = (
        numeric_summary.loc[
            numeric_summary["metric"] == "validation_balanced_accuracy",
            ["configuration", "display_name", "mean", "sd"],
        ]
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )
    if not validation_ranking.empty:
        validation_ranking.insert(0, "validation_rank", range(1, len(validation_ranking) + 1))
        validation_ranking["selected_by_validation"] = False
        validation_ranking.loc[0, "selected_by_validation"] = True
        (results_root / "selected_configuration.json").write_text(
            json.dumps(
                {
                    "selection_metric": "mean_validation_balanced_accuracy",
                    "configuration": validation_ranking.loc[0, "configuration"],
                    "mean": float(validation_ranking.loc[0, "mean"]),
                    "sample_sd": float(validation_ranking.loc[0, "sd"]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    all_fold_metrics.to_csv(results_root / "all_fold_metrics.csv", index=False)
    mean_sd.to_csv(results_root / "configuration_mean_sd.csv", index=False)
    numeric_summary.to_csv(results_root / "configuration_numeric_summary.csv", index=False)
    pooled_metrics.to_csv(results_root / "pooled_oof_metrics.csv", index=False)
    validation_ranking.to_csv(results_root / "validation_selection_ranking.csv", index=False)

    workbook = results_root / "final_comparison_summary.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        mean_sd.to_excel(writer, sheet_name="mean_sd", index=False)
        numeric_summary.to_excel(writer, sheet_name="numeric_summary", index=False)
        all_fold_metrics.to_excel(writer, sheet_name="fold_metrics", index=False)
        pooled_metrics.to_excel(writer, sheet_name="pooled_metrics", index=False)
        pooled_per_class_df.to_excel(writer, sheet_name="pooled_per_class", index=False)
        pooled_consistency_df.to_excel(
            writer, sheet_name="pooled_consistency", index=False
        )
        pooled_errors_df.to_excel(writer, sheet_name="pooled_common_errors", index=False)
        validation_ranking.to_excel(
            writer, sheet_name="validation_selection", index=False
        )


def main() -> None:
    args = parse_args()
    selected_specs = validate_requested_items(args)
    results_root = args.results_root.resolve()

    effective_base_configs = {
        spec.slug: get_report_experiment_config(spec.slug) for spec in selected_specs
    }
    for spec in selected_specs:
        validate_preprocessing(spec, effective_base_configs[spec.slug])
        print(f"[CONFIG] {spec.slug}")
        print("  source: Python code in report_experiment_configs.py")
        print(f"  run_tag: {effective_base_configs[spec.slug]['run_tag']}")
        print(f"  input construction: {spec.preprocessing_mode}")
        if spec.preprocessing_mode == "adaptive_crop":
            print("  adaptive occupancies: [10%, 20%, 30%, 40%, 50%, 60%]")
        effective = effective_base_configs[spec.slug]
        print(
            "  effective transforms: "
            f"resize={effective.get('img_size')}, "
            f"rotation={effective.get('rotation_deg')}, "
            f"hflip={effective.get('hflip')}@{effective.get('hflip_p')}, "
            f"vflip={effective.get('vflip')}@{effective.get('vflip_p')}, "
            "color jitter="
            f"{effective.get('color_jitter_brightness')}/"
            f"{effective.get('color_jitter_contrast')}/"
            f"{effective.get('color_jitter_saturation')}/"
            f"{effective.get('color_jitter_hue')}"
        )

    if args.dry_run:
        missing_masks = [str(path) for path in MASK_SOURCE_PATHS if not path.exists()]
        if missing_masks:
            print("Mask generation is still required before full training:")
            for path in missing_masks:
                print(f"  missing: {path}")
        print(f"Configurations: {[spec.slug for spec in selected_specs]}")
        print(f"Requested folds: {args.folds}")
        print(
            "Dry run completed from Python-defined configurations; no dataset, "
            "checkpoint, result TXT, or model training was required."
        )
        return

    results_root.mkdir(parents=True, exist_ok=True)
    meta, _ = load_ham10000_metadata()
    splits_dir = REPO_ROOT / "splits" / SPLIT_SUBDIR
    splits = make_or_load_repeated_train_val_test_splits(
        meta=meta,
        splits_dir=splits_dir,
        split_seed=SPLIT_SEED,
        n_splits=N_FOLDS,
        val_fraction=VAL_FRACTION,
        stratify_col="dx",
        overwrite=False,
        prefix=SPLIT_PREFIX,
    )
    if len(splits) != N_FOLDS:
        raise RuntimeError(f"Expected {N_FOLDS} folds, received {len(splits)}")
    validate_saved_split_assignments(splits)

    print(f"Results root: {results_root}")
    print(f"Configurations: {[spec.slug for spec in selected_specs]}")
    print(f"Folds: {args.folds}")

    combined_mask_path = make_combined_mask_npz(
        splits_dir / f"{SPLIT_PREFIX}_fused_masks_minimal_all.npz"
    )
    print(f"Combined mask pack: {combined_mask_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available in this Python environment. Use the lesion-sam "
            "environment, or pass --allow-cpu only if a CPU run is intentional."
        )
    print(f"Device: {device}")
    run_log: list[dict[str, Any]] = []

    for spec in selected_specs:
        config_dir = results_root / spec.slug
        config_dir.mkdir(parents=True, exist_ok=True)
        base_config = effective_base_configs[spec.slug]
        (config_dir / "code_defined_config.json").write_text(
            json.dumps(json_ready(base_config), indent=2), encoding="utf-8"
        )

        for fold_idx in args.folds:
            fold_dir = config_dir / f"fold_{fold_idx}"
            completed = load_completed_fold(fold_dir)
            if completed is not None:
                print(f"[SKIP] Completed: {spec.slug} fold_{fold_idx}")
                continue
            if fold_dir.exists():
                archived = archive_incomplete_fold(fold_dir, config_dir)
                print(
                    f"[RESUME] Incomplete {spec.slug} fold_{fold_idx} was archived "
                    f"at {archived}; restarting only this fold."
                )

            fold = splits[fold_idx]
            fold_dir.mkdir(parents=True, exist_ok=True)
            meta_features, _, _, meta_feature_dim = build_meta_features_leakage_free(
                meta, fold.train_df["row_idx"].values
            )
            config = copy.deepcopy(base_config)
            config.update(
                {
                    "run_tag": f"{spec.slug}_fold{fold_idx}",
                    "seed": MODEL_SEED,
                    "split_seed": SPLIT_SEED + fold_idx,
                    "mask_npz_train": str(combined_mask_path),
                    "mask_npz_val": str(combined_mask_path),
                    "mask_npz_test": str(combined_mask_path),
                }
            )
            validate_preprocessing(spec, config)
            (fold_dir / "effective_config.json").write_text(
                json.dumps(json_ready(config), indent=2), encoding="utf-8"
            )

            print("=" * 88)
            print(f"[RUN] {spec.display_name} | fold_{fold_idx}")
            print(
                f"train={len(fold.train_df)}, val={len(fold.val_df)}, "
                f"test={len(fold.test_df)}"
            )
            try:
                summary_path = Path(
                    run_efficientnet_experiment(
                        train_df=fold.train_df,
                        val_df=fold.val_df,
                        test_df=fold.test_df,
                        meta_features=meta_features,
                        meta_feature_dim=meta_feature_dim,
                        out_root=fold_dir,
                        device=device,
                        seed=MODEL_SEED,
                        split_seed=SPLIT_SEED + fold_idx,
                        param_grid=[config],
                        no_meta=bool(config.get("no_meta", False)),
                        target_class="mel",
                        early_stopping_metric="balanced_acc",
                    )
                )
                predictions_path, training_summary = find_training_artifacts(
                    fold_dir, summary_path
                )
                predictions, tables = save_prediction_and_result_artifacts(
                    predictions_path,
                    fold_dir,
                    f"{spec.display_name} — fold {fold_idx}",
                )
                enriched_summary = add_training_summary_metrics(
                    tables["summary"], training_summary
                )
                enriched_summary.insert(0, "fold", fold_idx)
                enriched_summary.insert(0, "display_name", spec.display_name)
                enriched_summary.insert(0, "configuration", spec.slug)
                metrics_path = fold_dir / "fold_metrics.csv"
                enriched_summary.to_csv(metrics_path, index=False)
                enriched_summary.to_excel(
                    fold_dir / "fold_metrics.xlsx", index=False, engine="openpyxl"
                )
                canonical_predictions_path = fold_dir / "test_predictions_for_summary.csv"
                predictions.to_csv(canonical_predictions_path, index=False)
                mark_fold_complete(
                    fold_dir,
                    canonical_predictions_path,
                    metrics_path,
                    spec.slug,
                )
                run_log.append(
                    {
                        "configuration": spec.slug,
                        "fold": fold_idx,
                        "status": "SUCCESS",
                        "summary_path": str(summary_path),
                        "predictions_path": str(predictions_path),
                    }
                )
            except Exception as exc:
                error_path = fold_dir / "error.json"
                error_path.write_text(
                    json.dumps(
                        {
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                run_log.append(
                    {
                        "configuration": spec.slug,
                        "fold": fold_idx,
                        "status": f"FAILED: {type(exc).__name__}",
                        "error": str(exc),
                    }
                )
                pd.DataFrame(run_log).to_csv(results_root / "run_log.csv", index=False)
                raise
            finally:
                meta_features = None
                config = None
                cleanup_cuda()

            pd.DataFrame(run_log).to_csv(results_root / "run_log.csv", index=False)

    completed_aggregates = []
    for spec in CONFIGURATIONS:
        aggregated = aggregate_configuration(spec, results_root / spec.slug)
        if aggregated is None:
            print(f"[SUMMARY WAITING] {spec.slug} does not yet have all five folds.")
            continue
        fold_metrics, pooled_predictions, tables = aggregated
        completed_aggregates.append(
            (spec, fold_metrics, pooled_predictions, tables)
        )

    if len(completed_aggregates) == len(CONFIGURATIONS):
        write_final_summary(completed_aggregates, results_root)
        print(f"Final comparison workbook: {results_root / 'final_comparison_summary.xlsx'}")
    else:
        print(
            "Final six-configuration summary was not written because one or more "
            "configurations do not yet have all five completed folds."
        )


if __name__ == "__main__":
    main()
