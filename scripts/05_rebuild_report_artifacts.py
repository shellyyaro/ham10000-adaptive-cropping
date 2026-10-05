"""Rebuild every data-derived figure and table embedded in the final report.

Figures 1 and 2 require the HAM10000 images and generated fused-mask NPZ files.
Figure 3 and Tables 1-2 rebuild from the compact, versioned OOF result files.
Table 3 is a curated literature comparison and is copied from its versioned CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from resnet_vit_pretrained_comparison.data.adaptive_crop_policy import (  # noqa: E402
    adaptive_crop_for_target_ratio,
)


CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CONFIGURATIONS = [
    ("full_image", "Full-image baseline"),
    ("fixed_crop", "Fixed crop"),
    ("adaptive_baseline", "Adaptive baseline"),
    ("adaptive_ungated_kl_common_aug", "Adaptive, ungated KL\ncommon augmentation"),
    ("adaptive_ungated_kl_mild_aug", "Adaptive, ungated KL\nmild augmentation"),
    ("adaptive_gated_kl_mild_aug", "Adaptive, gated KL\nmild augmentation (final model)"),
]
REPORT_MEDIA = {
    "image1.png": "report_cover_logo.png",
    "image2.png": "figure_1_mask_area_distribution.png",
    "image3.png": "figure_2_crop_examples.png",
    "image4.png": "figure_3_confusion_matrices.png",
}
REPORT_TABLE_NAMES = {
    "full_image_efficientnet_baseline": "Full image",
    "fixed_crop_viewce030_msecons005_baseline": "Fixed crop",
    "adaptive_crop_viewce030_msecons005_baseline": "Adaptive baseline",
    "adaptive_margin_softwrong_ungated_kl020": "Adaptive + alternative objective (ungated KL, common aug.)",
    "adaptive_ungated_kl_mild_aug": "Adaptive + alternative objective (ungated KL, mild aug.)",
    "adaptive_gated_kl_mild_aug": "Adaptive + alternative objective (gated KL, mild aug.)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "report_artifacts" / "generated",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="HAM10000 directory containing metadata and image-part folders",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        help="Directory containing fused_masks_train/val/test.npz",
    )
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Build result-only artifacts when the dataset or masks are unavailable",
    )
    parser.add_argument(
        "--verify-report-media",
        action="store_true",
        help="Verify that checked-in canonical figures equal the report media byte-for-byte",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_report_media() -> dict[str, str]:
    report = ROOT / "reports" / "final_report.docx"
    canonical = ROOT / "results" / "report_artifacts" / "canonical"
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(report) as archive:
        for media_name, checked_in_name in REPORT_MEDIA.items():
            report_bytes = archive.read(f"word/media/{media_name}")
            checked_in_bytes = (canonical / checked_in_name).read_bytes()
            if report_bytes != checked_in_bytes:
                raise RuntimeError(
                    f"Report media {media_name} differs from {checked_in_name}"
                )
            hashes[checked_in_name] = sha256_bytes(checked_in_bytes)
    return hashes


def build_table_1(output_dir: Path) -> Path:
    source = ROOT / "results" / "reported_metrics" / "five_fold_mean_sd.csv"
    frame = pd.read_csv(source)
    def ratio_to_percent(value: str) -> str:
        mean, sample_sd = (part.strip() for part in value.split("±"))
        return f"{float(mean) * 100:.2f}% ± {float(sample_sd) * 100:.2f}%"

    frame["display_name"] = frame["configuration"].map(REPORT_TABLE_NAMES)
    frame["macro_f1"] = frame["test_macro_f1"].map(ratio_to_percent)
    frame["validation_balanced_accuracy"] = frame[
        "validation_balanced_accuracy"
    ].map(ratio_to_percent)
    wanted = [
        "configuration",
        "display_name",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "lesion_consistency",
        "validation_balanced_accuracy",
    ]
    out = output_dir / "table_1_five_fold_performance.csv"
    frame[wanted].to_csv(out, index=False)
    return out


def build_table_2(output_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    result_root = ROOT / "results" / "reported_metrics"
    for slug, display_name in CONFIGURATIONS:
        matrix = pd.read_csv(
            result_root / slug / "confusion_matrix_normalized.csv", index_col=0
        ).loc[CLASS_ORDER, CLASS_ORDER]
        row: dict[str, object] = {
            "configuration": slug,
            "display_name": display_name.replace("\n", " "),
        }
        row.update(
            {
                label.upper(): f"{float(matrix.loc[label, label]) * 100.0:.2f}"
                for label in CLASS_ORDER
            }
        )
        rows.append(row)
    out = output_dir / "table_2_pooled_class_recall.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def build_table_3(output_dir: Path) -> Path:
    source = ROOT / "results" / "report_artifacts" / "table_3_literature_comparison.csv"
    out = output_dir / source.name
    out.write_bytes(source.read_bytes())
    return out


def plot_confusion_matrix_grid(output_dir: Path) -> Path:
    result_root = ROOT / "results" / "reported_metrics"
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.93), squeeze=False)
    for ax, (slug, title) in zip(axes.flat, CONFIGURATIONS):
        matrix = pd.read_csv(
            result_root / slug / "confusion_matrix_normalized.csv", index_col=0
        ).loc[CLASS_ORDER, CLASS_ORDER]
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontfamily="serif", fontweight="bold", fontsize=12)
        ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("True category")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value >= 0.50 else "black",
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = output_dir / "figure_3_confusion_matrices.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def load_mask_archive(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return {name: archive[name] for name in archive.files}


def plot_mask_area_distribution(mask_root: Path, output_dir: Path) -> Path:
    archive = load_mask_archive(mask_root / "fused_masks_train.npz")
    masks = archive["masks"].astype(np.float32)
    areas = (masks >= 0.55).sum(axis=(1, 2)) / (masks.shape[1] * masks.shape[2]) * 100
    frame = pd.DataFrame(
        {
            "image_id": archive["image_ids"].astype(str),
            "majority_labels": archive["majority_labels"].astype(str),
            "mask_area_percent": areas,
        }
    )
    bins = list(range(0, 101, 10))
    labels = [f"{bins[index]}-{bins[index + 1]}" for index in range(10)]
    frame["mask_area_bin"] = pd.cut(
        frame["mask_area_percent"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    frame.loc[frame["mask_area_percent"] == 100, "mask_area_bin"] = "90-100"
    counts = pd.crosstab(frame["mask_area_bin"], frame["majority_labels"], dropna=False)
    percentages = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    ax = percentages.plot(kind="bar", stacked=True, figsize=(10.5, 8))
    ax.set_ylabel("Percent within mask-area bin")
    ax.set_xlabel("Mask area percent bin")
    ax.set_title(
        "TRAIN - Majority-label distribution by mask area bin\n"
        "Text inside segment = % of images in that mask-area bin"
    )
    ax.legend(title="majority_labels", bbox_to_anchor=(1.02, 1), loc="upper left")
    for x_index, bin_name in enumerate(percentages.index):
        bottom = 0.0
        for label in percentages.columns:
            height = float(percentages.loc[bin_name, label])
            if height >= 1 and int(counts.loc[bin_name, label]) > 0:
                ax.text(
                    x_index,
                    bottom + height / 2,
                    f"{height:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            bottom += height
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    plt.tight_layout()
    out = output_dir / "figure_1_mask_area_distribution.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def locate_image(data_root: Path, image_id: str) -> Path:
    candidates = [
        data_root / "HAM10000_images_part_1" / f"{image_id}.jpg",
        data_root / "HAM10000_images_part_2" / f"{image_id}.jpg",
        data_root / f"{image_id}.jpg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate {image_id}.jpg under {data_root}")


def fixed_bbox(mask: np.ndarray, threshold: float = 0.30, margin: float = 0.05) -> tuple[int, int, int, int]:
    binary = np.asarray(mask) >= threshold
    ys = np.flatnonzero(binary.any(axis=1))
    xs = np.flatnonzero(binary.any(axis=0))
    if not len(xs) or not len(ys):
        return (0, 0, mask.shape[1], mask.shape[0])
    x1, x2 = int(xs[0]), int(xs[-1]) + 1
    y1, y2 = int(ys[0]), int(ys[-1]) + 1
    mx = int(round((x2 - x1) * margin))
    my = int(round((y2 - y1) * margin))
    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(mask.shape[1], x2 + mx),
        min(mask.shape[0], y2 + my),
    )


def crop_resize(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(bbox).resize((224, 224), Image.Resampling.LANCZOS)


def overlay_mask_and_boxes(
    image: Image.Image,
    mask: np.ndarray,
    fixed: tuple[int, int, int, int],
    adaptive: tuple[int, int, int, int],
) -> Image.Image:
    base = image.copy().convert("RGB")
    mask_image = Image.fromarray(((mask >= 0.60) * 255).astype(np.uint8), mode="L")
    mask_image = mask_image.resize(base.size, Image.Resampling.NEAREST)
    tint = Image.new("RGB", base.size, (235, 64, 52))
    base = Image.composite(Image.blend(base, tint, 0.34), base, mask_image)
    draw = ImageDraw.Draw(base)
    scale_x, scale_y = base.width / mask.shape[1], base.height / mask.shape[0]
    for box, color in ((fixed, (255, 204, 0)), (adaptive, (0, 190, 255))):
        x1, y1, x2, y2 = box
        draw.rectangle(
            (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
            outline=color,
            width=4,
        )
    return base


def plot_crop_examples(data_root: Path, mask_root: Path, output_dir: Path) -> Path:
    wanted = ["ISIC_0026236", "ISIC_0032692"]
    masks_by_id: dict[str, np.ndarray] = {}
    labels_by_id: dict[str, str] = {}
    for split in ("train", "val", "test"):
        archive = load_mask_archive(mask_root / f"fused_masks_{split}.npz")
        labels = archive.get("true_labels", archive.get("majority_labels"))
        for image_id, mask, label in zip(archive["image_ids"], archive["masks"], labels):
            decoded = image_id.decode() if isinstance(image_id, bytes) else str(image_id)
            if decoded in wanted:
                masks_by_id[decoded] = np.asarray(mask)
                labels_by_id[decoded] = label.decode() if isinstance(label, bytes) else str(label)
    missing = sorted(set(wanted) - set(masks_by_id))
    if missing:
        raise FileNotFoundError(f"Representative masks missing for: {missing}")

    descriptions = [
        "Small lesion with extensive background",
        "Large lesion occupying most of the frame",
    ]
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.55), squeeze=False)
    for row_index, (image_id, description) in enumerate(zip(wanted, descriptions)):
        image = Image.open(locate_image(data_root, image_id)).convert("RGB").resize(
            (224, 224), Image.Resampling.LANCZOS
        )
        mask = masks_by_id[image_id]
        fixed = fixed_bbox(mask)
        adaptive_01 = adaptive_crop_for_target_ratio(
            mask, target_ratio=0.10, image_size=(224, 224), mask_threshold=0.60, min_mask_area=10
        )
        adaptive_03 = adaptive_crop_for_target_ratio(
            mask, target_ratio=0.30, image_size=(224, 224), mask_threshold=0.60, min_mask_area=10
        )
        panels = [
            image,
            overlay_mask_and_boxes(image, mask, fixed, adaptive_03.bbox),
            crop_resize(image, fixed),
            crop_resize(image, adaptive_01.bbox),
            crop_resize(image, adaptive_03.bbox),
        ]
        titles = [
            "Full image",
            "Lesion mask and crop windows",
            "Fixed crop",
            "Adaptive target 0.1",
            "Adaptive target 0.3",
        ]
        for column, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[row_index, column]
            ax.imshow(panel)
            ax.axis("off")
            if row_index == 0:
                ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        axes[row_index, 0].text(
            0.02,
            0.98,
            f"{image_id} ({labels_by_id[image_id].upper()})\n{description}",
            transform=axes[row_index, 0].transAxes,
            va="top",
            fontsize=8.5,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.5},
        )
    fig.suptitle(
        "Representative effects of lesion cropping\n"
        "Yellow: fixed crop; cyan: adaptive crop at target mask-area ratio 0.3",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.91), h_pad=1.25, w_pad=0.8)
    out = output_dir / "figure_2_crop_examples.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("Always available: Figure 3, Tables 1-3, report-media verification")
        print(f"Figure 1-2 require data_root={args.data_root} mask_root={args.mask_root}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_table_1(args.output_dir),
        build_table_2(args.output_dir),
        build_table_3(args.output_dir),
        plot_confusion_matrix_grid(args.output_dir),
    ]

    data_ready = args.data_root is not None and args.mask_root is not None
    if data_ready:
        outputs.append(plot_mask_area_distribution(args.mask_root, args.output_dir))
        outputs.append(plot_crop_examples(args.data_root, args.mask_root, args.output_dir))
    elif not args.available_only:
        raise SystemExit(
            "Figures 1-2 require --data-root and --mask-root. "
            "Use --available-only to rebuild only versioned-data artifacts."
        )

    media_hashes = verify_report_media() if args.verify_report_media else {}
    manifest = {
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "canonical_report_media_sha256": media_hashes,
        "data_figures_rebuilt": data_ready,
    }
    (args.output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
