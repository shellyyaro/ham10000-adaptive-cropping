"""Train the 24 ResNet-18 configurations used by the mask ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resnet_vit_pretrained_comparison.data.data_loader import load_ham10000_metadata
from resnet_vit_pretrained_comparison.data.meta_features import (
    build_meta_features_leakage_free,
)
from resnet_vit_pretrained_comparison.scripts.run_resnet_with_meta import (
    run_resnet_experiment,
)
from resnet_vit_pretrained_comparison.scripts.splits import make_or_load_splits


SERIES = ("regular_baseline", "midlevel_fusion", "filter_size")
EXPECTED_COUNTS = {
    "regular_baseline": 9,
    "midlevel_fusion": 8,
    "filter_size": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the three ResNet ablation series used to create Grad-CAM masks."
    )
    parser.add_argument("--series", nargs="+", choices=SERIES, default=list(SERIES))
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Optional run_tag filter. Repeat to select multiple configurations.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "resnet_ensemble",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when a checkpoint already exists for the run tag.",
    )
    return parser.parse_args()


def load_specs(series_names: list[str], run_tags: set[str]) -> list[tuple[str, Path, dict]]:
    specs: list[tuple[str, Path, dict]] = []
    config_root = REPO_ROOT / "configs" / "resnet_ensemble"
    for series in series_names:
        paths = sorted((config_root / series).glob("*.json"))
        if len(paths) != EXPECTED_COUNTS[series]:
            raise RuntimeError(
                f"Expected {EXPECTED_COUNTS[series]} configs in {series}, found {len(paths)}."
            )
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = payload.get("config")
            if not isinstance(config, dict):
                raise ValueError(f"Missing config object in {path}")
            run_tag = str(config.get("run_tag", ""))
            if not run_tag:
                raise ValueError(f"Missing run_tag in {path}")
            if run_tags and run_tag not in run_tags:
                continue
            specs.append((series, path, config))
    if run_tags:
        found = {config["run_tag"] for _, _, config in specs}
        missing = sorted(run_tags - found)
        if missing:
            raise ValueError(f"Unknown run_tag values: {missing}")
    return specs


def validate_fixed_split_manifest(splits) -> None:
    manifest_path = REPO_ROOT / "splits" / "mask_generation_fixed_split.csv"
    manifest = pd.read_csv(manifest_path)
    for partition in ("train", "val", "test"):
        expected = set(
            map(
                tuple,
                manifest.loc[manifest["split"] == partition, ["image_id", "lesion_id"]]
                .astype(str)
                .to_numpy(),
            )
        )
        frame = getattr(splits, f"{partition}_df")
        actual = set(map(tuple, frame[["image_id", "lesion_id"]].astype(str).to_numpy()))
        if actual != expected:
            raise RuntimeError(f"Generated {partition} split does not match {manifest_path}")


def main() -> None:
    args = parse_args()
    specs = load_specs(args.series, set(args.config))
    print(f"Validated {len(specs)} ResNet configurations.")
    for series, path, config in specs:
        print(f"  {series:18s} {config['run_tag']}  ({path.name})")
    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Pass --allow-cpu only for an intentional CPU run.")

    metadata, _ = load_ham10000_metadata()
    splits = make_or_load_splits(
        meta=metadata,
        splits_dir=REPO_ROOT / "splits" / "mask_generation_fixed",
        split_seed=0,
        stratify_col="dx",
        overwrite=False,
        prefix="ham10000_mask_generation",
    )
    validate_fixed_split_manifest(splits)
    meta_features, _, _, meta_feature_dim = build_meta_features_leakage_free(
        metadata,
        splits.train_df["row_idx"].values,
    )

    for series, _, config in specs:
        output_root = args.results_root / series
        run_tag = str(config["run_tag"])
        existing = list((output_root / run_tag).rglob("*.pt"))
        if existing and not args.force:
            print(f"[SKIP] {run_tag}: checkpoint already exists")
            continue

        param_grid = {key: [value] for key, value in config.items()}
        run_resnet_experiment(
            train_df=splits.train_df,
            val_df=splits.val_df,
            test_df=splits.test_df,
            meta_features=meta_features,
            meta_feature_dim=meta_feature_dim,
            out_root=output_root,
            device=device,
            seed=0,
            split_seed=0,
            param_grid=param_grid,
            no_meta=bool(config.get("no_meta", False)),
            target_class="mel",
            early_stopping_metric="balanced_acc",
        )


if __name__ == "__main__":
    main()
