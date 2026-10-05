# resnet_vit_pretrained_comparison/data/data_loader.py
from __future__ import annotations

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd




def _repo_root() -> Path:
    """
    Returns repository root assuming this file is:
      <repo_root>/resnet_vit_pretrained_comparison/data/data_loader.py
    """
    # This submission uses a ``src/`` layout:
    # <repo_root>/src/resnet_vit_pretrained_comparison/data/data_loader.py
    return Path(__file__).resolve().parents[3]


def _default_ham_root() -> Path:
    """
    Default HAM10000 root folder.

    Priority:
      1) HAM10000_ROOT env var (recommended)
      2) <repo_root>/data/HAM10000  (Docker/Linux-friendly)
    """
    repo_root = _repo_root()
    return Path(os.environ.get("HAM10000_ROOT", str(repo_root / "data" / "HAM10000")))


def load_ham10000_metadata(
    part1_dir: str | None = None,
    part2_dir: str | None = None,
    meta_csv_path: str | None = None,
):
    """
    Loads HAM10000 images + metadata, maps image_id -> image path,
    and returns (meta_df, image_paths).

    Override options:
      - HAM10000_ROOT (recommended)
      - HAM10000_PART1_DIR
      - HAM10000_PART2_DIR
      - HAM10000_META_CSV
    """
    ham_root = _default_ham_root()

    part1_dir = part1_dir or os.environ.get(
        "HAM10000_PART1_DIR", str(ham_root / "HAM10000_images_part_1")
    )
    part2_dir = part2_dir or os.environ.get(
        "HAM10000_PART2_DIR", str(ham_root / "HAM10000_images_part_2")
    )
    meta_csv_path = meta_csv_path or os.environ.get(
        "HAM10000_META_CSV", str(ham_root / "HAM10000_metadata.csv")
    )

    # ---- DEBUG PRINTS (super important) ----
    print("HAM10000_ROOT resolved to:", ham_root)
    print("PART1_DIR:", part1_dir)
    print("PART2_DIR:", part2_dir)
    print("META_CSV:", meta_csv_path)

    # ---- VALIDATE meta_csv_path is a FILE ----
    meta_path = Path(meta_csv_path)

    # If someone passed a directory by mistake (like .../data/HAM10000),
    # try to recover by appending default filename.
    if meta_path.exists() and meta_path.is_dir():
        candidate = meta_path / "HAM10000_metadata.csv"
        if candidate.exists() and candidate.is_file():
            meta_path = candidate
        else:
            raise IsADirectoryError(
                f"meta_csv_path points to a directory: {meta_path}. "
                f"Expected a CSV file. Tried also: {candidate}"
            )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {meta_path} "
            f"(HAM10000_ROOT={ham_root})"
        )
    if not meta_path.is_file():
        raise RuntimeError(f"Metadata path is not a file: {meta_path}")

    # ---- COLLECT images ----
    image_paths: list[str] = []
    image_paths += glob.glob(os.path.join(part1_dir, "**", "*.jpg"), recursive=True)
    image_paths += glob.glob(os.path.join(part2_dir, "**", "*.jpg"), recursive=True)

    # Fallback: sometimes images are stored flat under ham_root
    if len(image_paths) == 0:
        image_paths += glob.glob(os.path.join(str(ham_root), "**", "*.jpg"), recursive=True)

    image_paths = sorted(set(image_paths))
    print("num images found:", len(image_paths))

    if len(image_paths) == 0:
        raise FileNotFoundError(
            "No .jpg images found. Check that PART1_DIR/PART2_DIR are correct.\n"
            f"PART1_DIR={part1_dir}\n"
            f"PART2_DIR={part2_dir}\n"
            f"HAM10000_ROOT={ham_root}"
        )

    # ---- READ metadata ----
    meta = pd.read_csv(meta_path)

    # ---- MAP image_id -> full path ----
    id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    meta["img_path"] = meta["image_id"].map(id_to_path)

    missing_rate = float(meta["img_path"].isna().mean())
    missing_count = int(meta["img_path"].isna().sum())
    print("missing rate:", missing_rate)
    print("missing count:", missing_count)

    # Drop rows without images
    meta = meta[~meta["img_path"].isna()].reset_index(drop=True)

    # Stable index for meta-features matrix alignment
    meta["row_idx"] = np.arange(len(meta), dtype=np.int32)

    return meta, image_paths


