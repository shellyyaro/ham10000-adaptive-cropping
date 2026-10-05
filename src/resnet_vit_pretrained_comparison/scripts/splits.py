from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

print("HAM10000_ROOT ENV =", os.environ.get("HAM10000_ROOT"))


@dataclass(frozen=True)
class Splits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass(frozen=True)
class CVFold:
    fold_idx: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame


@dataclass(frozen=True)
class CVSplits:
    folds: list[CVFold]
    test_df: pd.DataFrame


def _build_train_val_test_manifest(
    *,
    split_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact, reconstructable manifest for one train/val/test split."""
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    cols = _split_identifier_columns(combined)
    parts = []
    for split_name, split_df in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ):
        part = split_df.loc[:, cols].copy()
        part.insert(0, "split", split_name)
        part.insert(0, "split_idx", int(split_idx))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _get_free_path(base_path: Path) -> Path:
    """
    If the target path already exists, create a versioned path:
    file.csv -> file_V1.csv -> file_V2.csv -> ...
    """
    base_path = Path(base_path)
    if not base_path.exists():
        return base_path

    stem, suffix, parent = base_path.stem, base_path.suffix, base_path.parent
    i = 1
    while True:
        cand = parent / f"{stem}_V{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def _validate_required_columns(meta: pd.DataFrame, stratify_col: str) -> None:
    """
    Validate that the dataframe contains the required columns.
    """
    required_cols = ["lesion_id"]
    missing = [col for col in required_cols if col not in meta.columns]
    if missing:
        raise ValueError(
            f"meta is missing required columns: {missing}. "
            f"For leakage-free splitting, 'lesion_id' must exist."
        )

    if stratify_col not in meta.columns:
        raise ValueError(
            f"stratify_col='{stratify_col}' was not found in meta.columns. "
            f"Available columns: {list(meta.columns)}"
        )


def _build_lesion_level_table(
    meta: pd.DataFrame,
    stratify_col: str,
) -> pd.DataFrame:
    """
    Build one row per lesion_id for group-based splitting.

    Assumption:
    - all rows with the same lesion_id should belong to the same class label
      for stratification purposes.

    If not, we raise an error because that indicates inconsistent metadata.
    """
    lesion_label_counts = (
        meta.groupby("lesion_id")[stratify_col]
        .nunique(dropna=False)
        .reset_index(name="n_unique_labels")
    )

    bad_lesions = lesion_label_counts.loc[lesion_label_counts["n_unique_labels"] > 1, "lesion_id"]
    if not bad_lesions.empty:
        raise ValueError(
            f"Found lesion_id values with multiple '{stratify_col}' labels. "
            f"Example lesion_ids: {bad_lesions.head(10).tolist()}"
        )

    lesion_df = (
        meta.groupby("lesion_id", as_index=False)
        .agg({stratify_col: "first"})
        .reset_index(drop=True)
    )

    return lesion_df


def _subset_meta_by_lesions(meta: pd.DataFrame, lesion_ids: pd.Series) -> pd.DataFrame:
    """
    Return all image rows whose lesion_id is in lesion_ids.
    """
    return meta.loc[meta["lesion_id"].isin(set(lesion_ids))].copy()


def _split_identifier_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "row_idx",
        "image_id",
        "lesion_id",
        "dx",
        "dx_type",
        "age",
        "sex",
        "localization",
        "img_path",
    ]
    return [c for c in preferred if c in df.columns]


def _build_split_manifest(
    *,
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    cols = _split_identifier_columns(pd.concat([train_df, val_df], ignore_index=True))

    train_manifest = train_df.loc[:, cols].copy()
    train_manifest.insert(0, "split", "train")
    train_manifest.insert(0, "fold", int(fold_idx))

    val_manifest = val_df.loc[:, cols].copy()
    val_manifest.insert(0, "split", "val")
    val_manifest.insert(0, "fold", int(fold_idx))

    return pd.concat([train_manifest, val_manifest], ignore_index=True)


def _assert_lesion_disjoint(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Fail fast if any lesion_id appears in more than one split.

    This check is deliberately enforced both for newly generated assignments
    and for CSV files loaded from disk.  A stale or manually edited split must
    never be allowed to continue into training.
    """
    train_lesions = set(train_df["lesion_id"].unique())
    val_lesions = set(val_df["lesion_id"].unique())
    test_lesions = set(test_df["lesion_id"].unique())

    train_val_overlap = train_lesions & val_lesions
    train_test_overlap = train_lesions & test_lesions
    val_test_overlap = val_lesions & test_lesions

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError(
            "Lesion leakage detected across train/val/test splits: "
            f"train/val={len(train_val_overlap)}, "
            f"train/test={len(train_test_overlap)}, "
            f"val/test={len(val_test_overlap)}. "
            "Examples: "
            f"train/val={list(sorted(train_val_overlap))[:10]}, "
            f"train/test={list(sorted(train_test_overlap))[:10]}, "
            f"val/test={list(sorted(val_test_overlap))[:10]}"
        )

    print("\n[INFO] No lesion leakage detected across splits.")


def _assert_complete_partition(
    meta: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    context: str,
) -> None:
    """Ensure a three-way split contains every source image exactly once."""
    id_col = "image_id" if "image_id" in meta.columns else "row_idx"
    if id_col not in train_df.columns or id_col not in val_df.columns or id_col not in test_df.columns:
        raise ValueError(f"{context}: identifier column {id_col!r} is missing from a split")

    source_ids = meta[id_col].astype(str)
    combined_ids = pd.concat(
        [train_df[id_col], val_df[id_col], test_df[id_col]], ignore_index=True
    ).astype(str)
    duplicates = combined_ids[combined_ids.duplicated()].unique().tolist()
    missing = sorted(set(source_ids) - set(combined_ids))
    unexpected = sorted(set(combined_ids) - set(source_ids))
    if duplicates or missing or unexpected or len(combined_ids) != len(source_ids):
        raise RuntimeError(
            f"{context}: split does not partition every source image exactly once. "
            f"rows={len(combined_ids)}/{len(source_ids)}, "
            f"duplicate examples={duplicates[:10]}, missing examples={missing[:10]}, "
            f"unexpected examples={unexpected[:10]}"
        )


def _report_cv_fold_overlap(fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    train_lesions = set(train_df["lesion_id"].unique())
    val_lesions = set(val_df["lesion_id"].unique())
    overlap = train_lesions & val_lesions

    if overlap:
        raise RuntimeError(
            f"Lesion leakage detected in fold {fold_idx}: "
            f"{len(overlap)} lesion_id values appear in both train and val. "
            f"Examples: {list(sorted(overlap))[:10]}"
        )

    print(
        f"[INFO] Fold {fold_idx}: no lesion leakage | "
        f"train images={len(train_df):,}, val images={len(val_df):,}, "
        f"train lesions={train_df['lesion_id'].nunique():,}, "
        f"val lesions={val_df['lesion_id'].nunique():,}"
    )


def _report_cv_fold_test_overlap(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_lesions = set(train_df["lesion_id"].unique())
    val_lesions = set(val_df["lesion_id"].unique())
    test_lesions = set(test_df["lesion_id"].unique())

    train_test_overlap = train_lesions & test_lesions
    val_test_overlap = val_lesions & test_lesions
    if train_test_overlap or val_test_overlap:
        raise RuntimeError(
            f"Lesion leakage detected between fold {fold_idx} and fixed test set. "
            f"train/test overlap={len(train_test_overlap)}, "
            f"val/test overlap={len(val_test_overlap)}. "
            f"Examples train/test: {list(sorted(train_test_overlap))[:10]}, "
            f"examples val/test: {list(sorted(val_test_overlap))[:10]}"
        )

    print(
        f"[INFO] Fold {fold_idx}: fixed test leakage check OK | "
        f"test images={len(test_df):,}, test lesions={test_df['lesion_id'].nunique():,}"
    )


def make_or_load_splits(
    meta: pd.DataFrame,
    splits_dir: Path,
    split_seed: int,
    stratify_col: str = "dx",
    overwrite: bool = False,
    prefix: str = "ham10000",
) -> Splits:
    """
    Creates (or loads) deterministic train/val/test splits at the lesion level.

    Important:
    - Splitting is done by lesion_id, not by image row.
    - Therefore, all images belonging to the same lesion_id stay in the same split.
    - Stratification is also done at the lesion level using `stratify_col`.

    Saves:
      <splits_dir>/<prefix>_train.csv
      <splits_dir>/<prefix>_val.csv
      <splits_dir>/<prefix>_test.csv
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / f"{prefix}_train.csv"
    val_path = splits_dir / f"{prefix}_val.csv"
    test_path = splits_dir / f"{prefix}_test.csv"

    if (not overwrite) and train_path.exists() and val_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        _assert_lesion_disjoint(train_df, val_df, test_df)
        _assert_complete_partition(
            meta, train_df, val_df, test_df, context="loaded fixed split"
        )
        return Splits(train_df=train_df, val_df=val_df, test_df=test_df)

    _validate_required_columns(meta, stratify_col)

    meta = meta.copy().reset_index(drop=True)

    # ---------------------------------------------------------
    # Build lesion-level table: one row per lesion_id
    # ---------------------------------------------------------
    lesion_df = _build_lesion_level_table(meta=meta, stratify_col=stratify_col)

    # ---------------------------------------------------------
    # Split lesion IDs: 60/20/20
    # Step 1: lesion trainval / lesion test = 80/20
    # Step 2: lesion train / lesion val = 75/25 of trainval
    # ---------------------------------------------------------
    lesion_trainval_df, lesion_test_df = train_test_split(
        lesion_df,
        test_size=0.2,
        random_state=split_seed,
        stratify=lesion_df[stratify_col],
    )

    lesion_train_df, lesion_val_df = train_test_split(
        lesion_trainval_df,
        test_size=0.25,  # 0.25 * 0.8 = 0.2
        random_state=split_seed,
        stratify=lesion_trainval_df[stratify_col],
    )

    # ---------------------------------------------------------
    # Expand lesion splits back to full image-level rows
    # ---------------------------------------------------------
    train_df = _subset_meta_by_lesions(meta, lesion_train_df["lesion_id"])
    val_df = _subset_meta_by_lesions(meta, lesion_val_df["lesion_id"])
    test_df = _subset_meta_by_lesions(meta, lesion_test_df["lesion_id"])

    # Optional: keep deterministic ordering
    train_df = train_df.sort_values(["lesion_id"]).reset_index(drop=True)
    val_df = val_df.sort_values(["lesion_id"]).reset_index(drop=True)
    test_df = test_df.sort_values(["lesion_id"]).reset_index(drop=True)

    # Final safety gate: never save or train with cross-partition lesions.
    _assert_lesion_disjoint(train_df, val_df, test_df)
    _assert_complete_partition(
        meta, train_df, val_df, test_df, context="generated fixed split"
    )

    # Save
    out_train = train_path if overwrite or (not train_path.exists()) else _get_free_path(train_path)
    out_val = val_path if overwrite or (not val_path.exists()) else _get_free_path(val_path)
    out_test = test_path if overwrite or (not test_path.exists()) else _get_free_path(test_path)

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    print("Created lesion-level splits:")
    print(f"  train images  : {len(train_df):,}")
    print(f"  val images    : {len(val_df):,}")
    print(f"  test images   : {len(test_df):,}")
    print(f"  train lesions : {train_df['lesion_id'].nunique():,}")
    print(f"  val lesions   : {val_df['lesion_id'].nunique():,}")
    print(f"  test lesions  : {test_df['lesion_id'].nunique():,}")

    return Splits(train_df=train_df, val_df=val_df, test_df=test_df)


def make_or_load_cv_folds(
    meta: pd.DataFrame,
    splits_dir: Path,
    split_seed: int,
    n_splits: int = 5,
    stratify_col: str = "dx",
    overwrite: bool = False,
    prefix: str = "ham10000_lesion_cv5",
) -> list[CVFold]:
    """
    Create/load lesion-level stratified K-fold splits.

    Important:
    - The fold assignment is done on lesion_id, not image rows.
    - Therefore, all images from the same lesion_id are always kept together.
    - Stratification is done at lesion level using `stratify_col`.
    - Each fold saves:
        <prefix>_fold{K}_train.csv
        <prefix>_fold{K}_val.csv
        <prefix>_fold{K}_manifest.csv
    - A combined manifest is also saved:
        <prefix>_all_folds_manifest.csv

    The manifest is intentionally redundant: it keeps fold/split plus stable
    row/image/lesion identifiers so every experiment can be reconstructed.
    """
    if int(n_splits) < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    n_splits = int(n_splits)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    fold_paths = []
    for fold_idx in range(n_splits):
        fold_paths.append(
            (
                fold_idx,
                splits_dir / f"{prefix}_fold{fold_idx}_train.csv",
                splits_dir / f"{prefix}_fold{fold_idx}_val.csv",
                splits_dir / f"{prefix}_fold{fold_idx}_manifest.csv",
            )
        )

    all_manifest_path = splits_dir / f"{prefix}_all_folds_manifest.csv"

    if (not overwrite) and all(train.exists() and val.exists() for _, train, val, _ in fold_paths):
        folds: list[CVFold] = []
        all_manifests = []
        for fold_idx, train_path, val_path, manifest_path in fold_paths:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            _report_cv_fold_overlap(fold_idx, train_df, val_df)
            folds.append(CVFold(fold_idx=fold_idx, train_df=train_df, val_df=val_df))

            if manifest_path.exists():
                all_manifests.append(pd.read_csv(manifest_path))
            else:
                all_manifests.append(
                    _build_split_manifest(
                        fold_idx=fold_idx,
                        train_df=train_df,
                        val_df=val_df,
                    )
                )

        if (not all_manifest_path.exists()) and all_manifests:
            pd.concat(all_manifests, ignore_index=True).to_csv(
                all_manifest_path,
                index=False,
            )

        return folds

    _validate_required_columns(meta, stratify_col)

    meta = meta.copy().reset_index(drop=True)
    lesion_df = _build_lesion_level_table(meta=meta, stratify_col=stratify_col)

    class_counts = lesion_df[stratify_col].value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} stratified folds because the smallest "
            f"lesion-level class has only {min_class_count} lesions. "
            f"Class counts: {class_counts.to_dict()}"
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=split_seed,
    )

    folds: list[CVFold] = []
    all_manifests = []

    for fold_idx, (train_pos, val_pos) in enumerate(
        skf.split(lesion_df["lesion_id"], lesion_df[stratify_col])
    ):
        lesion_train_df = lesion_df.iloc[train_pos].reset_index(drop=True)
        lesion_val_df = lesion_df.iloc[val_pos].reset_index(drop=True)

        train_df = _subset_meta_by_lesions(meta, lesion_train_df["lesion_id"])
        val_df = _subset_meta_by_lesions(meta, lesion_val_df["lesion_id"])

        train_df = train_df.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)
        val_df = val_df.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)

        _report_cv_fold_overlap(fold_idx, train_df, val_df)

        _, train_path, val_path, manifest_path = fold_paths[fold_idx]
        out_train = train_path if overwrite or (not train_path.exists()) else _get_free_path(train_path)
        out_val = val_path if overwrite or (not val_path.exists()) else _get_free_path(val_path)
        out_manifest = (
            manifest_path
            if overwrite or (not manifest_path.exists())
            else _get_free_path(manifest_path)
        )

        train_df.to_csv(out_train, index=False)
        val_df.to_csv(out_val, index=False)

        manifest_df = _build_split_manifest(
            fold_idx=fold_idx,
            train_df=train_df,
            val_df=val_df,
        )
        manifest_df.to_csv(out_manifest, index=False)
        all_manifests.append(manifest_df)

        folds.append(CVFold(fold_idx=fold_idx, train_df=train_df, val_df=val_df))

    all_manifest = pd.concat(all_manifests, ignore_index=True)
    out_all_manifest = (
        all_manifest_path
        if overwrite or (not all_manifest_path.exists())
        else _get_free_path(all_manifest_path)
    )
    all_manifest.to_csv(out_all_manifest, index=False)

    print("Created lesion-level stratified CV folds:")
    print(f"  n_splits       : {n_splits}")
    print(f"  total images   : {len(meta):,}")
    print(f"  total lesions  : {meta['lesion_id'].nunique():,}")
    print(f"  manifest       : {out_all_manifest}")

    return folds


def make_or_load_cv_folds_with_test(
    meta: pd.DataFrame,
    splits_dir: Path,
    split_seed: int,
    n_splits: int = 5,
    test_size: float = 0.2,
    stratify_col: str = "dx",
    overwrite: bool = False,
    prefix: str = "ham10000_lesion_cv5_with_test",
) -> CVSplits:
    """
    Create/load a fixed lesion-level test set plus lesion-level CV folds.

    The split structure is:
      - fixed test set: `test_size` of lesions, never used in CV training/val
      - development set: remaining lesions
      - K-fold CV is created only inside the development set

    Saves:
        <prefix>_test.csv
        <prefix>_fold{K}_train.csv
        <prefix>_fold{K}_val.csv
        <prefix>_fold{K}_manifest.csv
        <prefix>_all_folds_manifest.csv
    """
    if int(n_splits) < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    n_splits = int(n_splits)
    test_size = float(test_size)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    test_path = splits_dir / f"{prefix}_test.csv"
    fold_paths = []
    for fold_idx in range(n_splits):
        fold_paths.append(
            (
                fold_idx,
                splits_dir / f"{prefix}_fold{fold_idx}_train.csv",
                splits_dir / f"{prefix}_fold{fold_idx}_val.csv",
                splits_dir / f"{prefix}_fold{fold_idx}_manifest.csv",
            )
        )
    all_manifest_path = splits_dir / f"{prefix}_all_folds_manifest.csv"

    if (
        (not overwrite)
        and test_path.exists()
        and all(train.exists() and val.exists() for _, train, val, _ in fold_paths)
    ):
        test_df = pd.read_csv(test_path)
        folds: list[CVFold] = []
        all_manifests = []

        for fold_idx, train_path, val_path, manifest_path in fold_paths:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            _report_cv_fold_overlap(fold_idx, train_df, val_df)
            _report_cv_fold_test_overlap(fold_idx, train_df, val_df, test_df)
            folds.append(CVFold(fold_idx=fold_idx, train_df=train_df, val_df=val_df))

            if manifest_path.exists():
                all_manifests.append(pd.read_csv(manifest_path))
            else:
                all_manifests.append(
                    _build_split_manifest(
                        fold_idx=fold_idx,
                        train_df=train_df,
                        val_df=val_df,
                    )
                )

        if (not all_manifest_path.exists()) and all_manifests:
            pd.concat(all_manifests, ignore_index=True).to_csv(
                all_manifest_path,
                index=False,
            )

        return CVSplits(folds=folds, test_df=test_df)

    _validate_required_columns(meta, stratify_col)

    meta = meta.copy().reset_index(drop=True)
    lesion_df = _build_lesion_level_table(meta=meta, stratify_col=stratify_col)

    lesion_dev_df, lesion_test_df = train_test_split(
        lesion_df,
        test_size=test_size,
        random_state=split_seed,
        stratify=lesion_df[stratify_col],
    )
    lesion_dev_df = lesion_dev_df.reset_index(drop=True)
    lesion_test_df = lesion_test_df.reset_index(drop=True)

    class_counts = lesion_dev_df[stratify_col].value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} stratified folds inside the development "
            f"set because the smallest lesion-level class has only "
            f"{min_class_count} lesions. Class counts: {class_counts.to_dict()}"
        )

    test_df = _subset_meta_by_lesions(meta, lesion_test_df["lesion_id"])
    test_df = test_df.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)

    out_test = test_path if overwrite or (not test_path.exists()) else _get_free_path(test_path)
    test_df.to_csv(out_test, index=False)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=split_seed,
    )

    folds: list[CVFold] = []
    all_manifests = []

    for fold_idx, (train_pos, val_pos) in enumerate(
        skf.split(lesion_dev_df["lesion_id"], lesion_dev_df[stratify_col])
    ):
        lesion_train_df = lesion_dev_df.iloc[train_pos].reset_index(drop=True)
        lesion_val_df = lesion_dev_df.iloc[val_pos].reset_index(drop=True)

        train_df = _subset_meta_by_lesions(meta, lesion_train_df["lesion_id"])
        val_df = _subset_meta_by_lesions(meta, lesion_val_df["lesion_id"])

        train_df = train_df.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)
        val_df = val_df.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)

        _report_cv_fold_overlap(fold_idx, train_df, val_df)
        _report_cv_fold_test_overlap(fold_idx, train_df, val_df, test_df)

        _, train_path, val_path, manifest_path = fold_paths[fold_idx]
        out_train = train_path if overwrite or (not train_path.exists()) else _get_free_path(train_path)
        out_val = val_path if overwrite or (not val_path.exists()) else _get_free_path(val_path)
        out_manifest = (
            manifest_path
            if overwrite or (not manifest_path.exists())
            else _get_free_path(manifest_path)
        )

        train_df.to_csv(out_train, index=False)
        val_df.to_csv(out_val, index=False)

        manifest_df = _build_split_manifest(
            fold_idx=fold_idx,
            train_df=train_df,
            val_df=val_df,
        )
        manifest_df.to_csv(out_manifest, index=False)
        all_manifests.append(manifest_df)

        folds.append(CVFold(fold_idx=fold_idx, train_df=train_df, val_df=val_df))

    all_manifest = pd.concat(all_manifests, ignore_index=True)
    out_all_manifest = (
        all_manifest_path
        if overwrite or (not all_manifest_path.exists())
        else _get_free_path(all_manifest_path)
    )
    all_manifest.to_csv(out_all_manifest, index=False)

    print("Created fixed lesion-level test set plus stratified CV folds:")
    print(f"  n_splits          : {n_splits}")
    print(f"  test_size         : {test_size}")
    print(f"  total images      : {len(meta):,}")
    print(f"  total lesions     : {meta['lesion_id'].nunique():,}")
    print(f"  development images: {sum(len(f.val_df) for f in folds):,}")
    print(f"  test images       : {len(test_df):,}")
    print(f"  test lesions      : {test_df['lesion_id'].nunique():,}")
    print(f"  test path         : {out_test}")
    print(f"  manifest          : {out_all_manifest}")

    return CVSplits(folds=folds, test_df=test_df)


def make_or_load_repeated_train_val_test_splits(
    meta: pd.DataFrame,
    splits_dir: Path,
    split_seed: int = 0,
    n_splits: int = 5,
    val_fraction: float = 0.2,
    stratify_col: str = "dx",
    overwrite: bool = False,
    prefix: str = "ham10000_lesion_5way_tvt",
) -> list[Splits]:
    """Create distinct, leakage-free train/val/test experiments.

    The outer StratifiedKFold assigns every lesion to exactly one test set.
    Consequently, test sets are pairwise disjoint at lesion level and together
    cover all lesions. For each experiment, ``val_fraction`` of the non-test
    lesions is selected as validation data; the rest is training data.

    ``val_fraction=0.2`` produces approximately 64/16/20 train/val/test.
    All splitting and stratification happen at lesion level, then all image rows
    belonging to each selected lesion are expanded back into the saved CSVs.
    """
    if int(n_splits) < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(
            f"val_fraction must be between 0 and 1, got {val_fraction}"
        )

    n_splits = int(n_splits)
    val_fraction = float(val_fraction)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_paths = [
        (
            split_idx,
            splits_dir / f"{prefix}_split{split_idx}_train.csv",
            splits_dir / f"{prefix}_split{split_idx}_val.csv",
            splits_dir / f"{prefix}_split{split_idx}_test.csv",
            splits_dir / f"{prefix}_split{split_idx}_manifest.csv",
        )
        for split_idx in range(n_splits)
    ]
    all_manifest_path = splits_dir / f"{prefix}_all_splits_manifest.csv"

    if (not overwrite) and all(
        train.exists() and val.exists() and test.exists()
        for _, train, val, test, _ in split_paths
    ):
        loaded = []
        manifests = []
        all_test_lesions: set = set()
        for split_idx, train_path, val_path, test_path, manifest_path in split_paths:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            _assert_three_way_lesion_disjoint(split_idx, train_df, val_df, test_df)
            _assert_complete_partition(
                meta,
                train_df,
                val_df,
                test_df,
                context=f"loaded final split {split_idx}",
            )
            test_lesions = set(test_df["lesion_id"])
            repeated = all_test_lesions & test_lesions
            if repeated:
                raise RuntimeError(
                    f"Test lesion overlap across saved splits before split {split_idx}: "
                    f"{list(sorted(repeated))[:10]}"
                )
            all_test_lesions.update(test_lesions)
            loaded.append(Splits(train_df, val_df, test_df))
            manifests.append(
                pd.read_csv(manifest_path)
                if manifest_path.exists()
                else _build_train_val_test_manifest(
                    split_idx=split_idx,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                )
            )
        expected_test_lesions = set(meta["lesion_id"])
        if all_test_lesions != expected_test_lesions:
            missing = sorted(expected_test_lesions - all_test_lesions)
            unexpected = sorted(all_test_lesions - expected_test_lesions)
            raise RuntimeError(
                "Saved test folds do not cover every lesion exactly once. "
                f"missing examples={missing[:10]}, unexpected examples={unexpected[:10]}"
            )
        if not all_manifest_path.exists():
            pd.concat(manifests, ignore_index=True).to_csv(all_manifest_path, index=False)
        return loaded

    _validate_required_columns(meta, stratify_col)
    meta = meta.copy().reset_index(drop=True)
    lesion_df = _build_lesion_level_table(meta, stratify_col)
    class_counts = lesion_df[stratify_col].value_counts()
    if int(class_counts.min()) < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} stratified test sets: smallest class has "
            f"only {int(class_counts.min())} lesions. Counts: {class_counts.to_dict()}"
        )

    outer = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=split_seed,
    )
    results = []
    manifests = []
    seen_test_lesions: set = set()

    for split_idx, (development_pos, test_pos) in enumerate(
        outer.split(lesion_df["lesion_id"], lesion_df[stratify_col])
    ):
        development_lesions = lesion_df.iloc[development_pos].reset_index(drop=True)
        test_lesions = lesion_df.iloc[test_pos].reset_index(drop=True)
        train_lesions, val_lesions = train_test_split(
            development_lesions,
            test_size=val_fraction,
            random_state=split_seed + split_idx,
            stratify=development_lesions[stratify_col],
        )

        train_df = _subset_meta_by_lesions(meta, train_lesions["lesion_id"])
        val_df = _subset_meta_by_lesions(meta, val_lesions["lesion_id"])
        test_df = _subset_meta_by_lesions(meta, test_lesions["lesion_id"])
        sort_cols = [c for c in ("lesion_id", "image_id") if c in meta.columns]
        train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
        val_df = val_df.sort_values(sort_cols).reset_index(drop=True)
        test_df = test_df.sort_values(sort_cols).reset_index(drop=True)

        _assert_three_way_lesion_disjoint(split_idx, train_df, val_df, test_df)
        _assert_complete_partition(
            meta,
            train_df,
            val_df,
            test_df,
            context=f"generated final split {split_idx}",
        )
        current_test = set(test_df["lesion_id"])
        repeated = seen_test_lesions & current_test
        if repeated:
            raise RuntimeError(
                f"Test lesion overlap across generated splits at split {split_idx}: "
                f"{list(sorted(repeated))[:10]}"
            )
        seen_test_lesions.update(current_test)

        _, train_path, val_path, test_path, manifest_path = split_paths[split_idx]
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        manifest = _build_train_val_test_manifest(
            split_idx=split_idx,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )
        manifest.to_csv(manifest_path, index=False)
        manifests.append(manifest)
        results.append(Splits(train_df, val_df, test_df))

        print(
            f"[INFO] Split {split_idx}: "
            f"train={len(train_df):,} images/{train_df['lesion_id'].nunique():,} lesions, "
            f"val={len(val_df):,}/{val_df['lesion_id'].nunique():,}, "
            f"test={len(test_df):,}/{test_df['lesion_id'].nunique():,}"
        )

    if seen_test_lesions != set(lesion_df["lesion_id"]):
        raise RuntimeError("The five test sets do not cover every lesion exactly once.")
    pd.concat(manifests, ignore_index=True).to_csv(all_manifest_path, index=False)
    print(f"Saved {n_splits} lesion-level train/val/test splits to: {splits_dir}")
    print(f"Combined manifest: {all_manifest_path}")
    return results


def _assert_three_way_lesion_disjoint(
    split_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train = set(train_df["lesion_id"])
    val = set(val_df["lesion_id"])
    test = set(test_df["lesion_id"])
    overlaps = {
        "train/val": train & val,
        "train/test": train & test,
        "val/test": val & test,
    }
    bad = {name: ids for name, ids in overlaps.items() if ids}
    if bad:
        details = {name: list(sorted(ids))[:10] for name, ids in bad.items()}
        raise RuntimeError(f"Lesion leakage in split {split_idx}: {details}")
