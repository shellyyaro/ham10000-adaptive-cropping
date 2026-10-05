"""Shared HAM10000 data-preparation helpers.

The report runners build fold-specific metadata features directly because each
fold has a different training partition.  These helpers preserve the original
project's reusable data-pipeline API while making the leakage-free path
explicit for standalone use.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence

from resnet_vit_pretrained_comparison.data.data_loader import (
    load_ham10000_metadata,
)
from resnet_vit_pretrained_comparison.data.meta_features import (
    build_meta_features,
    build_meta_features_leakage_free,
)


def prepare_base_data():
    """Load all data and fit metadata preprocessing on every row.

    This function is retained only for compatibility with early exploratory
    scripts.  It must not be used for report experiments because fitting the
    metadata scaler on validation/test rows leaks information.
    """
    warnings.warn(
        "prepare_base_data() fits metadata preprocessing on all rows. "
        "Use prepare_base_data_leakage_free(train_row_idxs) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print("Loading metadata...")
    start = time.time()
    metadata, image_paths = load_ham10000_metadata()
    print(f"load_ham10000_metadata took {time.time() - start:.1f}s")

    print("Building metadata features on all rows (deprecated)...")
    start = time.time()
    features, feature_frame, scaler, feature_dim = build_meta_features(metadata)
    print(f"build_meta_features took {time.time() - start:.1f}s")
    return (
        metadata,
        image_paths,
        features,
        feature_frame,
        scaler,
        feature_dim,
    )


def prepare_base_data_leakage_free(train_row_idxs: Sequence[int]):
    """Load HAM10000 and fit metadata preprocessing on training rows only.

    ``train_row_idxs`` must be produced by the lesion-level split code.  The
    returned feature matrix covers all rows, but imputation, one-hot schema,
    and standardization statistics are learned exclusively from the supplied
    training indices.
    """
    print("Loading metadata...")
    start = time.time()
    metadata, image_paths = load_ham10000_metadata()
    print(f"load_ham10000_metadata took {time.time() - start:.1f}s")

    print("Building leakage-free metadata features (training rows only)...")
    start = time.time()
    features, feature_frame, scaler, feature_dim = (
        build_meta_features_leakage_free(metadata, train_row_idxs)
    )
    print(
        "build_meta_features_leakage_free took "
        f"{time.time() - start:.1f}s"
    )
    return (
        metadata,
        image_paths,
        features,
        feature_frame,
        scaler,
        feature_dim,
    )
