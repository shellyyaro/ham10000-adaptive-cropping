import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_meta_features(
    meta: pd.DataFrame,
    feature_cols=None,
    scaler: StandardScaler | None = None,
):
    """
    ⚠️ DEPRECATED — fits scaler on ALL rows (leaks val/test statistics).
    Use build_meta_features_leakage_free() instead.
    """
    warnings.warn(
        "build_meta_features fits the scaler on ALL rows (including val/test). "
        "Use build_meta_features_leakage_free() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    """
    מקבל DataFrame של meta מה-HAM10000,
    מחזיר:
      - meta_features: מטריצה מנורמלת (np.ndarray)
      - meta_features_df: הדאטה-פריים אחרי one-hot
      - scaler: האובייקט של StandardScaler
      - meta_feature_dim: מספר הפיצ'רים
    """

    if feature_cols is None:
        feature_cols = ["age", "sex", "localization"]

    meta_feat = meta.copy()

    # טיפול בגיל
    meta_feat["age"] = meta_feat["age"].fillna(meta_feat["age"].median())

    # טיפול בקטגוריות
    meta_feat["sex"] = meta_feat["sex"].fillna("unknown")
    meta_feat["localization"] = meta_feat["localization"].fillna("unknown")

    # one-hot encoding
    meta_features_df = pd.get_dummies(
        meta_feat[feature_cols],
        columns=["sex", "localization"],
        drop_first=False,
    )

    # נרמול
    if scaler is None:
        scaler = StandardScaler()
        meta_features = scaler.fit_transform(meta_features_df.values.astype("float32"))
    else:
        meta_features = scaler.transform(meta_features_df.values.astype("float32"))

    meta_feature_dim = meta_features.shape[1]

    return meta_features, meta_features_df, scaler, meta_feature_dim


def build_meta_features_leakage_free(
    meta: pd.DataFrame,
    train_row_idxs: np.ndarray,
    feature_cols: list[str] | None = None,
):
    """
    Build meta features without data leakage.

    Steps:
      1. Impute age with the TRAIN median (not the global median).
      2. One-hot encode on the FULL dataset (so all categories appear).
      3. Fit StandardScaler on TRAIN rows only.
      4. Transform ALL rows with the train-fitted scaler.

    Returns:
      meta_features   : np.ndarray (N, D) — scaled features for every row
      meta_features_df: pd.DataFrame      — pre-scaling one-hot DataFrame
      scaler          : StandardScaler     — fitted on train only
      meta_feature_dim: int
    """
    if feature_cols is None:
        feature_cols = ["age", "sex", "localization"]

    meta_feat = meta.copy()

    # Impute age with TRAIN median only
    train_age_median = meta_feat.iloc[train_row_idxs]["age"].median()
    meta_feat["age"] = meta_feat["age"].fillna(train_age_median)

    # Categorical imputation (no leakage concern — just filling with a constant)
    meta_feat["sex"] = meta_feat["sex"].fillna("unknown")
    meta_feat["localization"] = meta_feat["localization"].fillna("unknown")

    # One-hot on FULL data (ensures consistent column set)
    meta_features_df = pd.get_dummies(
        meta_feat[feature_cols],
        columns=["sex", "localization"],
        drop_first=False,
    )

    all_values = meta_features_df.values.astype("float32")

    # Fit scaler on TRAIN rows only
    scaler = StandardScaler()
    scaler.fit(all_values[train_row_idxs])

    # Transform ALL rows with train-fitted scaler
    meta_features = scaler.transform(all_values)
    meta_feature_dim = meta_features.shape[1]

    return meta_features, meta_features_df, scaler, meta_feature_dim