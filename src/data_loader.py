"""Loading utilities for the UCI SECOM dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURES_FILE, LABELS_FILE, TARGET_COL, TIME_COL

N_FEATURES = 590


def load_secom(
    features_path: str | Path = FEATURES_FILE,
    labels_path: str | Path = LABELS_FILE,
) -> pd.DataFrame:
    """Load SECOM features and labels into one time-ordered DataFrame.

    - Features: 1567 x 590 space-separated sensor readings ("NaN" = missing),
      named ``f000`` .. ``f589``.
    - Labels: original -1 (pass) / 1 (fail) recoded to 0 (pass) / 1 (fail).
    - Timestamps: parsed from "dd/mm/yyyy hh:mm:ss".
    - Rows are sorted by timestamp (stable sort, so ties keep file order) and
      the index is reset. The original row number is kept in ``orig_index``.

    Returns a DataFrame with columns:
        orig_index, timestamp, label, f000 .. f589
    """
    features = pd.read_csv(
        features_path, sep=r"\s+", header=None, na_values=["NaN"], engine="python"
    )
    if features.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} feature columns, got {features.shape[1]}")
    features.columns = [f"f{i:03d}" for i in range(N_FEATURES)]

    labels = pd.read_csv(
        labels_path, sep=" ", header=None, names=["raw_label", TIME_COL], quotechar='"'
    )
    if len(labels) != len(features):
        raise ValueError(f"Row mismatch: {len(features)} features vs {len(labels)} labels")

    unexpected = set(labels["raw_label"].unique()) - {-1, 1}
    if unexpected:
        raise ValueError(f"Unexpected label values: {unexpected}")

    meta = pd.DataFrame(
        {
            "orig_index": np.arange(len(labels)),
            TIME_COL: pd.to_datetime(labels[TIME_COL], format="%d/%m/%Y %H:%M:%S"),
            TARGET_COL: (labels["raw_label"] == 1).astype(int),
        }
    )

    df = pd.concat([meta, features], axis=1)
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with only the sensor features in X."""
    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    return df[feature_cols], df[TARGET_COL]
