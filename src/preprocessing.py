"""Time-ordered splitting and the leakage-safe preprocessing pipeline for SECOM.

Every transformer here learns its decisions (which columns to drop, medians,
Yeo-Johnson lambdas, means/SDs) in ``fit`` from the TRAINING data only and
applies them unchanged in ``transform``. All steps keep pandas column names,
so the surviving columns are still the original sensor names (f000..f589).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.config import TIME_COL


def time_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split a time-sorted DataFrame: first (1 - test_size) rows -> train, rest -> test.

    No shuffling: the model is always tested on wafers produced *after* the ones
    it was trained on, as it would be used in the fab.
    """
    if not df[TIME_COL].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by time (use load_secom()).")
    n_train = int(round(len(df) * (1 - test_size)))
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


class _ColumnDropper(BaseEstimator, TransformerMixin):
    """Base class: subclasses set ``self.keep_`` (columns to keep) in ``fit``."""

    def _check_df(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} expects a pandas DataFrame.")
        return X

    def transform(self, X):
        X = self._check_df(X)
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_, dtype=object)

    @property
    def dropped_(self) -> list[str]:
        return [c for c in self.feature_names_in_ if c not in set(self.keep_)]


class DropConstant(_ColumnDropper):
    """Drop features with at most one unique non-missing value in the training data."""

    def fit(self, X, y=None):
        X = self._check_df(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        nunique = X.nunique(dropna=True)
        self.keep_ = nunique[nunique > 1].index.tolist()
        return self


class DropHighMissing(_ColumnDropper):
    """Drop features whose missing fraction in the training data exceeds ``threshold``."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = self._check_df(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.missing_frac_ = X.isna().mean()
        self.keep_ = self.missing_frac_[self.missing_frac_ <= self.threshold].index.tolist()
        return self


class DropCorrelated(_ColumnDropper):
    """Drop one feature from each pair with |Pearson r| > ``threshold``.

    Greedy, in column order: a feature is kept only if its |r| with every
    already-kept feature is <= threshold, so the first feature of each
    near-duplicate group survives. ``drop_map_`` records which kept feature
    each dropped one duplicated.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = self._check_df(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        corr = np.abs(np.corrcoef(X.to_numpy(dtype=float), rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0)
        cols = list(X.columns)
        kept_idx: list[int] = []
        self.drop_map_: dict[str, str] = {}
        for j in range(len(cols)):
            if kept_idx:
                r = corr[j, kept_idx]
                k = int(np.argmax(r))
                if r[k] > self.threshold:
                    self.drop_map_[cols[j]] = cols[kept_idx[k]]
                    continue
            kept_idx.append(j)
        self.keep_ = [cols[i] for i in kept_idx]
        return self


def build_preprocessing_pipeline(missing_threshold: float = 0.5, corr_threshold: float = 0.95) -> Pipeline:
    """Constant drop -> high-missing drop -> median impute -> Yeo-Johnson -> standardise -> correlated drop.

    ``pre_scale`` standardises before Yeo-Johnson. Without it, sensors with large raw
    values and a small spread (e.g. ~400 +/- 10) get extreme lambdas (down to about -47)
    and the transform squashes them to a numerically constant column, destroying them.
    """
    pipe = Pipeline([
        ("drop_constant", DropConstant()),
        ("drop_missing", DropHighMissing(threshold=missing_threshold)),
        ("impute", SimpleImputer(strategy="median")),
        ("pre_scale", StandardScaler()),
        ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scale", StandardScaler()),
        ("drop_correlated", DropCorrelated(threshold=corr_threshold)),
    ])
    return pipe.set_output(transform="pandas")


def preprocessing_steps(missing_threshold: float = 0.5, corr_threshold: float = 0.95) -> list:
    """The same steps as ``build_preprocessing_pipeline`` as a flat list of (name, transformer).

    imblearn's Pipeline (needed to put SMOTE inside CV folds) does not accept a nested
    Pipeline, so use ``imblearn.pipeline.Pipeline(preprocessing_steps() + [...])``.
    Each step is already configured to output pandas DataFrames.
    """
    return build_preprocessing_pipeline(missing_threshold, corr_threshold).steps
