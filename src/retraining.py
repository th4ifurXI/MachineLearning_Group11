"""Drift-aware retraining: periodic refits on all labelled wafers, with recency weights.

The static model (04/05) is fitted once on July-September wafers. In the test period the process
drifted and a new failure pattern appeared in early October, which the static model could not see.
In a fab every wafer receives its line-test label shortly after it is scored, so the model can be
refitted on all wafers labelled so far before each new block of production. Recency weights
(``0.5 ** (age_in_days / half_life)``) let the most recent wafers count more.

Evaluation is prequential: each block is scored by a model fitted only on wafers produced
*before* that block, so no wafer is ever scored by a model that saw its own label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

DEFAULT_FREQ = "W"          # refit weekly (block = calendar week, Monday start)
DEFAULT_HALF_LIFE = 30.0    # days


def recency_weights(train_times: pd.Series, ref_time: pd.Timestamp, half_life_days: float | None) -> np.ndarray | None:
    """Weight 1 for a wafer produced at ``ref_time``, 0.5 for one ``half_life_days`` older, and so on."""
    if half_life_days is None:
        return None
    age = (ref_time - pd.Series(train_times)).dt.total_seconds().to_numpy() / 86400.0
    return np.power(0.5, age / half_life_days)


def blocks(times: pd.Series, positions: np.ndarray, freq: str = DEFAULT_FREQ) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_positions, score_positions) for each day ('D') or week ('W') among ``positions``.

    ``times`` must be sorted (load_secom() does this); each block is trained on every earlier row.
    """
    t = pd.Series(times).iloc[positions]
    key = t.dt.floor("D") if freq == "D" else t.dt.to_period("W").dt.start_time
    out = []
    for k in pd.unique(key):
        idx = positions[(key == k).to_numpy()]
        out.append((np.arange(idx[0]), idx))
    return out


def _fit_score(estimator, X, y, times, train_idx, score_idx, half_life_days):
    m = clone(estimator)
    w = recency_weights(times.iloc[train_idx], times.iloc[score_idx[0]], half_life_days)
    kw = {} if w is None else {"clf__sample_weight": w}
    m.fit(X.iloc[train_idx], y.iloc[train_idx], **kw)
    return score_idx, m.predict_proba(X.iloc[score_idx])[:, 1]


def rolling_scores(estimator, X: pd.DataFrame, y: pd.Series, times: pd.Series, positions: np.ndarray,
                   freq: str = DEFAULT_FREQ, half_life_days: float | None = DEFAULT_HALF_LIFE,
                   n_jobs: int = -1) -> np.ndarray:
    """Prequential fail scores for the rows at ``positions``, refitting before every block.

    ``estimator`` is an (imblearn) Pipeline whose final step is named ``clf`` and accepts
    ``sample_weight``. Returns scores aligned with ``positions``.
    """
    times = pd.Series(times).reset_index(drop=True)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    jobs = blocks(times, np.asarray(positions), freq)
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_score)(estimator, X, y, times, tr, sc, half_life_days) for tr, sc in jobs)
    pos = np.concatenate([p for p, _ in res])
    s = np.concatenate([q for _, q in res])
    order = np.argsort(pos)
    assert np.array_equal(pos[order], np.sort(np.asarray(positions)))
    return s[order]


def static_fold_scores(estimator, X, y, times, folds, half_life_days: float | None = None, n_jobs: int = 4):
    """Time-ordered CV scores (fixed folds) with optional recency weights inside each training fold."""
    times = pd.Series(times).reset_index(drop=True)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_score)(estimator, X, y, times, tr, va, half_life_days) for tr, va in folds)
    pos = np.concatenate([p for p, _ in res])
    s = np.concatenate([q for _, q in res])
    return s[np.argsort(pos)]
