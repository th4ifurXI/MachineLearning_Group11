"""Time-ordered cross-validation for the (time-sorted) SECOM training set.

Shuffled K-fold mixes wafers from every period, so each validation fold looks like the
training data and scores come out optimistic (05_Evaluation showed ROC-AUC ~0.72 in shuffled
CV vs ~0.6 on later wafers). Here every validation fold comes strictly AFTER the wafers the
model was trained on, as in real use.

TimeSeriesSplit(4) on the 1254 training wafers: expanding training windows of 254, 504, 754
and 1004 wafers, each validated on the next 250 wafers (20, 8, 9 and 11 fails).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

N_TIME_SPLITS = 4


def time_cv(n_splits: int = N_TIME_SPLITS) -> TimeSeriesSplit:
    """Expanding-window, time-ordered CV. X must be sorted by time (load_secom() does this)."""
    return TimeSeriesSplit(n_splits=n_splits)


def time_oof_scores(estimator, X, y, cv=None):
    """Fail scores for every validation fold of a time-ordered CV.

    Returns (positions, scores): positions of the scored rows in X (the first training
    window is never scored) and the matching scores (predict_proba[:, 1], or
    decision_function for models without probabilities). Each fold uses a fresh clone.
    """
    cv = cv or time_cv()
    pos, out = [], []
    for train_idx, val_idx in cv.split(X, y):
        m = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        Xv = X.iloc[val_idx]
        s = m.predict_proba(Xv)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xv)
        pos.append(val_idx)
        out.append(s)
    return np.concatenate(pos), np.concatenate(out)
