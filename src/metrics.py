"""Cost-sensitive evaluation metrics for energy-aware fault detection.

Rationale
---------
A missed fault (false negative) lets a defective wafer continue through the
remaining, energy-intensive process steps before it is scrapped, so it wastes
far more energy/cost than a false alarm (false positive), which only triggers
an extra inspection. The default 10:1 ratio is a project ASSUMPTION — state it
in the report and run a sensitivity analysis over ``cost_fn`` / ``cost_fp``.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer

DEFAULT_COST_FN = 10.0  # missed fault -> downstream energy wasted on a bad wafer
DEFAULT_COST_FP = 1.0   # false alarm -> one unnecessary inspection


def total_cost(y_true, y_pred, cost_fn: float = DEFAULT_COST_FN, cost_fp: float = DEFAULT_COST_FP) -> float:
    """Total misclassification cost = cost_fn * FN + cost_fp * FP (lower is better)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(cost_fn * fn + cost_fp * fp)


def normalized_cost(y_true, y_pred, cost_fn: float = DEFAULT_COST_FN, cost_fp: float = DEFAULT_COST_FP) -> float:
    """Cost relative to the trivial 'predict all pass' baseline (= cost_fn * #faults).

    0 = perfect, 1 = no better than never flagging a fault, >1 = worse.
    """
    y_true = np.asarray(y_true)
    baseline = cost_fn * (y_true == 1).sum()
    if baseline == 0:
        return float("nan")
    return total_cost(y_true, y_pred, cost_fn, cost_fp) / baseline


def best_threshold(y_true, y_proba, cost_fn: float = DEFAULT_COST_FN, cost_fp: float = DEFAULT_COST_FP,
                   thresholds=None) -> tuple[float, float]:
    """Find the probability threshold that minimises total cost.

    Tune this on validation data only, never on the test set.
    Returns (threshold, cost).
    """
    y_proba = np.asarray(y_proba)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y_true, (y_proba >= t).astype(int), cost_fn, cost_fp) for t in thresholds]
    i = int(np.argmin(costs))
    return float(thresholds[i]), float(costs[i])


def cost_scorer(cost_fn: float = DEFAULT_COST_FN, cost_fp: float = DEFAULT_COST_FP):
    """sklearn scorer for GridSearchCV etc. (greater_is_better=False, so scores are negative cost)."""
    return make_scorer(total_cost, greater_is_better=False, cost_fn=cost_fn, cost_fp=cost_fp)
