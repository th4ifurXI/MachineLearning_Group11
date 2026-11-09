"""Energy-aware cost metric, cost-optimal thresholds and the evaluation metrics used in 05.

Energy cost = FN x C_FN + FP x C_FP

- FN (missed fault): a defective wafer continues through the remaining energy-intensive
  process steps before it is scrapped, so its downstream energy is wasted.
- FP (false alarm): a good wafer is pulled for one extra inspection.

The 10:1 C_FN:C_FP ratio is a project ASSUMPTION. Report it, and use the 5:1 and 20:1
ratios in COST_RATIOS as a sensitivity analysis.

Scores can be probabilities or any decision score (e.g. SVM decision_function): a wafer
is flagged as a fault when ``score >= threshold``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, make_scorer, precision_score, recall_score, roc_auc_score)

C_FN = 10.0   # cost of a missed fault
C_FP = 1.0    # cost of a false alarm
COST_RATIOS = {"5:1": (5.0, 1.0), "10:1": (10.0, 1.0), "20:1": (20.0, 1.0)}   # name -> (C_FN, C_FP)


def energy_cost(y_true, y_pred, c_fn: float = C_FN, c_fp: float = C_FP) -> float:
    """Total energy cost = FN x C_FN + FP x C_FP (lower is better)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(c_fn * fn + c_fp * fp)


def flag_nothing_cost(y_true, c_fn: float = C_FN) -> float:
    """Cost of the trivial 'never flag a fault' policy = C_FN x number of faults."""
    return float(c_fn * (np.asarray(y_true) == 1).sum())


def flag_everything_cost(y_true, c_fp: float = C_FP) -> float:
    """Cost of the trivial 'flag every wafer' policy = C_FP x number of good wafers."""
    return float(c_fp * (np.asarray(y_true) == 0).sum())


def normalized_energy_cost(y_true, y_pred, c_fn: float = C_FN, c_fp: float = C_FP) -> float:
    """Cost / flag-nothing cost: 0 = perfect, 1 = no better than never flagging, > 1 = worse."""
    base = flag_nothing_cost(y_true, c_fn)
    return float("nan") if base == 0 else energy_cost(y_true, y_pred, c_fn, c_fp) / base


def cost_curve(y_true, scores, c_fn: float = C_FN, c_fp: float = C_FP) -> pd.DataFrame:
    """Exact cost at every distinct threshold (one row per unique score, plus +inf = flag nothing).

    Returns columns: threshold, tp, fp, fn, tn, cost. Rows are ordered from the highest
    threshold (flag nothing) to the lowest (flag everything).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]   # last index of each distinct score
    P, N = int(y.sum()), int(len(y) - y.sum())
    df = pd.DataFrame({
        "threshold": np.r_[np.inf, s[last]],
        "tp": np.r_[0, tp[last]],
        "fp": np.r_[0, fp[last]],
    })
    df["fn"] = P - df["tp"]
    df["tn"] = N - df["fp"]
    df["cost"] = c_fn * df["fn"] + c_fp * df["fp"]
    return df


def best_threshold(y_true, scores, c_fn: float = C_FN, c_fp: float = C_FP) -> tuple[float, float]:
    """Threshold that minimises energy cost on (y_true, scores). Returns (threshold, cost).

    Tune on cross-validated (out-of-fold) training predictions only, never on the test set.
    When several thresholds tie for the minimum, the middle one is used so the choice is
    not pushed to the edge of the flat region.
    """
    curve = cost_curve(y_true, scores, c_fn, c_fp)
    ties = np.flatnonzero(curve["cost"].to_numpy() == curve["cost"].min())
    i = ties[len(ties) // 2]
    return float(curve["threshold"].iloc[i]), float(curve["cost"].iloc[i])


def evaluate(y_true, scores, threshold: float, c_fn: float = C_FN, c_fp: float = C_FP) -> dict:
    """All report metrics for one model at one threshold.

    Threshold-free: ROC-AUC, PR-AUC. At the threshold: recall, precision, F1, F2, balanced
    error rate (BER = mean of the two class error rates), accuracy, energy cost.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(threshold),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "ber": float(0.5 * (fnr + fpr)),
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "accuracy": accuracy_score(y, pred),
        "energy_cost": float(c_fn * fn + c_fp * fp),
        "normalized_cost": float(c_fn * fn + c_fp * fp) / flag_nothing_cost(y, c_fn),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def cost_scorer(c_fn: float = C_FN, c_fp: float = C_FP):
    """sklearn scorer (negative energy cost at the estimator's default decision rule)."""
    return make_scorer(energy_cost, greater_is_better=False, c_fn=c_fn, c_fp=c_fp)
