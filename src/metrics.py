"""Backward-compatible names for the cost metric (used by 04_Modelling).

New code should import from ``src.cost_metric``, which holds the implementation.
"""
from __future__ import annotations

from src.cost_metric import (C_FN as DEFAULT_COST_FN, C_FP as DEFAULT_COST_FP, best_threshold,
                             cost_scorer, energy_cost as total_cost,
                             normalized_energy_cost as normalized_cost)

__all__ = ["DEFAULT_COST_FN", "DEFAULT_COST_FP", "total_cost", "normalized_cost", "best_threshold",
           "cost_scorer"]
