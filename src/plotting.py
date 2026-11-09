"""Shared plotting style so every notebook's figures look consistent in the report."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

# Colour roles (colourblind-safe pair for pass/fail, validated categorical slots 1-2)
PASS_COLOR = "#2a78d6"   # blue
FAIL_COLOR = "#eb6834"   # orange
NEUTRAL_COLOR = "#2a78d6"
REF_LINE_COLOR = "#52514e"  # thresholds / reference lines
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"

DPI = 300


def apply_style() -> None:
    """Thin marks, recessive axes and grid, neutral text."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": TEXT_PRIMARY,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "font.family": "sans-serif",
    })


def save_fig(fig, name: str, folder: str | Path = FIGURES_DIR) -> Path:
    """Save a figure as PNG at 300 dpi into outputs/figures/ and return its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path
