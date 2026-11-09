"""Project-wide constants and paths.

Import from here instead of hard-coding paths or seeds in notebooks so every
team member gets identical, reproducible results.
"""
from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Dataset"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

FEATURES_FILE = DATA_DIR / "secom.data"
LABELS_FILE = DATA_DIR / "secom_labels.data"

TARGET_COL = "label"
TIME_COL = "timestamp"
