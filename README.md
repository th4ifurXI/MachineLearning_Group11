# Energy-Aware Early Fault Detection in Semiconductor Wafer Fabrication Using Machine Learning

**TEB3123 Machine Learning — Group 11**

## Project overview

Semiconductor fabs monitor hundreds of sensors per wafer. Detecting a faulty wafer early stops it from
consuming energy in later, energy-intensive process steps before it is scrapped anyway. This project uses
the [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom) to build classifiers that flag
failing wafers from in-line sensor data, and evaluates them with an **energy-aware cost metric** that
penalises missed faults (false negatives) more heavily than false alarms (false positives).

**Dataset** (`Dataset/`)

| File | Contents |
|---|---|
| `secom.data` | 1567 wafers × 590 sensor features, space-separated, `NaN` = missing |
| `secom_labels.data` | Label (`-1` pass, `1` fail) + timestamp `"dd/mm/yyyy hh:mm:ss"` |
| `secom.names` | Original dataset description |

Key challenges: heavy class imbalance (~6.6% fails), many missing values, many constant/redundant features,
and time-ordered data (we split by time to avoid leakage).

## Repository structure

```
Dataset/                  Raw SECOM data (read-only)
notebooks/
  01_EDA.ipynb            Class balance, missingness, distributions, fail rate over time
  02_Preprocessing.ipynb  80/20 time-ordered split; fits and saves the preprocessing pipeline
  03_Feature_Selection.ipynb  ANOVA / mutual info / L1 / RF importance vs PCA; saves selected features
  04_Modelling.ipynb      Logistic Regression, Random Forest, XGBoost + tuning (imbalance handling here)
  05_Evaluation.ipynb     Standard metrics + energy-aware cost, threshold tuning
  06_Interpretation.ipynb SHAP explanations
src/
  config.py               Paths and RANDOM_STATE = 42
  data_loader.py          load_secom(), split_xy()
  metrics.py              Energy-aware cost metric, threshold search, sklearn scorer
  preprocessing.py        time_train_test_split(), build_preprocessing_pipeline() and custom drop steps
  plotting.py             Shared chart style and save_fig() (300 dpi)
outputs/
  figures/                Saved plots for the report
  models/                 selected_features.json, CV results, rankings (committed);
                          *.joblib fitted objects (git-ignored - re-run 02 to create)
```

Notebooks 03–06 load `outputs/models/preprocessing_pipeline.joblib`, which is not in git. After cloning, run
`02_Preprocessing.ipynb` once to create it.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order (01 → 06). Each notebook's first cell adds the project root to `sys.path`
and loads the data:

```python
from src.data_loader import load_secom, split_xy
df = load_secom()      # sorted by time; label: 0 = pass, 1 = fail
X, y = split_xy(df)    # X = 590 sensor columns f000..f589
```

## Conventions

- **Reproducibility:** use `RANDOM_STATE` from `src/config.py` (= 42) for every split, model and resampler.
- **No leakage:** fit imputers/scalers/selectors/SMOTE on training data only (use `Pipeline`s).
- **Cost metric:** `src/metrics.py` defaults to cost(FN) = 10, cost(FP) = 1. This ratio is an assumption —
  report it and include a sensitivity analysis.
- Save figures to `outputs/figures/` with descriptive names (e.g. `05_cost_vs_threshold.png`).

## Team roles

| Member | Role | Main notebooks |
|---|---|---|
| _TBD_ | Project lead / data & EDA | 01_EDA |
| _TBD_ | Preprocessing & feature selection | 02, 03 |
| _TBD_ | Modelling & tuning | 04 |
| _TBD_ | Evaluation & interpretation | 05, 06 |
