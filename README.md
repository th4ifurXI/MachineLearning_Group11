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
  04_Modelling.ipynb      LR, SVM, RF, XGBoost x (class weights | SMOTE), RandomizedSearchCV on PR-AUC,
                          time-ordered CV (TimeSeriesSplit(4))
  05_Evaluation.ipynb     Cost-optimal thresholds (OOF), test-set metrics, 5:1/10:1/20:1 sensitivity, drift check
  06_Interpretation.ipynb SHAP + permutation importance (top 10 sensors); energy and CO2e savings
src/
  config.py               Paths and RANDOM_STATE = 42
  data_loader.py          load_secom(), split_xy()
  cost_metric.py          Energy cost (FN x C_FN + FP x C_FP), cost curves, best threshold, evaluate()
  metrics.py              Old names for the cost metric (kept so 04 still runs)
  preprocessing.py        time_train_test_split(), build_preprocessing_pipeline() and custom drop steps
  plotting.py             Shared chart style and save_fig() (300 dpi)
  validation.py           time_cv() (TimeSeriesSplit(4)) and time_oof_scores() for time-ordered CV
outputs/
  figures/                Saved plots for the report
  models/                 selected_features.json, best_models.json, CV results, rankings (committed);
                          *.joblib fitted objects (git-ignored - re-run 02 and 04 to create)
  04_cv_results.csv       Best CV scores per model x imbalance option (+ all_candidates version)
  05_*.csv                Test results, cost sensitivity, tuned thresholds, test-set scores
  *_shuffled_cv.csv       Earlier results from shuffled StratifiedKFold, kept for comparison
  06_*.csv                Top 10 sensors, energy savings, energy context and sensitivity tables
```

The fitted `.joblib` files are not in git. After cloning, run `02_Preprocessing.ipynb` (needed by 03) and
`04_Modelling.ipynb` (~12 min; creates the four tuned models used by 05 and 06). Each saved model is a full
pipeline: raw 590 sensors in, fail score out.

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
- **Validate forward in time:** use `src/validation.time_cv()`, not shuffled K-fold. The process drifts, and
  shuffled CV gave optimistic scores (ROC-AUC ~0.73 vs ~0.6 on later wafers).
- **Cost metric:** `src/cost_metric.py` defaults to cost(FN) = 10, cost(FP) = 1. This ratio is an assumption —
  report it and include a sensitivity analysis.
- Save figures to `outputs/figures/` with descriptive names (e.g. `05_cost_vs_threshold.png`).
