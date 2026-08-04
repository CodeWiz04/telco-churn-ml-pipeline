# Telco Customer Churn — ML Pipeline

A reproducible, end-to-end machine learning pipeline that trains, tunes, and compares classical ML
algorithms to predict customer churn for a telecom provider.

---

## Problem Type & Target

- **Problem type:** Binary classification
- **Target column:** `Churn` (`Yes` / `No`)
- **Dataset:** [Telco Customer Churn (IBM sample dataset)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 columns, ~27% churn rate (imbalanced).

---

## Project Structure

```
ml-algorithms-task2/
│
├── data/
│   ├── raw/                     # original downloaded CSV, untouched
│   └── processed/                # cleaned, feature-ready data
│
├── src/
│   ├── preprocessing.py          # build_features(), get_train_test_split()
│   ├── modeling.py               # train_and_evaluate(), tuning logic
│   ├── evaluation.py             # metrics, confusion matrix, plots
│   └── utils.py                  # shared helpers, path constants, logging
│
├── models/
│   └── best_model.joblib         # final trained model, persisted
│
├── results/
│   ├── model_comparison.csv      # all models x all metrics
│   └── confusion_matrix.png
│
├── reports/
│   └── final_report.pdf          # stakeholder-facing summary
│
├── main.py                       # single entry point, runs pipeline end-to-end
├── pyproject.toml                # pinned dependencies
└── README.md
```

---

## Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management with all versions pinned in `pyproject.toml`.

```bash
# 1. Clone the repository
git clone <https://github.com/CodeWiz04/telco-churn-ml-pipeline>
cd ml-algorithms-task2

# 2. Create and activate a virtual environment
uv venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
uv pip install -e .
# or, if adding fresh:
uv add scikit-learn pandas numpy matplotlib seaborn joblib scipy reportlab
```

### Get the dataset

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in `data/raw/`:

```bash
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
```

---

## How to Run

Run the full pipeline end-to-end with a single command:

```bash
python main.py
```

This will, in order:

1. Load and validate the raw dataset (shape, dtypes, missing values, target distribution)
2. Clean the data (fix `TotalCharges`, handle the 11 resulting nulls)
3. Build features (`X`) and target (`y`), one-hot encode categoricals, scale numeric features
4. Split into stratified 80/20 train/test sets
5. Train baseline Logistic Regression and Random Forest models
6. Run 5-fold cross-validation for both models
7. Tune the top candidates with `GridSearchCV`
8. Evaluate all four candidates (baseline + tuned, per model) on Accuracy, Precision, Recall, F1, ROC-AUC
9. Save all metrics to `results/model_comparison.csv` and the confusion matrices to console/plots
10. Persist the best model to `models/best_model.joblib`
11. Answer the four required analysis questions (best model, statistical significance, feature importance, error segments) and print them to console

---

## Key Modeling Decisions

| Decision | Choice | Why |
|---|---|---|
| Encoding | One-hot encoding for all categorical columns | None of the categorical fields (Contract, PaymentMethod, InternetService, etc.) have a natural order, so one-hot avoids implying a false numeric relationship that label/ordinal encoding would create. |
| Scaling | StandardScaler on numeric features, applied only ahead of Logistic Regression | Logistic Regression is gradient/distance-based and sensitive to feature scale; Random Forest splits on thresholds and doesn't need scaling. |
| Class imbalance | Class-weighting (`class_weight='balanced'`) | Preserves all data (no under/oversampling) while penalizing misclassification of the minority churn class more heavily. |
| Train/test split | 80/20, stratified on `Churn` | Keeps the ~27% churn rate consistent in both sets; without stratification the test set could end up with too few churn cases, silently inflating accuracy. |
| Primary metric | Recall | Missing a customer who is about to churn (false negative) is costlier than a false retention-team follow-up on a loyal customer (false positive). Chosen before viewing any results. |
| Baseline model | Logistic Regression | Simple, fast, interpretable — sets the bar that a more complex model must clear to justify its use. |
| Second model | Random Forest | Captures non-linear feature interactions the linear baseline can't; a natural, higher-complexity comparison point. |

---

## Final Results

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | Logistic Regression (Baseline) | 0.7264 | 0.4909 | **0.7941** | 0.6067 | 0.8353 |
| 2 | Logistic Regression (Tuned) | 0.7249 | 0.4893 | 0.7941 | 0.6055 | 0.8354 |
| 3 | Random Forest (Tuned) | 0.7548 | 0.5259 | 0.7861 | 0.6302 | 0.8365 |
| 4 | Random Forest (Baseline) | 0.7740 | 0.5648 | 0.6524 | 0.6055 | 0.8221 |

*(Ranked by primary metric: Recall. Full metrics logged in `results/model_comparison.csv`.)*

**Recommendation:** Ship **Logistic Regression (Baseline)**. It ties for the best Recall (79.41%) of
any candidate, and a paired t-test on the cross-validation fold scores of Logistic Regression vs.
Random Forest returned a p-value of **0.82** — the difference between the two models is **not
statistically significant**. Given equal performance, the simpler, more interpretable, and cheaper-to-
retrain model is the better production choice.

The strongest churn predictors are **tenure**, **total/monthly charges**, and **contract type**
(two-year contracts strongly reduce churn risk). The model's highest error rates occur on
**month-to-month, low-tenure, fiber-optic customers** — the segment with the least commitment and
the least account history, which is also the group most worth prioritizing for retention outreach.

See `reports/final_report.pdf` for the full stakeholder-facing writeup, including confusion matrices,
statistical testing detail, feature importance, and per-segment error analysis.

---

## Notes

- `data/`, `models/*.joblib`, and other large/generated files are excluded via `.gitignore`. Only
  source code, `results/*.csv`/`*.png`, this README, and `reports/final_report.pdf` are pushed to
  the repository.
- Every function in `src/` is documented with docstrings, and data loading / model training steps
  include exception handling for missing files or malformed data.