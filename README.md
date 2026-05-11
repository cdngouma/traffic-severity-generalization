# Traffic Accident Severity – Cross-City Generalization Study

A machine learning project focused on **geographic generalization** in accident severity prediction.

Instead of optimizing for in-sample performance, this project evaluates whether a model trained on multiple U.S. cities can generalize to a **fully unseen city** (Boston, MA) and remain robust under later temporal shift.

## Overview

The goal of this project is to test whether accident severity models can transfer across cities with different road infrastructure, traffic patterns, and reporting practices.

Rather than maximizing performance on random train/test splits, the project is designed around a harder and more realistic question:

> Can a model trained on one set of cities still perform well in a geographically unseen city?

## Why this project is different

- **Geographic holdout evaluation:** Boston is fully excluded from training and used only for final testing.
- **Leakage-resistant design:** High-cardinality location identifiers and post-accident features are removed.
- **Robustness focus:** The model is also tested on later Boston data (2019–2023) to assess temporal stability under distribution shift.

## Technical Approach

### 1. Data Processing

- **Data audit:** Removed ~102k duplicated accident reports and examined missingness, label stability, and temporal drift.
- **Label stabilization:** Restricted modeling to 2016–2018 due to structural severity drift in later years.
- **Feature engineering:**
  - Derived `Speed_Class` (High / Medium / Low) from street names to capture road type without memorizing locations
  - Engineered cyclical time features and weekend indicators
  - Used weather and infrastructure variables with stable cross-city behavior
- **Generalization strategy:**
  - Removed high-cardinality geographic identifiers (e.g., street names, coordinates)
  - Excluded post-accident variables to prevent leakage
  - Applied grouped cross-validation by city

### 2. Modeling

- **Models:** Logistic Regression (interpretable baseline) and XGBoost (non-linear benchmark)
- **Validation:** GroupKFold by city
- **Final evaluation:** Boston fully held out for geographic generalization testing
- **Metrics:** ROC-AUC, PR-AUC, F1, Recall
- **Threshold optimization:** Adjusted the classification threshold to better reflect deployment class prevalence

## Key Results

## Key Results

| Evaluation Setting | ROC-AUC | PR-AUC | F1-score | Recall |
|---|---:|---:|---:|---:|
| **Boston Holdout (unseen city, 2016–2018)** | **0.801** | **0.688** | **0.721** | **0.776** |
| **Boston Temporal Robustness (2019–2023)** | **0.790** | — | — | **0.730** |

The model maintains strong performances when transferred to a geographically unseen city and remains relatively stable under later temporal distribution shift.

## Project Structure

```text
├── data/
│   ├── raw/                # Original traffic accident dataset
│   └── preprocessed/       # Processed data ready for modeling
├── models/                 # Serialized model artifacts
├── notebooks/
│   ├── 01_data_audit.ipynb # Data cleaning, checks, and assumptions
│   └── 02_modeling.ipynb   # Feature engineering, modeling, and evaluation
└── scripts/
    └── preprocess.py       # Preprocessing pipeline
```

## How to Run
### Preprocess the raw data

```bash
python scripts/preprocess.py
```

### Optional preprocessing flags

```bash
python scripts/preprocess.py --post --boston
```

### Explore and reproduce modeling results

Use the notebooks in `notebooks/`:
- `01_data_audit.ipynb` for data cleaning and assumptions
- `02_modeling.ipynb` for feature engineering, training, and evaluation
