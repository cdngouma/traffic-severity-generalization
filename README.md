# Traffic Accident Severity – Cross-City Generalization Study
## Project Outcome
This project evaluates whether a machine learning model trained on traffic data from various U.S. cities can accurately predict accident severity in a **geographically unseen city** (Boston, MA).

The objective is not maximizing in-sample accuracy, but testing real-world transferability—ensuring the model generalizes to new environments with different infrastructure, traffic patterns, and reporting practices.

## Technical Approach
### 1. Data Processing
- **Data Audit:** Removed ~102k duplicated accident reports and evaluated missingness, label stability, and temporal drift.
- **Label Stabilization:** Restricted modeling to 2016–2018 due to structural severity drift in later years.
- **Feature Engineering:**
    - Derived road Speed_Class (High / Medium / Low) from street names to capture traffic regime without memorizing locations.
    - Engineered cyclical time features and weekend indicators.
    - Used weather and infrastructure signals with stable cross-city behavior.
- **Generalization Strategy:**
    - Removed high-cardinality geographic identifiers (e.g., street names, coordinates).
    - Excluded post-accident variables to prevent leakage.
    - Applied cross-city validation using GroupKFold.

### 2. Modeling
- **Algorithms:** Logistic Regression (interpretable baseline) and XGBoost (non-linear benchmark).
- **Validation:** Cross-validation grouped by city.
- **Evaluation:** Boston fully held out for geographic generalization testing.
- **Metrics:** ROC-AUC (ranking ability), PR-AUC, F1, Recall.
- **Threshold Optimization:** Adjusted decision threshold to account for changing class prevalence in deployment.

## Key Results (Boston Holdout)
After threshold optimization:
- ROC-AUC: **0.801**
- PR-AUC (Precision-Recall): **0.688**
- F1-Score: **0.721**
- Recall: **0.776**

The model maintains strong ranking performance and balanced error trade-offs when applied to an unseen city.

### Temporal Robustness (2019–2023 Boston)
To evaluate resilience to distribution drift, the trained model was applied to post-2018 Boston data:
- ROC-AUC: **0.79**
- Recall: **0.73**

Despite a reduced share of high-severity accidents, ranking performance remained stable, indicating robustness to moderate label and base-rate shifts.

## Project Structure
```graphql
├── data/
│   ├── raw/                # Original traffic accident dataset
│   └── preprocessed/       # Processed data ready for modeling
├── models/                 # Serialized model artifacts
├── notebooks/
│   ├── 01_data_audit.ipynb # EDA, data cleaning, and assumptions
│   └── 02_modeling.ipynb   # Feature engineering pipeline, modeling, and evaluation
└── scripts/
    └── preprocess.py       # Production script for data transformation
```

## How to Run
The project pipeline includes a preprocessing script to clean and format raw traffic data.

To run the standard preprocessing pipeline:
```bash
python scripts/preprocess.py
```

To run with specific flags (e.g., using post-2019 data or focusing specifically on Boston):
```bash
python scripts/preprocess.py --post --boston
```
