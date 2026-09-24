# Customer Churn Prediction: End-to-End Machine Learning Pipeline

An enterprise machine learning project to identify telecom customers at risk of churn, prioritize retention outreach, and diagnose the key operational drivers behind customer attrition.

Built using the **Telco Customer Churn Dataset** (Kaggle / BlastChar, 7,043 customer accounts).

---

## Executive Summary & Business Impact

In subscription telecommunications, acquiring a new customer is **5x to 7x more expensive** than retaining an existing one. Classification errors have heavily asymmetric business costs:
- **Cost of a False Negative (Missed Churner):** The customer departs unnoticed, permanently erasing their Customer Lifetime Value ($1,200+ on average) and requiring costly marketing replacement.
- **Cost of a False Positive (False Alarm):** A loyal customer receives a proactive retention check-in or service offer, incurring only a negligible operational cost ($5–$15).

Because raw accuracy is misleading on an imbalanced dataset (~26.5% churn), our pipeline was engineered with **cost-sensitive class balancing** and **hyperparameter optimization** to maximize **Recall on Churn** while preserving high discriminative ability (**ROC-AUC**).

### Key Results
- **Final Model:** Tuned Gradient Boosting Classifier
- **Churn Recall:** **80.21%** (catches **300 out of 374** churners in the 20% test holdout, cutting missed churners by more than half)
- **ROC-AUC Score:** **0.8476**
- **F1-Score:** **0.6303**

---

## Repository Structure

```
churn-prediction/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw source dataset (7,043 rows, 21 columns)
│   └── cleaned_churn.csv                     # Cleaned & prepared dataset (7,043 rows, 20 columns)
├── models/
│   ├── churn_model.pkl                       # Serialized Champion Gradient Boosting model
│   ├── scaler.pkl                            # Fitted StandardScaler (leakage-free)
│   ├── feature_columns.json                  # Ordered feature schema for inference alignment
│   └── feature_columns.pkl                   # Pickled feature schema
├── notebooks/
│   ├── 01_exploration.ipynb                  # Step 1: Exploratory data analysis (EDA)
│   ├── 01_exploration.py                     # Step 1: Executable script
│   ├── 02_cleaning.ipynb                     # Step 2: Data cleaning & preprocessing
│   ├── 02_cleaning.py                        # Step 2: Executable script
│   ├── 03_modeling.ipynb                     # Step 3: Baseline & class-balanced modeling
│   ├── 03_modeling.py                        # Step 3: Executable script
│   ├── 04_tuning.ipynb                       # Step 4: Hyperparameter tuning via RandomizedSearchCV
│   ├── 04_tuning.py                          # Step 4: Executable script
│   ├── 05_final_model.ipynb                  # Step 5: Final retraining, packaging & summary
│   ├── 05_final_model.py                     # Step 5: Executable script
│   └── 06_test_inference.py                  # Step 6: End-to-end inference verification script
└── README.md                                 # Project documentation
```

---

## How to Reproduce Results (Execution Order)

You can run either the interactive Jupyter notebooks inside `notebooks/` or execute the Python companion scripts directly from your terminal in the following sequence:

### Step 1: Exploratory Data Analysis
Examines shape, data types, standard nulls, distributions, and detects the 11 blank whitespace values in `TotalCharges` (customers with `tenure == 0`).
```bash
python notebooks/01_exploration.py
```

### Step 2: Data Cleaning & Feature Preparation
Converts `TotalCharges` to `float64` (imputing `0.0` for new customers), drops non-predictive `customerID`, simplifies multi-level services to clean binary Yes/No, converts `Churn` to integer (1/0), and saves `data/cleaned_churn.csv`.
```bash
python notebooks/02_cleaning.py
```

### Step 3: Baseline & Class-Balanced Modeling
Applies one-hot encoding (`drop_first=True`), performs stratified 80/20 train/test splitting (`random_state=42`), fits `StandardScaler` strictly on `X_train` to prevent data leakage, and benchmarks default vs. class-balanced versions of Logistic Regression, Random Forest, and Gradient Boosting.
```bash
python notebooks/03_modeling.py
```

### Step 4: Hyperparameter Tuning
Performs 5-Fold Stratified Cross-Validation using `RandomizedSearchCV` (`n_iter=20`, `scoring='roc_auc'`) to optimize hyperparameters for Random Forest and Gradient Boosting while maintaining class-weight balancing.
```bash
python notebooks/04_tuning.py
```

### Step 5: Final Production Model Packaging
Retrains the tuned Gradient Boosting model on the training set, serializes all production artifacts to `models/`, validates exact reload reproduction, and generates business driver interpretations.
```bash
python notebooks/05_final_model.py
```

### Step 6: End-to-End Inference Verification
Runs end-to-end inference on sample customer profiles across low, moderate, and high risk tiers, and re-validates the test holdout metrics against benchmark standards.
```bash
python notebooks/06_test_inference.py
```

---

## Model Benchmark & Evolution Table (Unseen 20% Test Set, $N = 1,409$)

| Stage | Model Configuration | Accuracy | Precision | Recall (Churn=1) | F1-Score | ROC-AUC | Churners Caught (of 374) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Logistic Regression | **80.55%** | 65.82% | 55.62% | 0.6029 | 0.8422 | 208 (55.6%) |
| **Baseline** | Random Forest | **80.70%** | **67.83%** | 51.87% | 0.5879 | 0.8428 | 194 (51.9%) |
| **Baseline** | Gradient Boosting | 80.27% | 66.22% | 52.41% | 0.5851 | 0.8433 | 196 (52.4%) |
| **Balanced** | Logistic Regression (Bal) | 74.02% | 50.69% | 78.61% | 0.6164 | 0.8419 | 294 (78.6%) |
| **Balanced** | Random Forest (Bal) | 75.87% | 53.16% | 76.47% | 0.6272 | 0.8417 | 286 (76.5%) |
| **Balanced** | Gradient Boosting (Bal) | 74.52% | 51.32% | 77.81% | 0.6185 | 0.8388 | 291 (77.8%) |
| **Tuned** | Random Forest (Tuned) | 75.73% | 52.85% | 79.41% | **0.6346** | 0.8459 | 297 (79.4%) |
| **Tuned (Champion)** | **Gradient Boosting (Tuned)** | **75.02%** | **51.90%** | **80.21%** | **0.6303** | **0.8476** | **300 (80.2%)** |

---

## Top Drivers of Customer Churn

Feature importance analysis from our final Gradient Boosting model highlights the following core factors:

1. **Contract Type (21.8% Importance):**
   - Customers on month-to-month contracts churn at ~42.7%, compared to less than 3% for customers on two-year agreements.
2. **Customer Tenure (20.7% Importance):**
   - The first 6–12 months represent the highest churn risk period. Beyond 24 months, churn drops sharply.
3. **Internet Service Type — Fiber Optic (12.2% Importance):**
   - Fiber Optic subscribers exhibit a 41.9% churn rate (more than double DSL subscribers at 19.0%), pointing to price sensitivity or service reliability friction.
4. **Contract Commitment — One Year (11.1% Importance):**
   - Moving from month-to-month to a 1-year contract cuts churn probability by more than half.
5. **Billing Friction — Electronic Check (5.3% Importance):**
   - Customers paying manually via electronic checks churn at ~45%, compared to ~16% for automatic payment methods.

---

## Actionable Business Recommendations

1. **Incentivize Annual Contract Conversions:**
   - Offer targeted discounts (e.g. 15% off or free speed boost) to month-to-month customers in exchange for signing a 1- or 2-year contract.
2. **First 90 Days Customer Success Program:**
   - Deploy automated onboarding check-ins and free premium tech support during months 1–3 to reduce early-tenure drop-off.
3. **Fiber Optic Service Review & Auto-Pay Incentive:**
   - Investigate service quality and customer satisfaction for Fiber Optic accounts.
   - Provide a one-time bill credit ($10–$15) to transition electronic check users onto automated credit card or bank payments.

---

## Quickstart: Loading the Model for Inference

```python
import json
import joblib
import pandas as pd

# Load saved production artifacts
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Ensure new customer dataframe X_new is one-hot encoded and aligned
# X_new_encoded = pd.get_dummies(X_new, drop_first=True).reindex(columns=feature_columns, fill_value=0)
# X_new_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_new_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Generate churn probabilities & predictions
# churn_probabilities = model.predict_proba(X_new_encoded)[:, 1]
# churn_predictions = (churn_probabilities >= 0.5).astype(int)
```
