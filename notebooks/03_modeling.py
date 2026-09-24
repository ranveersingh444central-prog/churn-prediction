"""
Telco Customer Churn - Step 3: Modeling & Evaluation (Updated)
Pipeline:
 1. Load data/cleaned_churn.csv
 2. Split into features (X) and target (y = Churn)
 3. One-hot encode categorical features (drop_first=True)
 4. Stratified 80/20 train/test split (random_state=42)
 5. Feature scaling via StandardScaler (fit on train only!)
 6. Train baseline models: Logistic Regression, Random Forest, Gradient Boosting
 7. Train balanced models: class_weight='balanced' / sample_weight='balanced'
 8. Side-by-side benchmark comparison (Baseline vs Balanced) highlighting Recall & F1
"""

from pathlib import Path
import pandas as pd
import numpy as np


def locate_dataset(filename: str) -> Path:
    """Find file path whether running from project root or notebooks dir."""
    possible_paths = [
        Path("data") / filename,
        Path("../data") / filename,
        Path(__file__).resolve().parent.parent / "data" / filename,
    ]
    for p in possible_paths:
        if p.exists():
            return p.resolve()
    return (Path(__file__).resolve().parent.parent / "data" / filename).resolve()


def evaluate_model(model, X, y, proba=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if proba else None
    return {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1-Score": f1_score(y, preds),
        "ROC-AUC": roc_auc_score(y, probs) if probs is not None else np.nan,
    }


def main():
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import confusion_matrix

    print("=" * 85)
    print(" TELCO CHURN PREDICTION - STEP 3: MODELING & EVALUATION (BASELINE + BALANCED)")
    print(f" scikit-learn version: {sklearn.__version__}")
    print("=" * 85)

    # 1. Load cleaned data
    csv_path = locate_dataset("cleaned_churn.csv")
    print(f"\n[1] Loading cleaned dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 2. Split features and target
    print("\n[2] Splitting features (X) and target (y = Churn)...")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. One-hot encode categorical features (drop_first=True)
    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]
    print(f"\n[3] One-hot encoding {len(categorical_cols)} categorical columns (drop_first=True)...")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    print(f"    Encoded feature matrix: {X_encoded.shape[0]:,} rows, {X_encoded.shape[1]} columns")

    # 4. Stratified 80/20 train/test split
    print("\n[4] Performing stratified 80/20 train-test split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train: {X_train.shape[0]:,} rows (Churn: {y_train.mean()*100:.2f}%)")
    print(f"    Test:  {X_test.shape[0]:,} rows (Churn: {y_test.mean()*100:.2f}%)")

    # 5. Scale numeric features (fit on train ONLY)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    print(f"\n[5] Scaling numeric columns {numeric_cols} via StandardScaler (fit on train only)...")
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # -------------------------------------------------------------------------
    # 6. Part 1: Baseline Models (Unweighted)
    # -------------------------------------------------------------------------
    print("\n[6] Training 3 Baseline Models (Unweighted)...")
    
    # Model 1: Logistic Regression
    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    lr_base.fit(X_train_scaled, y_train)

    # Model 2: Random Forest
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_base.fit(X_train_scaled, y_train)

    # Model 3: Gradient Boosting
    gb_base = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb_base.fit(X_train_scaled, y_train)

    print("\n" + "=" * 70)
    print(" BASELINE MODEL PERFORMANCE (Default / Unweighted)")
    print("=" * 70)
    baseline_metrics = pd.DataFrame({
        "Logistic Regression": evaluate_model(lr_base, X_test_scaled, y_test),
        "Random Forest": evaluate_model(rf_base, X_test_scaled, y_test),
        "Gradient Boosting": evaluate_model(gb_base, X_test_scaled, y_test),
    })
    print(baseline_metrics.round(4).to_string())

    # -------------------------------------------------------------------------
    # 7. Part 2: Balanced Models (Class Balancing)
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print(" CLASS-BALANCED MODELS (Optimized for Churn Recall)")
    print("=" * 70)
    print("Addressing minority class imbalance (~26.5% churn rate):")
    print("  - Logistic Regression: class_weight='balanced'")
    print("  - Random Forest:       class_weight='balanced'")
    print("  - Gradient Boosting:   sample_weight=compute_sample_weight('balanced', y_train)")

    # Model 1 Balanced: Logistic Regression
    lr_bal = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_bal.fit(X_train_scaled, y_train)

    # Model 2 Balanced: Random Forest
    rf_bal = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42)
    rf_bal.fit(X_train_scaled, y_train)

    # Model 3 Balanced: Gradient Boosting with balanced sample weights
    sample_weights_train = compute_sample_weight("balanced", y_train)
    gb_bal = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb_bal.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)

    balanced_metrics = pd.DataFrame({
        "Logistic Regression (Bal)": evaluate_model(lr_bal, X_test_scaled, y_test),
        "Random Forest (Bal)": evaluate_model(rf_bal, X_test_scaled, y_test),
        "Gradient Boosting (Bal)": evaluate_model(gb_bal, X_test_scaled, y_test),
    })
    print("\n" + balanced_metrics.round(4).to_string())

    # -------------------------------------------------------------------------
    # 8. Side-by-Side Comparison & Confusion Matrices
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 85)
    print(" SIDE-BY-SIDE RECALL & F1-SCORE COMPARISON")
    print("=" * 85)
    
    comp_df = pd.DataFrame({
        "Model": [
            "Logistic Regression", "Logistic Regression (Balanced)",
            "Random Forest", "Random Forest (Balanced)",
            "Gradient Boosting", "Gradient Boosting (Balanced)",
        ],
        "Accuracy": [
            baseline_metrics.loc["Accuracy", "Logistic Regression"],
            balanced_metrics.loc["Accuracy", "Logistic Regression (Bal)"],
            baseline_metrics.loc["Accuracy", "Random Forest"],
            balanced_metrics.loc["Accuracy", "Random Forest (Bal)"],
            baseline_metrics.loc["Accuracy", "Gradient Boosting"],
            balanced_metrics.loc["Accuracy", "Gradient Boosting (Bal)"],
        ],
        "Precision": [
            baseline_metrics.loc["Precision", "Logistic Regression"],
            balanced_metrics.loc["Precision", "Logistic Regression (Bal)"],
            baseline_metrics.loc["Precision", "Random Forest"],
            balanced_metrics.loc["Precision", "Random Forest (Bal)"],
            baseline_metrics.loc["Precision", "Gradient Boosting"],
            balanced_metrics.loc["Precision", "Gradient Boosting (Bal)"],
        ],
        "Recall": [
            baseline_metrics.loc["Recall", "Logistic Regression"],
            balanced_metrics.loc["Recall", "Logistic Regression (Bal)"],
            baseline_metrics.loc["Recall", "Random Forest"],
            balanced_metrics.loc["Recall", "Random Forest (Bal)"],
            baseline_metrics.loc["Recall", "Gradient Boosting"],
            balanced_metrics.loc["Recall", "Gradient Boosting (Bal)"],
        ],
        "F1-Score": [
            baseline_metrics.loc["F1-Score", "Logistic Regression"],
            balanced_metrics.loc["F1-Score", "Logistic Regression (Bal)"],
            baseline_metrics.loc["F1-Score", "Random Forest"],
            balanced_metrics.loc["F1-Score", "Random Forest (Bal)"],
            baseline_metrics.loc["F1-Score", "Gradient Boosting"],
            balanced_metrics.loc["F1-Score", "Gradient Boosting (Bal)"],
        ],
        "ROC-AUC": [
            baseline_metrics.loc["ROC-AUC", "Logistic Regression"],
            balanced_metrics.loc["ROC-AUC", "Logistic Regression (Bal)"],
            baseline_metrics.loc["ROC-AUC", "Random Forest"],
            balanced_metrics.loc["ROC-AUC", "Random Forest (Bal)"],
            baseline_metrics.loc["ROC-AUC", "Gradient Boosting"],
            balanced_metrics.loc["ROC-AUC", "Gradient Boosting (Bal)"],
        ],
    })
    print(comp_df.to_string(index=False))

    print("\n" + "-" * 70)
    print(" CHURNERS CAUGHT VS MISSED (CONFUSION MATRICES ON 374 TOTAL CHURNERS):")
    print("-" * 70)
    models_dict = {
        "Logistic Regression (Baseline)": lr_base,
        "Logistic Regression (Balanced)": lr_bal,
        "Random Forest (Baseline)": rf_base,
        "Random Forest (Balanced)": rf_bal,
        "Gradient Boosting (Baseline)": gb_base,
        "Gradient Boosting (Balanced)": gb_bal,
    }
    for name, model in models_dict.items():
        cm = confusion_matrix(y_test, model.predict(X_test_scaled))
        caught = cm[1, 1]
        missed = cm[1, 0]
        print(f"{name:<33} -> Caught: {caught:3d} ({caught/374*100:.1f}%) | Missed: {missed:3d} ({missed/374*100:.1f}%)")

    print("\n" + "=" * 85)
    print(" STEP 3 UPDATE COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    main()
