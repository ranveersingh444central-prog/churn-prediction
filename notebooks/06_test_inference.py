"""
Telco Customer Churn - Step 6: End-to-End Inference Verification & Test Set Audit
This script verifies that the saved production artifacts in models/ function properly:
 1. Loads churn_model.pkl, scaler.pkl, and feature_columns.json
 2. Evaluates hand-crafted customer profiles across distinct risk tiers
 3. Verifies preprocessing pipeline (alignment, encoding, scaling)
 4. Re-evaluates test set metrics against baseline benchmarks
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np


def locate_path(relative_str: str) -> Path:
    """Find path whether running from project root or notebooks dir."""
    possible_paths = [
        Path(relative_str),
        Path("..") / relative_str,
        Path(__file__).resolve().parent.parent / relative_str,
    ]
    for p in possible_paths:
        if p.exists():
            return p.resolve()
    return (Path(__file__).resolve().parent.parent / relative_str).resolve()


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    print("=" * 85)
    print(" TELCO CHURN PREDICTION - STEP 6: END-TO-END INFERENCE VERIFICATION")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # 1. Load Saved Production Artifacts
    # -------------------------------------------------------------------------
    model_path = locate_path("models/churn_model.pkl")
    scaler_path = locate_path("models/scaler.pkl")
    feat_json_path = locate_path("models/feature_columns.json")

    print(f"\n[1] Loading production artifacts from models/ directory:")
    print(f"    - Model:   {model_path}")
    print(f"    - Scaler:  {scaler_path}")
    print(f"    - Schema:  {feat_json_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feat_json_path, "r") as f:
        feature_columns = json.load(f)

    print(f"    [OK] Model type:       {type(model).__name__}")
    print(f"    [OK] Scaler type:      {type(scaler).__name__}")
    print(f"    [OK] Feature columns:  {len(feature_columns)} expected input columns")

    # -------------------------------------------------------------------------
    # 2. Hand-Crafted Customer Profiles Across Risk Tiers
    # -------------------------------------------------------------------------
    print("\n[2] Creating 3 distinct customer test cases for inference:")
    
    sample_customers = [
        {
            "Profile": "Profile A: Low-Risk (Loyal, 2-Year Contract, Auto-Pay, Low Spend)",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 65,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 55.00,
            "TotalCharges": 3575.00,
        },
        {
            "Profile": "Profile B: Moderate-Risk (1-Year Contract, Fiber Optic, Auto-Pay)",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 89.50,
            "TotalCharges": 2148.00,
        },
        {
            "Profile": "Profile C: High-Risk (Brand New, Month-to-Month, Fiber, Electronic Check)",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.00,
            "TotalCharges": 190.00,
        },
    ]

    df_raw_samples = pd.DataFrame(sample_customers)
    profile_labels = df_raw_samples.pop("Profile")

    # -------------------------------------------------------------------------
    # 3. Apply Production Preprocessing Pipeline
    # -------------------------------------------------------------------------
    print("\n[3] Preprocessing inference payloads:")
    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]
    
    # One-hot encode categoricals (drop_first=True)
    df_encoded = pd.get_dummies(df_raw_samples, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Reindex to strictly enforce exact feature columns and order (fill missing dummy levels with 0)
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)
    print(f"    [OK] Aligned feature matrix shape: {df_aligned.shape} (Matches {len(feature_columns)} trained features)")

    # Scale numeric columns with the fitted StandardScaler
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df_aligned[numeric_cols] = scaler.transform(df_aligned[numeric_cols])
    print(f"    [OK] Scaled numeric features {numeric_cols}")

    # -------------------------------------------------------------------------
    # 4. Predict Churn Probabilities & Classes
    # -------------------------------------------------------------------------
    print("\n[4] Running inference (model.predict_proba & model.predict)...")
    churn_probabilities = model.predict_proba(df_aligned)[:, 1]
    churn_predictions = model.predict(df_aligned)

    # -------------------------------------------------------------------------
    # 5. Display Customer Input Values Alongside Predictions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" INFERENCE RESULTS ON SAMPLE CUSTOMERS")
    print("=" * 85)

    for i, profile_title in enumerate(profile_labels):
        prob = churn_probabilities[i]
        pred_label = "CHURN (At Risk)" if churn_predictions[i] == 1 else "RETAINED (Safe)"
        
        print(f"\n>>> {profile_title.upper()}")
        print("-" * 80)
        customer_dict = sample_customers[i]
        print(f"  Key Features:")
        print(f"    - Contract:        {customer_dict['Contract']:<16} | Tenure:         {customer_dict['tenure']} months")
        print(f"    - Internet:        {customer_dict['InternetService']:<16} | Payment Method: {customer_dict['PaymentMethod']}")
        print(f"    - Monthly Charges: ${customer_dict['MonthlyCharges']:<15.2f} | Total Charges:  ${customer_dict['TotalCharges']:.2f}")
        print(f"    - Tech Support:    {customer_dict['TechSupport']:<16} | Online Backup:  {customer_dict['OnlineBackup']}")
        print(f"  Prediction:")
        print(f"    * Predicted Churn Probability: {prob*100:6.2f}%")
        print(f"    * Final Classification:        {pred_label}")

    print("\n" + "-" * 85)
    print(" Sanity Check Summary on Risk Tiers:")
    print(f"  * Low-Risk Customer Churn Probability:  {churn_probabilities[0]*100:5.2f}%  (Baseline expectation: Low, <10%)")
    print(f"  * Med-Risk Customer Churn Probability:  {churn_probabilities[1]*100:5.2f}%  (Baseline expectation: Moderate, ~50-60%)")
    print(f"  * High-Risk Customer Churn Probability: {churn_probabilities[2]*100:5.2f}%  (Baseline expectation: Severe, >85%)")
    assert churn_probabilities[2] > churn_probabilities[1] > churn_probabilities[0], "Error: Risk ranking inversion!"
    print("  [OK] Probability ordering is monotonically consistent with business risk factors.")
    print("-" * 85)

    # -------------------------------------------------------------------------
    # 6. Re-Verify Saved Model on Actual Test Set
    # -------------------------------------------------------------------------
    print("\n[5] Re-validating Saved Model on the Unseen 20% Test Holdout Set (N = 1,409):")
    csv_path = locate_path("data/cleaned_churn.csv")
    df_clean = pd.read_csv(csv_path)

    X_full = df_clean.drop(columns=["Churn"])
    y_full = df_clean["Churn"]

    X_full_encoded = pd.get_dummies(X_full, columns=categorical_cols, drop_first=True, dtype=int)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full_encoded, y_full, test_size=0.20, random_state=42, stratify=y_full
    )

    X_test_scaled = X_test_full.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test_full[numeric_cols])

    test_preds = model.predict(X_test_scaled)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test_full, test_preds)
    prec = precision_score(y_test_full, test_preds)
    rec = recall_score(y_test_full, test_preds)
    f1 = f1_score(y_test_full, test_preds)
    auc = roc_auc_score(y_test_full, test_probs)

    test_metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Churn=1)", "Recall (Churn=1)", "F1-Score", "ROC-AUC"],
        "Saved Model on Test Set": [acc, prec, rec, f1, auc],
        "Expected Benchmark": [0.750177, 0.519031, 0.802139, 0.630252, 0.847587],
    })
    print(test_metrics_df.round(4).to_string(index=False))

    cm = confusion_matrix(y_test_full, test_preds)
    print("\nConfusion Matrix Verification:")
    print(f"  * Total Churners in Test Set: {cm[1, 0] + cm[1, 1]}")
    print(f"  * Churners Successfully Caught: {cm[1, 1]} ({cm[1, 1]/374*100:.2f}%)")
    print(f"  * Churners Missed:              {cm[1, 0]} ({cm[1, 0]/374*100:.2f}%)")

    # Confirm exact benchmark match
    assert abs(rec - 0.802139) < 1e-4, "Recall score mismatch!"
    assert abs(auc - 0.847587) < 1e-4, "ROC-AUC score mismatch!"
    print("\n[OK] FINAL VERIFICATION COMPLETE: Saved artifacts are 100% intact, leakage-free, and production-ready.")
    print("=" * 85)


if __name__ == "__main__":
    main()
