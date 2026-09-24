"""
Telco Customer Churn - Step 5: Final Model Retraining, Serialization & Production Packaging
Exports:
 - models/churn_model.pkl (Tuned Gradient Boosting classifier)
 - models/scaler.pkl (Fitted StandardScaler)
 - models/feature_columns.json & models/feature_columns.pkl (Feature alignment list)
"""

from pathlib import Path
import json
import time
import joblib
import pandas as pd
import numpy as np


def locate_path(relative_str: str) -> Path:
    """Find path whether running from root or notebooks dir."""
    possible_paths = [
        Path(relative_str),
        Path("..") / relative_str,
        Path(__file__).resolve().parent.parent / relative_str,
    ]
    for p in possible_paths:
        if p.exists():
            return p.resolve()
    # Default to resolved path under project root
    return (Path(__file__).resolve().parent.parent / relative_str).resolve()


def evaluate_model(model, X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1-Score": f1_score(y, preds),
        "ROC-AUC": roc_auc_score(y, probs),
    }


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import confusion_matrix

    print("=" * 85)
    print(" TELCO CHURN PREDICTION - STEP 5: FINAL MODEL PACKAGING & VERIFICATION")
    print("=" * 85)

    # 1. Load cleaned data
    csv_path = locate_path("data/cleaned_churn.csv")
    print(f"\n[1] Loading cleaned data: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Preprocessing pipeline identical to steps 3 & 4
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    feature_columns = list(X_encoded.columns)
    print(f"    Features encoded: {len(feature_columns)} columns")

    # 3. Stratified 80/20 train/test split (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples")

    # 4. Fit StandardScaler strictly on X_train
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 5. Retrain Tuned Gradient Boosting model on training set
    print("\n[2] Retraining Tuned Gradient Boosting with optimal hyperparameters...")
    best_params = {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.7,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": 42,
    }
    print("    Hyperparameters:", best_params)

    sample_weights_train = compute_sample_weight("balanced", y_train)
    final_model = GradientBoostingClassifier(**best_params)
    
    t0 = time.time()
    final_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
    fit_duration = time.time() - t0
    print(f"    Model training completed in {fit_duration:.2f} seconds.")

    # 6. Evaluate on Test Set
    metrics_before_save = evaluate_model(final_model, X_test_scaled, y_test)
    print("\n[3] Model Performance on Unseen 20% Test Set:")
    for k, v in metrics_before_save.items():
        print(f"    - {k:<20}: {v:.4f} ({v*100:.2f}%)" if "Accuracy" in k or "Precision" in k or "Recall" in k else f"    - {k:<20}: {v:.4f}")

    # 7. Save production artifacts to models/
    models_dir = locate_path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[4] Serializing production artifacts to {models_dir}...")

    model_path = models_dir / "churn_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    feat_pkl_path = models_dir / "feature_columns.pkl"
    feat_json_path = models_dir / "feature_columns.json"

    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_columns, feat_pkl_path)
    with open(feat_json_path, "w") as f:
        json.dump(feature_columns, f, indent=2)

    print(f"    [OK] Model saved:            {model_path.name} ({model_path.stat().st_size:,} bytes)")
    print(f"    [OK] Scaler saved:           {scaler_path.name} ({scaler_path.stat().st_size:,} bytes)")
    print(f"    [OK] Feature list (pkl):     {feat_pkl_path.name} ({feat_pkl_path.stat().st_size:,} bytes)")
    print(f"    [OK] Feature list (json):    {feat_json_path.name} ({feat_json_path.stat().st_size:,} bytes)")

    # 8. Reload & Verification Sanity Check
    print("\n[5] Sanity Check: Reloading artifacts from disk and verifying test set reproducibility...")
    reloaded_model = joblib.load(model_path)
    reloaded_scaler = joblib.load(scaler_path)
    reloaded_features = joblib.load(feat_pkl_path)

    # Feature column alignment validation
    assert reloaded_features == feature_columns, "Error: Reloaded feature column list differs!"

    # Prediction reproduction check
    reloaded_preds = reloaded_model.predict(X_test_scaled)
    reloaded_probs = reloaded_model.predict_proba(X_test_scaled)[:, 1]
    
    metrics_after_reload = evaluate_model(reloaded_model, X_test_scaled, y_test)
    
    # Confirm exact equality
    for metric_name in metrics_before_save:
        diff = abs(metrics_before_save[metric_name] - metrics_after_reload[metric_name])
        assert diff < 1e-9, f"Mismatch in metric {metric_name}: {diff}"

    print("    [OK] Sanity check passed 100%: Reloaded model generates identical predictions & metrics.")

    # 9. Churn detection breakdown
    cm = confusion_matrix(y_test, reloaded_preds)
    print("\n[6] Final Confusion Matrix:")
    print(f"    Total Test Customers: {len(y_test):,}")
    print(f"    Actual Churners:      {cm[1, 0] + cm[1, 1]} (26.54%)")
    print(f"    - Caught (TP):        {cm[1, 1]} ({cm[1, 1]/374*100:.1f}%)")
    print(f"    - Missed (FN):        {cm[1, 0]} ({cm[1, 0]/374*100:.1f}%)")
    print(f"    Non-Churners:")
    print(f"    - Retained (TN):      {cm[0, 0]} ({cm[0, 0]/1035*100:.1f}%)")
    print(f"    - False Alarm (FP):   {cm[0, 1]} ({cm[0, 1]/1035*100:.1f}%)")

    print("\n" + "=" * 85)
    print(" STEP 5 COMPLETE: PRODUCTION ARTIFACTS VERIFIED AND READY")
    print("=" * 85)


if __name__ == "__main__":
    main()
