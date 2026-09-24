"""
Telco Customer Churn - Step 4: Hyperparameter Tuning via RandomizedSearchCV
Pipeline:
 1. Load data/cleaned_churn.csv with exact same preprocessing as 03_modeling.ipynb
 2. Stratified 80/20 train/test split (random_state=42)
 3. Fit StandardScaler on X_train only, transform both train and test
 4. RandomizedSearchCV (5-Fold Stratified CV, scoring='roc_auc', n_iter=20)
    - Random Forest (class_weight='balanced')
    - Gradient Boosting (balanced sample_weight)
 5. Benchmark comparisons: Baseline vs Default Balanced vs Tuned
"""

from pathlib import Path
import time
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
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import confusion_matrix

    print("=" * 85)
    print(" TELCO CHURN PREDICTION - STEP 4: HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("=" * 85)

    # 1. Load cleaned data
    csv_path = locate_dataset("cleaned_churn.csv")
    print(f"\n[1] Loading cleaned dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. Categorical one-hot encoding (identical to 03_modeling.ipynb)
    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

    # 4. Stratified 80/20 train/test split (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[2] Identical Stratified 80/20 Split: Train={len(X_train):,}, Test={len(X_test):,}")

    # 5. Fit StandardScaler strictly on X_train
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 5-fold cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sample_weights_train = compute_sample_weight("balanced", y_train)

    # -------------------------------------------------------------------------
    # 6. Tune Random Forest Classifier
    # -------------------------------------------------------------------------
    print("\n[3] Tuning Random Forest (5-fold CV, scoring='roc_auc', n_iter=20)...")
    param_dist_rf = {
        "n_estimators": [100, 150, 200, 250],
        "max_depth": [4, 6, 8, 10, 12],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", "log2", 0.4, 0.6],
    }
    rf_base = RandomForestClassifier(class_weight="balanced", random_state=42)
    rs_rf = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist_rf,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    rs_rf.fit(X_train_scaled, y_train)
    t_rf = time.time() - t0

    print(f"    * RF Tuning completed in {t_rf:.1f}s")
    print(f"    * Best 5-Fold CV ROC-AUC: {rs_rf.best_score_:.4f}")
    print("    * Optimal Parameters:")
    for k, v in rs_rf.best_params_.items():
        print(f"        - {k:<20}: {v}")

    # -------------------------------------------------------------------------
    # 7. Tune Gradient Boosting Classifier
    # -------------------------------------------------------------------------
    print("\n[4] Tuning Gradient Boosting (5-fold CV, scoring='roc_auc', n_iter=20)...")
    param_dist_gb = {
        "n_estimators": [100, 150, 200, 250],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "min_samples_split": [5, 10, 20, 30],
        "min_samples_leaf": [2, 4, 8, 12],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }
    gb_base = GradientBoostingClassifier(random_state=42)
    rs_gb = RandomizedSearchCV(
        estimator=gb_base,
        param_distributions=param_dist_gb,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    rs_gb.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
    t_gb = time.time() - t0

    print(f"    * GB Tuning completed in {t_gb:.1f}s")
    print(f"    * Best 5-Fold CV ROC-AUC: {rs_gb.best_score_:.4f}")
    print("    * Optimal Parameters:")
    for k, v in rs_gb.best_params_.items():
        print(f"        - {k:<20}: {v}")

    # -------------------------------------------------------------------------
    # 8. Evaluate on Unseen Test Set
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 85)
    print(" UNSEEN 20% TEST SET PERFORMANCE BENCHMARK")
    print("=" * 85)

    # Reference untuned balanced models for direct before/after comparison
    rf_untuned = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42)
    rf_untuned.fit(X_train_scaled, y_train)

    gb_untuned = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb_untuned.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)

    tuned_rf_best = rs_rf.best_estimator_
    tuned_gb_best = rs_gb.best_estimator_

    comp_results = {
        "Random Forest (Default Bal)": evaluate_model(rf_untuned, X_test_scaled, y_test),
        "Random Forest (Tuned)":       evaluate_model(tuned_rf_best, X_test_scaled, y_test),
        "Gradient Boosting (Default Bal)": evaluate_model(gb_untuned, X_test_scaled, y_test),
        "Gradient Boosting (Tuned)":       evaluate_model(tuned_gb_best, X_test_scaled, y_test),
    }

    comp_df = pd.DataFrame(comp_results).T
    print(comp_df.round(4).to_string())

    print("\n" + "-" * 75)
    print(" CHURN DETECTION EFFICIENCY (TEST SET: 374 TOTAL CHURNERS)")
    print("-" * 75)
    for name, model in [
        ("Random Forest (Default Bal)", rf_untuned),
        ("Random Forest (Tuned)", tuned_rf_best),
        ("Gradient Boosting (Default Bal)", gb_untuned),
        ("Gradient Boosting (Tuned)", tuned_gb_best),
    ]:
        cm = confusion_matrix(y_test, model.predict(X_test_scaled))
        caught, missed = cm[1, 1], cm[1, 0]
        print(f"{name:<34} -> Caught: {caught:3d} ({caught/374*100:.1f}%) | Missed: {missed:3d} ({missed/374*100:.1f}%)")

    # Top Feature Importance from Best Tuned Gradient Boosting Model
    print("\n" + "-" * 75)
    print(" TOP 8 CHURN PREDICTORS (From Best Tuned Gradient Boosting Model)")
    print("-" * 75)
    feat_imp = pd.Series(tuned_gb_best.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)
    for feat, val in feat_imp.head(8).items():
        print(f"   {feat:<35}: {val:.4f} ({val*100:.1f}%)")

    print("\n" + "=" * 85)
    print(" STEP 4 HYPERPARAMETER TUNING COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    main()
