"""
Churn Intelligence - Production Prediction Engine
Encapsulates artifact loading, input normalization, exact schema alignment,
and inference calculation for the Telco Customer Churn model.
"""

from pathlib import Path
import json
from typing import Dict, Any, Tuple, List, Optional
import joblib
import pandas as pd
import numpy as np


def get_project_root() -> Path:
    """Resolve the absolute root directory of the project."""
    return Path(__file__).resolve().parent.parent


class ChurnPredictor:
    """Production inference engine for Telco Customer Churn prediction."""

    CATEGORICAL_COLS = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]
    NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

    REQUIRED_INPUT_COLUMNS = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges"
    ]
    OPTIONAL_INPUT_COLUMNS = ["customerID", "TotalCharges", "Churn"]

    # Pre-crafted benchmark profiles for quick demonstration & testing
    PRESET_PROFILES = {
        "Low-Risk Customer (Long-term, 2-Year Contract, Auto-Pay)": {
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
        "Moderate-Risk Customer (1-Year Contract, Fiber Optic, Auto-Pay)": {
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
        "High-Risk Customer (New, Month-to-Month, Fiber, Electronic Check)": {
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
    }

    def __init__(self, models_dir: Optional[Path] = None):
        """Load and cache the trained model, fitted scaler, and feature schema."""
        if models_dir is None:
            models_dir = get_project_root() / "models"
        self.models_dir = Path(models_dir)

        model_path = self.models_dir / "churn_model.pkl"
        scaler_path = self.models_dir / "scaler.pkl"
        schema_path = self.models_dir / "feature_columns.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact missing at {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact missing at {scaler_path}")
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema artifact missing at {schema_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(schema_path, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

    def normalize_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize raw user entries, handling compound service categories safely
        and auto-calculating TotalCharges if unspecified.
        """
        data = dict(raw_input)

        # SeniorCitizen normalization (0 or 1)
        senior_val = data.get("SeniorCitizen", 0)
        if isinstance(senior_val, str):
            data["SeniorCitizen"] = 1 if senior_val.strip().lower() in ["yes", "1", "true"] else 0
        else:
            data["SeniorCitizen"] = int(senior_val)

        # Numeric bounds checking
        data["tenure"] = max(0, int(data.get("tenure", 0)))
        data["MonthlyCharges"] = max(0.0, float(data.get("MonthlyCharges", 0.0)))

        # TotalCharges auto-imputation if 0 or missing
        if "TotalCharges" not in data or data["TotalCharges"] is None or str(data["TotalCharges"]).strip() == "":
            data["TotalCharges"] = round(data["tenure"] * data["MonthlyCharges"], 2)
        else:
            try:
                tc = float(data["TotalCharges"])
                data["TotalCharges"] = max(0.0, tc)
            except (ValueError, TypeError):
                data["TotalCharges"] = round(data["tenure"] * data["MonthlyCharges"], 2)

        # Compound category normalization
        # If no internet service, internet add-ons are strictly 'No'
        if data.get("InternetService") == "No":
            for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                data[col] = "No"
        else:
            for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                if data.get(col) == "No internet service":
                    data[col] = "No"

        # If no phone service, MultipleLines is strictly 'No'
        if data.get("PhoneService") == "No":
            data["MultipleLines"] = "No"
        elif data.get("MultipleLines") == "No phone service":
            data["MultipleLines"] = "No"

        return data

    def transform_to_features(self, normalized_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform normalized dictionary into scaled 23-column aligned DataFrame.
        Uses exact one-hot feature encoding matching the training pipeline.
        """
        row = {
            "SeniorCitizen": int(normalized_data["SeniorCitizen"]),
            "tenure": float(normalized_data["tenure"]),
            "MonthlyCharges": float(normalized_data["MonthlyCharges"]),
            "TotalCharges": float(normalized_data["TotalCharges"]),
            "gender_Male": 1 if normalized_data.get("gender") == "Male" else 0,
            "Partner_Yes": 1 if normalized_data.get("Partner") == "Yes" else 0,
            "Dependents_Yes": 1 if normalized_data.get("Dependents") == "Yes" else 0,
            "PhoneService_Yes": 1 if normalized_data.get("PhoneService") == "Yes" else 0,
            "MultipleLines_Yes": 1 if normalized_data.get("MultipleLines") == "Yes" else 0,
            "InternetService_Fiber optic": 1 if normalized_data.get("InternetService") == "Fiber optic" else 0,
            "InternetService_No": 1 if normalized_data.get("InternetService") == "No" else 0,
            "OnlineSecurity_Yes": 1 if normalized_data.get("OnlineSecurity") == "Yes" else 0,
            "OnlineBackup_Yes": 1 if normalized_data.get("OnlineBackup") == "Yes" else 0,
            "DeviceProtection_Yes": 1 if normalized_data.get("DeviceProtection") == "Yes" else 0,
            "TechSupport_Yes": 1 if normalized_data.get("TechSupport") == "Yes" else 0,
            "StreamingTV_Yes": 1 if normalized_data.get("StreamingTV") == "Yes" else 0,
            "StreamingMovies_Yes": 1 if normalized_data.get("StreamingMovies") == "Yes" else 0,
            "Contract_One year": 1 if normalized_data.get("Contract") == "One year" else 0,
            "Contract_Two year": 1 if normalized_data.get("Contract") == "Two year" else 0,
            "PaperlessBilling_Yes": 1 if normalized_data.get("PaperlessBilling") == "Yes" else 0,
            "PaymentMethod_Credit card (automatic)": 1 if normalized_data.get("PaymentMethod") == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check": 1 if normalized_data.get("PaymentMethod") == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if normalized_data.get("PaymentMethod") == "Mailed check" else 0,
        }

        # Enforce exact column order from schema
        df = pd.DataFrame([row])[self.feature_columns]

        # Apply pre-fitted StandardScaler strictly to numeric features
        df[self.NUMERIC_COLS] = self.scaler.transform(df[self.NUMERIC_COLS])
        return df

    def diagnose_factors(self, data: Dict[str, Any], probability: float) -> Tuple[List[str], List[str]]:
        """Identify key drivers contributing to churn risk or retention stability."""
        risk_drivers = []
        retention_anchors = []

        # Contract Type (Top driver ~21.8% importance)
        contract = data.get("Contract", "Month-to-month")
        if contract == "Month-to-month":
            risk_drivers.append("Month-to-month contract terms (~42.7% baseline churn category)")
        elif contract in ["One year", "Two year"]:
            retention_anchors.append(f"{contract} contractual commitment (significant churn deterrent)")

        # Tenure (Top driver ~20.7% importance)
        tenure = data.get("tenure", 0)
        if tenure <= 6:
            risk_drivers.append(f"Early customer lifecycle stage ({tenure} month{'s' if tenure != 1 else ''} tenure)")
        elif tenure >= 24:
            retention_anchors.append(f"High account tenure & historical loyalty ({tenure} months)")

        # Internet Service (Top driver ~12.2% importance)
        internet = data.get("InternetService", "DSL")
        if internet == "Fiber optic":
            risk_drivers.append("Fiber optic subscription (elevated churn sensitivity: 41.9%)")
        elif internet == "No":
            retention_anchors.append("Low-complexity account (No internet friction)")

        # Payment Method (Top driver ~5.3% importance)
        payment = data.get("PaymentMethod", "")
        if payment == "Electronic check":
            risk_drivers.append("Electronic check manual billing (45.3% churn category)")
        elif "automatic" in payment.lower():
            retention_anchors.append("Automated recurring payment method enabled")

        # Support & Security Add-ons
        tech_support = data.get("TechSupport", "No")
        online_sec = data.get("OnlineSecurity", "No")
        if tech_support == "No" and internet != "No":
            risk_drivers.append("Absence of technical support coverage")
        elif tech_support == "Yes":
            retention_anchors.append("Active technical support service")

        if online_sec == "Yes":
            retention_anchors.append("Online security protection enabled")

        # Monthly Charges
        charges = data.get("MonthlyCharges", 0.0)
        if charges >= 85.0:
            risk_drivers.append(f"High monthly bill tier (${charges:.2f}/mo)")
        elif charges <= 35.0:
            retention_anchors.append(f"Budget-friendly billing tier (${charges:.2f}/mo)")

        return risk_drivers, retention_anchors

    def predict(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run end-to-end churn prediction on a customer profile.
        Returns a structured dictionary with probabilities, classification,
        risk band, styling metadata, and contextual diagnostic factors.
        """
        normalized_data = self.normalize_input(raw_input)
        feature_df = self.transform_to_features(normalized_data)

        # Model scoring
        prob = float(self.model.predict_proba(feature_df)[0, 1])
        pred = int(self.model.predict(feature_df)[0])
        percentage = round(prob * 100, 1)

        # Product-defined risk classification bands
        if prob < 0.35:
            risk_band = "Low Risk"
            risk_category = "Safe"
            risk_color = "#4F7959"       # Muted green
            risk_bg = "#EBF2ED"          # Pale green tint
            risk_border = "#C6DEC9"
            explanation = "Customer exhibits strong loyalty indicators and structural retention anchors."
        elif prob <= 0.65:
            risk_band = "Moderate Risk"
            risk_category = "Monitor"
            risk_color = "#B87B3E"       # Muted clay / warm amber
            risk_bg = "#FAF3E8"          # Warm amber tint
            risk_border = "#E6D1B5"
            explanation = "Customer exhibits balanced retention signals with identifiable friction points."
        else:
            risk_band = "High Risk"
            risk_category = "At Risk"
            risk_color = "#C85A46"       # Restrained terracotta
            risk_bg = "#F9ECE9"          # Pale terracotta tint
            risk_border = "#E8BCB3"
            explanation = "Customer displays key attrition drivers (e.g. month-to-month terms, high charges)."

        outcome_label = "Likely to Churn" if pred == 1 else "Likely to Stay"
        risk_drivers, retention_anchors = self.diagnose_factors(normalized_data, prob)
        retention_actions = self.suggest_retention_actions(normalized_data, prob)

        return {
            "probability": prob,
            "percentage": percentage,
            "prediction": pred,
            "outcome_label": outcome_label,
            "risk_band": risk_band,
            "risk_category": risk_category,
            "risk_color": risk_color,
            "risk_bg": risk_bg,
            "risk_border": risk_border,
            "explanation": explanation,
            "risk_drivers": risk_drivers,
            "retention_anchors": retention_anchors,
            "retention_actions": retention_actions,
            "normalized_input": normalized_data,
        }

    def suggest_retention_actions(self, data: Dict[str, Any], probability: float) -> List[str]:
        """
        Derive prioritized, rule-based operational retention recommendations
        tailored to the customer's specific contract, services, tenure, and risk score.
        """
        actions = []
        contract = str(data.get("Contract", "Month-to-month"))
        payment = str(data.get("PaymentMethod", ""))
        tenure = float(data.get("tenure", 0))
        monthly = float(data.get("MonthlyCharges", 0.0))
        internet = str(data.get("InternetService", "DSL"))
        tech_support = str(data.get("TechSupport", "No"))
        online_sec = str(data.get("OnlineSecurity", "No"))

        # 1. Contract Upgrade Incentive (Highest feature sensitivity)
        if contract == "Month-to-month":
            actions.append("Propose 12-month commitment upgrade with a 15% promotional billing discount")
        elif contract == "One year" and probability > 0.50:
            actions.append("Present early 2-year renewal agreement with locked-in rates and loyalty perks")

        # 2. Payment Friction Resolution
        if payment == "Electronic check":
            actions.append("Incentivize migration to automatic billing (Credit Card / Bank ACH) with a $5/mo statement credit")

        # 3. Monthly Spend & Bundle Optimization
        if monthly >= 80.0:
            actions.append(f"Perform bundle optimization audit to restructure services below ${monthly * 0.85:.0f}/mo")

        # 4. Service Support & Security Coverage
        if internet != "No":
            if tech_support == "No":
                actions.append("Include complimentary Premium Tech Support & Wi-Fi coverage diagnostic for 6 months")
            if online_sec == "No":
                actions.append("Activate complimentary cybersecurity & identity theft protection trial")

        # 5. Lifecycle-Specific Interventions
        if tenure <= 6:
            actions.append("Enroll in VIP White-Glove Onboarding with a dedicated 30-day customer success check-in")
        elif tenure >= 48 and probability >= 0.35:
            actions.append("Assign Senior Retention Account Executive; offer customer loyalty anniversary renewal gift")

        # 6. Low-risk / Stable Account Action
        if probability < 0.35 and not actions:
            actions.append("Account is stable. Enroll in customer advocacy referral program with mutual bill credits")

        if not actions:
            actions.append("Conduct periodic satisfaction survey and review service utilization")

        return actions[:3]

    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """
        Validate an uploaded or provided batch DataFrame against required schema.
        Returns:
            is_valid (bool): True if all required columns exist and data has rows.
            missing_cols (List[str]): List of missing required column names.
            messages (List[str]): Explanations or warnings for the user.
        """
        messages = []
        if df is None or len(df) == 0:
            return False, [], ["Uploaded dataset contains no records or is completely empty."]

        # Check column names (case-insensitive mapping if needed)
        existing_cols = list(df.columns)
        existing_lower_map = {str(c).strip().lower(): c for c in existing_cols}

        missing_cols = []
        for req in self.REQUIRED_INPUT_COLUMNS:
            if req not in existing_cols:
                if req.lower() in existing_lower_map:
                    df.rename(columns={existing_lower_map[req.lower()]: req}, inplace=True)
                else:
                    missing_cols.append(req)

        if missing_cols:
            messages.append(
                f"Missing {len(missing_cols)} required column(s): {', '.join(missing_cols)}. "
                "Please verify your CSV matches the Telco customer feature schema."
            )
            return False, missing_cols, messages

        # Check for missing values in required columns
        null_counts = df[self.REQUIRED_INPUT_COLUMNS].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            messages.append(
                f"Notice: Missing values detected in {len(cols_with_nulls)} column(s). "
                "Automatic default imputation will be applied during preprocessing."
            )

        return True, [], messages

    def predict_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a batch DataFrame of customer records, running vectorized feature
        transformation and model inference matching the exact production pipeline.
        Returns:
            results_df (pd.DataFrame): Enriched DataFrame with probabilities, risk bands, and actions.
            analytics (Dict[str, Any]): Summary KPIs, counts, distributions, and rankings.
        """
        d = df.copy()

        # Handle customerID
        if "customerID" in d.columns:
            customer_ids = d["customerID"].astype(str).tolist()
        else:
            customer_ids = [f"CUST-{i+1:04d}" for i in range(len(d))]

        # SeniorCitizen normalization
        if "SeniorCitizen" in d.columns:
            if d["SeniorCitizen"].dtype == object:
                senior = d["SeniorCitizen"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
            else:
                senior = d["SeniorCitizen"].fillna(0).astype(int)
        else:
            senior = pd.Series(0, index=d.index)

        # Tenure & MonthlyCharges
        tenure = pd.to_numeric(d.get("tenure", 0), errors="coerce").fillna(0).clip(lower=0).astype(float)
        monthly = pd.to_numeric(d.get("MonthlyCharges", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)

        # TotalCharges
        if "TotalCharges" in d.columns:
            total = pd.to_numeric(d["TotalCharges"], errors="coerce")
            total = total.fillna(tenure * monthly).clip(lower=0.0).astype(float)
        else:
            total = (tenure * monthly).astype(float)

        internet = d.get("InternetService", "DSL").fillna("DSL").astype(str)
        phone = d.get("PhoneService", "Yes").fillna("Yes").astype(str)

        def binary_series(col_name: str, match_val: str = "Yes", req_internet: bool = False, req_phone: bool = False) -> pd.Series:
            s = (d.get(col_name, "No").fillna("No").astype(str) == match_val).astype(int)
            if req_internet:
                s = s.where(internet != "No", 0)
            if req_phone:
                s = s.where(phone != "No", 0)
            return s

        feat_df = pd.DataFrame({
            "SeniorCitizen": senior,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "gender_Male": (d.get("gender", "Female").fillna("Female").astype(str) == "Male").astype(int),
            "Partner_Yes": (d.get("Partner", "No").fillna("No").astype(str) == "Yes").astype(int),
            "Dependents_Yes": (d.get("Dependents", "No").fillna("No").astype(str) == "Yes").astype(int),
            "PhoneService_Yes": (phone == "Yes").astype(int),
            "MultipleLines_Yes": binary_series("MultipleLines", "Yes", req_phone=True),
            "InternetService_Fiber optic": (internet == "Fiber optic").astype(int),
            "InternetService_No": (internet == "No").astype(int),
            "OnlineSecurity_Yes": binary_series("OnlineSecurity", "Yes", req_internet=True),
            "OnlineBackup_Yes": binary_series("OnlineBackup", "Yes", req_internet=True),
            "DeviceProtection_Yes": binary_series("DeviceProtection", "Yes", req_internet=True),
            "TechSupport_Yes": binary_series("TechSupport", "Yes", req_internet=True),
            "StreamingTV_Yes": binary_series("StreamingTV", "Yes", req_internet=True),
            "StreamingMovies_Yes": binary_series("StreamingMovies", "Yes", req_internet=True),
            "Contract_One year": (d.get("Contract", "").fillna("").astype(str) == "One year").astype(int),
            "Contract_Two year": (d.get("Contract", "").fillna("").astype(str) == "Two year").astype(int),
            "PaperlessBilling_Yes": (d.get("PaperlessBilling", "No").fillna("No").astype(str) == "Yes").astype(int),
            "PaymentMethod_Credit card (automatic)": (d.get("PaymentMethod", "").fillna("").astype(str) == "Credit card (automatic)").astype(int),
            "PaymentMethod_Electronic check": (d.get("PaymentMethod", "").fillna("").astype(str) == "Electronic check").astype(int),
            "PaymentMethod_Mailed check": (d.get("PaymentMethod", "").fillna("").astype(str) == "Mailed check").astype(int),
        })

        # Feature order alignment & Scaling
        feat_df = feat_df[self.feature_columns]
        feat_df[self.NUMERIC_COLS] = self.scaler.transform(feat_df[self.NUMERIC_COLS])

        # Inference
        probs = self.model.predict_proba(feat_df)[:, 1]
        preds = (probs >= 0.50).astype(int)

        # Risk band assignments
        risk_bands = []
        risk_categories = []
        for p in probs:
            if p < 0.35:
                risk_bands.append("Low Risk")
                risk_categories.append("Safe")
            elif p <= 0.65:
                risk_bands.append("Moderate Risk")
                risk_categories.append("Monitor")
            else:
                risk_bands.append("High Risk")
                risk_categories.append("At Risk")

        outcome_labels = ["Likely to Churn" if pr == 1 else "Likely to Stay" for pr in preds]

        # Generate top action for each record
        top_actions = []
        records = d.to_dict(orient="records")
        for i, rec in enumerate(records):
            acts = self.suggest_retention_actions(rec, probs[i])
            top_actions.append(acts[0] if acts else "Maintain standard customer service")

        # Build output DataFrame
        results_df = pd.DataFrame({
            "customerID": customer_ids,
            "Contract": d.get("Contract", "Month-to-month"),
            "tenure": tenure.astype(int),
            "MonthlyCharges": monthly.round(2),
            "TotalCharges": total.round(2),
            "InternetService": internet,
            "PaymentMethod": d.get("PaymentMethod", "Electronic check"),
            "churn_probability": np.round(probs, 4),
            "churn_percentage": np.round(probs * 100, 1),
            "prediction": preds,
            "outcome_label": outcome_labels,
            "risk_band": risk_bands,
            "risk_category": risk_categories,
            "recommended_action": top_actions,
        })

        # Calculate analytics
        n_total = len(results_df)
        n_high = int((results_df["risk_band"] == "High Risk").sum())
        n_mod = int((results_df["risk_band"] == "Moderate Risk").sum())
        n_low = int((results_df["risk_band"] == "Low Risk").sum())
        n_pred_churn = int((results_df["prediction"] == 1).sum())

        analytics = {
            "total_customers": n_total,
            "high_risk_count": n_high,
            "high_risk_pct": round((n_high / n_total) * 100, 1) if n_total > 0 else 0.0,
            "moderate_risk_count": n_mod,
            "moderate_risk_pct": round((n_mod / n_total) * 100, 1) if n_total > 0 else 0.0,
            "low_risk_count": n_low,
            "low_risk_pct": round((n_low / n_total) * 100, 1) if n_total > 0 else 0.0,
            "avg_probability": round(float(np.mean(probs)) * 100, 1) if n_total > 0 else 0.0,
            "predicted_churn_count": n_pred_churn,
            "predicted_churn_pct": round((n_pred_churn / n_total) * 100, 1) if n_total > 0 else 0.0,
        }

        return results_df, analytics
