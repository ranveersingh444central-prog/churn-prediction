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
            "normalized_input": normalized_data,
        }
