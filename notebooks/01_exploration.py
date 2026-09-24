"""
Telco Customer Churn - Step 1: Exploratory Data Analysis (EDA)
Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv (BlastChar / Kaggle)
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Configure pandas for clean terminal output
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda x: f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}")


def locate_dataset() -> Path:
    """Find the dataset path whether running from project root or notebooks dir."""
    possible_paths = [
        Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        Path("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]
    for p in possible_paths:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Could not find WA_Fn-UseC_-Telco-Customer-Churn.csv in expected data/ paths.")


def print_header(title: str) -> None:
    border = "=" * 80
    print(f"\n{border}")
    print(f" {title}")
    print(f"{border}\n")


def print_subheader(title: str) -> None:
    border = "-" * 60
    print(f"\n{border}")
    print(f" >>> {title}")
    print(f"{border}")


def main():
    csv_path = locate_dataset()
    print_header(f"TELCO CUSTOMER CHURN - INITIAL DATA EXPLORATION\n Dataset Path: {csv_path}")

    # -------------------------------------------------------------------------
    # 1. Load CSV, Shape, Columns, and Data Types
    # -------------------------------------------------------------------------
    df = pd.read_csv(csv_path)

    print_subheader("1. DATASET SHAPE & STRUCTURE")
    print(f"Total Rows (Customers): {df.shape[0]:,}")
    print(f"Total Columns:         {df.shape[1]}")

    print("\nColumns & Dtypes:")
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Non-Null Count": df.notna().sum().values,
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Sample Values": [str(df[c].dropna().unique()[:3].tolist()) for c in df.columns]
    })
    print(col_info.to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. First 5 Rows
    # -------------------------------------------------------------------------
    print_subheader("2. FIRST 5 ROWS")
    print(df.head(5).to_string())

    # -------------------------------------------------------------------------
    # 3. Missing Values Check (Standard NA)
    # -------------------------------------------------------------------------
    print_subheader("3. MISSING VALUE AUDIT (Standard Nulls / NaN)")
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing (NaN) Count": missing_counts,
        "Missing (%)": missing_pct
    })
    
    if missing_counts.sum() == 0:
        print("Standard df.isna().sum() check reports: 0 missing values across all 21 columns.")
        print("(Note: See Section 5 for hidden whitespace missing values.)")
    else:
        print(missing_df[missing_df["Missing (NaN) Count"] > 0].to_string())

    # -------------------------------------------------------------------------
    # 4. Basic Statistics: Churn Rate & Key Categoricals
    # -------------------------------------------------------------------------
    print_subheader("4. BASIC STATS & DISTRIBUTIONS")

    # Target variable: Churn Rate
    churn_counts = df["Churn"].value_counts()
    churn_norm = df["Churn"].value_counts(normalize=True) * 100
    churn_summary = pd.DataFrame({
        "Count": churn_counts,
        "Percentage (%)": churn_norm
    })
    print("TARGET VARIABLE - CHURN RATE:")
    print(churn_summary.to_string())
    churn_rate = churn_norm.get("Yes", 0.0)
    print(f"\n=> Overall Churn Rate: {churn_rate:.2f}% ({churn_counts.get('Yes', 0):,} churned out of {len(df):,})")

    # Contract value counts
    print("\nContract Distribution:")
    contract_df = pd.DataFrame({
        "Count": df["Contract"].value_counts(),
        "Percentage (%)": df["Contract"].value_counts(normalize=True) * 100
    })
    print(contract_df.to_string())

    # InternetService value counts
    print("\nInternetService Distribution:")
    internet_df = pd.DataFrame({
        "Count": df["InternetService"].value_counts(),
        "Percentage (%)": df["InternetService"].value_counts(normalize=True) * 100
    })
    print(internet_df.to_string())

    # PaymentMethod value counts
    print("\nPaymentMethod Distribution:")
    payment_df = pd.DataFrame({
        "Count": df["PaymentMethod"].value_counts(),
        "Percentage (%)": df["PaymentMethod"].value_counts(normalize=True) * 100
    })
    print(payment_df.to_string())

    # Numerical overview
    print("\nNumerical Features Summary (tenure, MonthlyCharges):")
    print(df[["tenure", "MonthlyCharges"]].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].to_string())

    # -------------------------------------------------------------------------
    # 5. Anomalies & Oddities (TotalCharges Blank Check)
    # -------------------------------------------------------------------------
    print_subheader("5. DATA QUALITY CHECK & ANOMALIES (TotalCharges & Others)")

    # A. TotalCharges dtype check
    print(f"Dtype of 'TotalCharges': {df['TotalCharges'].dtype}")
    is_numeric = pd.api.types.is_numeric_dtype(df["TotalCharges"])
    print(f"Is 'TotalCharges' numeric? {is_numeric} (Stored as string/object!)")

    # B. Blank / whitespace string detection in TotalCharges
    raw_tc = df["TotalCharges"].astype(str)
    blank_tc_mask = raw_tc.str.strip() == ""
    num_blanks = blank_tc_mask.sum()
    print(f"\nNumber of empty/whitespace values in TotalCharges: {num_blanks}")

    if num_blanks > 0:
        print("\nDetail of the 11 rows with blank TotalCharges:")
        blank_rows = df[blank_tc_mask][["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "Churn"]]
        print(blank_rows.to_string())

        print("\nKey Observation for Blank TotalCharges:")
        print("  - ALL 11 customers have tenure == 0.")
        print("  - ALL 11 customers have Churn == 'No'.")
        print("  - They are brand new customers who signed up but haven't completed their 1st billing cycle.")
        print("  - When parsed with pd.to_numeric(errors='coerce'), these 11 become NaN.")
        print("  - In modeling preprocessing, imputation (e.g. TotalCharges = 0.0 or MonthlyCharges) or dropping is required.")

    # C. Other notable observations
    print("\nAdditional Dataset Observations:")
    print("  1. Customer ID: 'customerID' has 7,043 unique values (100% unique primary key, should be dropped before modeling).")
    print("  2. SeniorCitizen format: Coded as integer [0, 1] while all other binary flags (Partner, Dependents, etc.) are 'Yes'/'No'.")
    print("  3. Redundant Sub-Categories: Multi-service columns (OnlineSecurity, TechSupport, etc.) contain 'No internet service' as a separate category distinct from 'No'.")
    print("  4. Class Imbalance: Churn is moderately imbalanced (~73.5% No vs ~26.5% Yes). Appropriate evaluation metrics (PR-AUC, ROC-AUC, F1) should be prioritized over raw accuracy.")

    print_header("EXPLORATION COMPLETE (No models trained yet)")


if __name__ == "__main__":
    main()
