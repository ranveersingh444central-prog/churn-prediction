"""
Telco Customer Churn - Step 2: Data Cleaning and Feature Preparation
Saves cleaned dataset to data/cleaned_churn.csv
"""

from pathlib import Path
import pandas as pd


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
    # Default to root data directory if writing a new file
    return (Path(__file__).resolve().parent.parent / "data" / filename).resolve()


def main():
    print("=" * 70)
    print(" STEP 2: DATA CLEANING & FEATURE PREPARATION")
    print("=" * 70)

    # 1. Load the CSV
    raw_path = locate_dataset("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"\n[1] Loading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"    Initial Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 2. Fix TotalCharges
    print("\n[2] Fixing 'TotalCharges'...")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nan_count = df["TotalCharges"].isna().sum()
    all_zero_tenure = (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all()
    print(f"    - Blank rows coerced to NaN: {nan_count}")
    print(f"    - All blank rows have tenure == 0: {all_zero_tenure}")
    
    # Impute 0.0 for new customers
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    print(f"    - Filled {nan_count} NaNs with 0.0.")
    print(f"    - 'TotalCharges' dtype confirmed: {df['TotalCharges'].dtype}")

    # 3. Drop customerID
    print("\n[3] Dropping 'customerID'...")
    df = df.drop(columns=["customerID"])
    print(f"    - Remaining columns count: {len(df.columns)}")

    # 4. Simplify 'No internet service' / 'No phone service' categories
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines",
    ]
    print("\n[4] Simplifying compound categories into binary 'Yes' / 'No'...")
    for col in service_cols:
        before_vals = sorted(df[col].unique().tolist())
        df[col] = df[col].replace({
            "No internet service": "No",
            "No phone service": "No",
        })
        after_vals = sorted(df[col].unique().tolist())
        print(f"    - {col:<18}: {before_vals} -> {after_vals}")

    # 5. Convert Churn to binary (1/0)
    print("\n[5] Converting target 'Churn' to binary integer (1/0)...")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int64")
    print(f"    - Value counts for Churn:\n{df['Churn'].value_counts().to_string()}")

    # 6. Show dtypes and final null check
    print("\n[6] Validating Cleaned DataFrame:")
    print("    Column Data Types:")
    dtype_summary = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Null Count": df.isna().sum(),
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Sample Unique Values": [str(df[c].unique()[:4].tolist()) for c in df.columns]
    })
    print(dtype_summary.to_string())

    total_nulls = df.isna().sum().sum()
    print(f"\n    Total null values across entire dataset: {total_nulls}")
    assert total_nulls == 0, "Error: Unexpected null values found!"

    # 7. Save cleaned DataFrame to data/cleaned_churn.csv
    output_path = locate_dataset("cleaned_churn.csv")
    df.to_csv(output_path, index=False)
    print(f"\n[7] Cleaned dataset saved successfully to:\n    {output_path}")
    print(f"    Final Shape: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    print("=" * 70)


if __name__ == "__main__":
    main()
