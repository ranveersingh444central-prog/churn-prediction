"""
CHURN INTELLIGENCE — Customer Retention & Predictive Analytics
Premium Streamlit Web Application

Design Philosophy: Editorial, minimal, restrained, sophisticated (Metric / KALON inspired).
Palette: Warm ivory (#F5F3EE), White (#FFFFFF), Charcoal (#292B2A),
         Muted grey (#777873), Muted teal (#547C78), Restrained terracotta (#C85A46),
         Warm amber (#B87B3E), Muted green (#4F7959).
"""

import io
import textwrap
from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from src.predictor import ChurnPredictor

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Editorial Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CHURN INTELLIGENCE • Customer Retention Analytics",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Predictor with caching
@st.cache_resource(show_spinner=False)
def load_predictor():
    return ChurnPredictor()

predictor = load_predictor()

# Inject Custom Design System CSS
st.markdown(textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* Global Reset & Base */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #292B2A;
    background-color: #F5F3EE !important;
}

.stApp {
    background-color: #F5F3EE !important;
}

/* Hide default streamlit chrome */
#MainMenu, header, footer {
    visibility: hidden;
    height: 0;
}
div.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1180px;
    margin: 0 auto;
}

/* Typography Classes */
.editorial-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.75rem;
    font-weight: 500;
    line-height: 1.15;
    letter-spacing: -0.02em;
    color: #292B2A;
    margin-bottom: 0.5rem;
}

.editorial-sub {
    font-size: 1.02rem;
    line-height: 1.6;
    color: #6E706B;
    max-width: 720px;
    margin-bottom: 1.5rem;
}

.section-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #547C78;
    margin-bottom: 0.5rem;
}

.group-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.3rem;
    font-weight: 500;
    color: #292B2A;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #E5E2DA;
}

/* Cards & Surfaces */
.surface-card {
    background: #FFFFFF;
    border: 1px solid #E5E2DA;
    border-radius: 4px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}

.surface-card-subtle {
    background: #FAF9F5;
    border: 1px solid #EAE7DE;
    border-radius: 4px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.25rem;
}

/* Navigation bar */
.nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.25rem;
    margin-bottom: 1.75rem;
    border-bottom: 1px solid #E5E2DA;
}

.wordmark {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    color: #292B2A;
}

.wordmark-sub {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: #777873;
    text-transform: uppercase;
    margin-top: -2px;
}

.badge-tag {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    border-radius: 2px;
    background: #EAE7DF;
    color: #4A4D4A;
}

/* Executive Metric KPI Cards (Metric Style) */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E2DA;
    border-radius: 4px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 110px;
}

.kpi-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #777873;
    margin-bottom: 0.35rem;
}

.kpi-value {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.15rem;
    font-weight: 500;
    line-height: 1.1;
    color: #292B2A;
    margin-bottom: 0.25rem;
}

.kpi-badge {
    display: inline-flex;
    align-items: center;
    width: fit-content;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
    border-bottom: 1px solid #E5E2DA;
    margin-bottom: 1.75rem;
}

.stTabs [data-baseweb="tab"] {
    height: 3rem;
    white-space: pre-wrap;
    background-color: transparent !important;
    border: none !important;
    color: #777873 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0 0.5rem !important;
    border-bottom: 2px solid transparent !important;
}

.stTabs [aria-selected="true"] {
    color: #292B2A !important;
    border-bottom: 2px solid #547C78 !important;
    background-color: transparent !important;
}

/* Results Visuals */
.risk-meter-bg {
    width: 100%;
    height: 10px;
    background: #EBE8E1;
    border-radius: 5px;
    margin: 0.75rem 0 0.25rem 0;
    overflow: hidden;
}

.risk-meter-fill {
    height: 100%;
    border-radius: 5px;
}

.metric-number {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 3.4rem;
    font-weight: 500;
    line-height: 1;
    letter-spacing: -0.03em;
}

.risk-pill {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    border-radius: 3px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

/* List Items in Diagnostics */
.diagnostic-item {
    padding: 0.55rem 0;
    border-bottom: 1px dashed #EDEAE1;
    font-size: 0.86rem;
    line-height: 1.45;
    color: #383A39;
    display: flex;
    align-items: flex-start;
}
.diagnostic-item:last-child {
    border-bottom: none;
}
.bullet-risk {
    color: #C85A46;
    margin-right: 0.6rem;
    font-weight: bold;
    font-size: 0.8rem;
}
.bullet-anchor {
    color: #4F7959;
    margin-right: 0.6rem;
    font-weight: bold;
    font-size: 0.8rem;
}
.bullet-action {
    color: #547C78;
    margin-right: 0.6rem;
    font-weight: bold;
    font-size: 0.85rem;
}

/* Streamlit Form & Input Styling */
div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E5E2DA;
    border-radius: 4px;
    padding: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}

label[data-testid="stWidgetLabel"] p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #4A4D4A !important;
    margin-bottom: 0.25rem !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #FAF9F6 !important;
    border-color: #DCD8CE !important;
    border-radius: 3px !important;
    color: #292B2A !important;
    font-size: 0.9rem !important;
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover {
    border-color: #547C78 !important;
}

/* Primary Action Buttons */
div.stButton > button[kind="primary"] {
    background-color: #547C78 !important;
    border: 1px solid #456763 !important;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.65rem 1.75rem !important;
    border-radius: 3px !important;
    transition: background 0.15s ease;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #436461 !important;
}

div.stButton > button[kind="secondary"] {
    background-color: #FAF9F5 !important;
    border: 1px solid #DCD8CE !important;
    color: #4A4D4A !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 3px !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: #547C78 !important;
    color: #547C78 !important;
}
</style>
"""), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. State Management
# -----------------------------------------------------------------------------
DEFAULT_PROFILE = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 18,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 1437.30,
}

if "customer_input" not in st.session_state:
    st.session_state["customer_input"] = dict(DEFAULT_PROFILE)

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

if "batch_results_df" not in st.session_state:
    st.session_state["batch_results_df"] = None

if "batch_analytics" not in st.session_state:
    st.session_state["batch_analytics"] = None

if "batch_source_name" not in st.session_state:
    st.session_state["batch_source_name"] = None


def apply_preset(preset_key: str):
    """Callback to load one of the 3 validated benchmark presets."""
    preset_data = predictor.PRESET_PROFILES[preset_key]
    new_input = dict(preset_data)
    new_input["SeniorCitizen"] = "Yes" if new_input.get("SeniorCitizen") == 1 else "No"
    st.session_state["customer_input"] = new_input
    st.session_state["prediction_result"] = predictor.predict(new_input)


def reset_inputs():
    """Reset to default profile and clear results."""
    st.session_state["customer_input"] = dict(DEFAULT_PROFILE)
    st.session_state["prediction_result"] = None


def load_benchmark_sample():
    """Load verified Telco 100-customer sample for instant batch exploration."""
    sample_path = predictor.models_dir.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if sample_path.exists():
        raw_sample = pd.read_csv(sample_path, nrows=100)
        # Drop Churn column if present to simulate production unseen active records
        if "Churn" in raw_sample.columns:
            raw_sample = raw_sample.drop(columns=["Churn"])
        res_df, analytics = predictor.predict_batch(raw_sample)
        st.session_state["batch_results_df"] = res_df
        st.session_state["batch_analytics"] = analytics
        st.session_state["batch_source_name"] = "Benchmark Telco Sample (100 Active Records)"


def clear_batch_state():
    """Clear batch state."""
    st.session_state["batch_results_df"] = None
    st.session_state["batch_analytics"] = None
    st.session_state["batch_source_name"] = None


# -----------------------------------------------------------------------------
# 3. Top Navigation & Header
# -----------------------------------------------------------------------------
st.markdown(textwrap.dedent("""
<div class="nav-bar">
    <div>
        <div class="wordmark">CHURN INTELLIGENCE</div>
        <div class="wordmark-sub">Customer Retention & Predictive Analytics</div>
    </div>
    <div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">
        <span class="badge-tag">Production Gradient Boosting</span>
        <span class="badge-tag">Recall: 80.2%</span>
        <span class="badge-tag">ROC-AUC: 0.848</span>
    </div>
</div>
"""), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. Hero Section
# -----------------------------------------------------------------------------
st.markdown(textwrap.dedent("""
<div>
    <div class="section-label">Executive Predictive Suite</div>
    <div class="editorial-title">Understand churn before it happens.</div>
    <div class="editorial-sub">
        Diagnose retention friction points for individual subscribers or analyze entire customer cohorts
        using our enterprise cost-sensitive machine learning pipeline.
    </div>
</div>
"""), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Dual Prediction Modes Navigation Tabs
# -----------------------------------------------------------------------------
tab_indiv, tab_batch = st.tabs([
    "◎ Individual Customer Assessment",
    "▦ Batch Prediction & Portfolio Analytics"
])


# =============================================================================
# MODE A: INDIVIDUAL PREDICTION
# =============================================================================
with tab_indiv:
    # Benchmark Profiles Loader (Quick Demo Bar)
    st.markdown(textwrap.dedent("""
    <div class="surface-card-subtle">
        <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #547C78; margin-bottom: 0.25rem;">
            Demonstration Benchmarks & Verified Test Profiles
        </div>
        <div style="font-size: 0.85rem; color: #595B58;">
            Select a pre-validated test profile from our holdout audit, or customize attributes directly in the form below:
        </div>
    </div>
    """), unsafe_allow_html=True)

    col_p1, col_p2, col_p3, col_p4 = st.columns([1, 1, 1, 0.6])
    with col_p1:
        if st.button("◎ Low-Risk Benchmark", use_container_width=True, help="Loyal customer, 2-year contract, bank auto-pay"):
            apply_preset("Low-Risk Customer (Long-term, 2-Year Contract, Auto-Pay)")
    with col_p2:
        if st.button("◎ Moderate-Risk Benchmark", use_container_width=True, help="1-year contract, Fiber optic, credit card auto-pay"):
            apply_preset("Moderate-Risk Customer (1-Year Contract, Fiber Optic, Auto-Pay)")
    with col_p3:
        if st.button("◎ High-Risk Benchmark", use_container_width=True, help="New customer, month-to-month, electronic check"):
            apply_preset("High-Risk Customer (New, Month-to-Month, Fiber, Electronic Check)")
    with col_p4:
        if st.button("Reset Form", use_container_width=True, on_click=reset_inputs):
            pass

    # Customer Profile Input Form
    curr = st.session_state["customer_input"]

    with st.form("churn_prediction_form"):
        st.markdown('<div class="section-label">Account Specification</div>', unsafe_allow_html=True)
        col_group1, col_group2, col_group3 = st.columns(3, gap="large")

        # --- GROUP 1: Customer Demographics ---
        with col_group1:
            st.markdown('<div class="group-title">1. Customer Information</div>', unsafe_allow_html=True)
            
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male"],
                index=["Female", "Male"].index(curr.get("gender", "Female")),
                help="Customer biological gender"
            )
            
            senior_citizen = st.selectbox(
                "Senior Citizen",
                options=["No", "Yes"],
                index=0 if curr.get("SeniorCitizen") in ["No", 0] else 1,
                help="Indicates whether customer is 65 years or older"
            )
            
            partner = st.selectbox(
                "Partner",
                options=["No", "Yes"],
                index=["No", "Yes"].index(curr.get("Partner", "No")),
                help="Whether the customer has a domestic partner"
            )
            
            dependents = st.selectbox(
                "Dependents",
                options=["No", "Yes"],
                index=["No", "Yes"].index(curr.get("Dependents", "No")),
                help="Whether the customer has children or financial dependents"
            )
            
            tenure = st.number_input(
                "Account Tenure (Months)",
                min_value=0,
                max_value=72,
                value=int(curr.get("tenure", 12)),
                step=1,
                help="Number of months customer has been subscribed"
            )

        # --- GROUP 2: Services Subscribed ---
        with col_group2:
            st.markdown('<div class="group-title">2. Services & Technology</div>', unsafe_allow_html=True)
            
            phone_service = st.selectbox(
                "Phone Service",
                options=["Yes", "No"],
                index=["Yes", "No"].index(curr.get("PhoneService", "Yes")),
                help="Primary landline voice service"
            )
            
            multiple_lines_options = ["No", "Yes", "No phone service"] if phone_service == "No" else ["No", "Yes"]
            curr_mult = curr.get("MultipleLines", "No")
            mult_idx = multiple_lines_options.index(curr_mult) if curr_mult in multiple_lines_options else 0
            multiple_lines = st.selectbox(
                "Multiple Phone Lines",
                options=multiple_lines_options,
                index=mult_idx,
                help="Customer has multiple telephone lines"
            )
            
            internet_service = st.selectbox(
                "Internet Service Provider",
                options=["DSL", "Fiber optic", "No"],
                index=["DSL", "Fiber optic", "No"].index(curr.get("InternetService", "Fiber optic")),
                help="Broadband technology tier"
            )
            
            sec_options = ["No", "Yes"] if internet_service != "No" else ["No", "No internet service"]
            
            def safe_index(val, opts):
                return opts.index(val) if val in opts else 0
            
            online_security = st.selectbox(
                "Online Security",
                options=sec_options,
                index=safe_index(curr.get("OnlineSecurity", "No"), sec_options),
                help="Cybersecurity & malicious URL filtering add-on"
            )
            
            online_backup = st.selectbox(
                "Online Cloud Backup",
                options=sec_options,
                index=safe_index(curr.get("OnlineBackup", "No"), sec_options),
                help="Cloud storage and automated backup service"
            )
            
            device_protection = st.selectbox(
                "Device Protection Plan",
                options=sec_options,
                index=safe_index(curr.get("DeviceProtection", "No"), sec_options),
                help="Hardware warranty and damage replacement coverage"
            )
            
            tech_support = st.selectbox(
                "Premium Tech Support",
                options=sec_options,
                index=safe_index(curr.get("TechSupport", "No"), sec_options),
                help="Dedicated technical assistance and priority ticketing"
            )
            
            streaming_tv = st.selectbox(
                "Streaming TV",
                options=sec_options,
                index=safe_index(curr.get("StreamingTV", "No"), sec_options),
                help="Third-party streaming television provider package"
            )
            
            streaming_movies = st.selectbox(
                "Streaming Movies",
                options=sec_options,
                index=safe_index(curr.get("StreamingMovies", "No"), sec_options),
                help="Movie streaming digital subscription"
            )

        # --- GROUP 3: Contract & Financials ---
        with col_group3:
            st.markdown('<div class="group-title">3. Contract & Financials</div>', unsafe_allow_html=True)
            
            contract = st.selectbox(
                "Contract Term",
                options=["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(curr.get("Contract", "Month-to-month")),
                help="Primary churn predictor (~21.8% model importance)"
            )
            
            paperless_billing = st.selectbox(
                "Paperless Billing",
                options=["Yes", "No"],
                index=["Yes", "No"].index(curr.get("PaperlessBilling", "Yes")),
                help="Receives digital statements rather than physical mail"
            )
            
            payment_methods = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
            payment_method = st.selectbox(
                "Payment Method",
                options=payment_methods,
                index=payment_methods.index(curr.get("PaymentMethod", "Electronic check")),
                help="Electronic checks exhibit highest historical attrition"
            )
            
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=18.00,
                max_value=125.00,
                value=float(curr.get("MonthlyCharges", 70.00)),
                step=1.00,
                format="%.2f",
                help="Recurring monthly billing amount"
            )
            
            suggested_total = round(tenure * monthly_charges, 2)
            total_charges = st.number_input(
                "Total Accumulated Charges ($)",
                min_value=0.00,
                max_value=9000.00,
                value=float(curr.get("TotalCharges", suggested_total)),
                step=10.00,
                format="%.2f",
                help="Cumulative charges across customer lifetime."
            )

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([1.2, 4])
        with btn_col1:
            submit_btn = st.form_submit_button("Analyze Customer Profile", type="primary", use_container_width=True)

        if submit_btn:
            raw_payload = {
                "gender": gender,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
            st.session_state["customer_input"] = raw_payload
            st.session_state["prediction_result"] = predictor.predict(raw_payload)

    # Prediction Results Section
    res = st.session_state.get("prediction_result")

    if res is not None:
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Predictive Assessment</div>', unsafe_allow_html=True)

        prob_pct = res["percentage"]
        risk_color = res["risk_color"]
        risk_bg = res["risk_bg"]
        risk_border = res["risk_border"]
        risk_band = res["risk_band"]
        outcome_label = res["outcome_label"]

        # Main Result Card
        card_html = f"""
        <div class="surface-card" style="border-left: 6px solid {risk_color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <span class="risk-pill" style="background-color: {risk_bg}; color: {risk_color}; border: 1px solid {risk_border};">
                        {risk_band} • {res["risk_category"].upper()}
                    </span>
                    <div style="font-family: 'Newsreader', serif; font-size: 2.15rem; font-weight: 500; color: #292B2A; margin: 0.25rem 0;">
                        {outcome_label}
                    </div>
                    <div style="color: #6A6D69; font-size: 0.95rem; max-width: 620px;">
                        {res["explanation"]}
                    </div>
                </div>
                <div style="text-align: right; min-width: 180px;">
                    <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #777873;">
                        Estimated Churn Probability
                    </div>
                    <div class="metric-number" style="color: {risk_color};">
                        {prob_pct:.1f}%
                    </div>
                    <div style="font-size: 0.72rem; color: #8F918C;">
                        Decision Threshold: 50.0%
                    </div>
                </div>
            </div>
            <div style="margin-top: 1.5rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #777873; font-weight: 600;">
                    <span>0% (Secure Loyalty)</span>
                    <span>Product Bands: Low (&lt;35%) • Moderate (35–65%) • High (&gt;65%)</span>
                    <span>100% (Imminent Attrition)</span>
                </div>
                <div class="risk-meter-bg">
                    <div class="risk-meter-fill" style="width: {min(100.0, prob_pct)}%; background-color: {risk_color};"></div>
                </div>
            </div>
        </div>
        """
        st.markdown(textwrap.dedent(card_html), unsafe_allow_html=True)

        # Diagnostic Breakdown (Drivers, Anchors, and Tailored Retention Actions)
        col_diag1, col_diag2, col_diag3 = st.columns(3, gap="medium")

        with col_diag1:
            st.markdown(textwrap.dedent("""
            <div class="surface-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                    <div style="font-family: 'Newsreader', serif; font-size: 1.2rem; font-weight: 500; color: #292B2A;">
                        Key Risk Drivers
                    </div>
                    <span style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #C85A46; background: #F9ECE9; padding: 0.2rem 0.5rem; border-radius: 2px;">
                        Friction
                    </span>
                </div>
            """), unsafe_allow_html=True)

            if res["risk_drivers"]:
                for driver in res["risk_drivers"]:
                    st.markdown(textwrap.dedent(f"""
                    <div class="diagnostic-item">
                        <span class="bullet-risk">▲</span>
                        <span>{driver}</span>
                    </div>
                    """), unsafe_allow_html=True)
            else:
                st.markdown(textwrap.dedent("""
                <div style="font-size: 0.85rem; color: #777873; padding: 0.5rem 0;">
                    No major friction drivers identified. Account demonstrates balanced, low-risk operational patterns.
                </div>
                """), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_diag2:
            st.markdown(textwrap.dedent("""
            <div class="surface-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                    <div style="font-family: 'Newsreader', serif; font-size: 1.2rem; font-weight: 500; color: #292B2A;">
                        Retention Anchors
                    </div>
                    <span style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #4F7959; background: #EBF2ED; padding: 0.2rem 0.5rem; border-radius: 2px;">
                        Protective
                    </span>
                </div>
            """), unsafe_allow_html=True)

            if res["retention_anchors"]:
                for anchor in res["retention_anchors"]:
                    st.markdown(textwrap.dedent(f"""
                    <div class="diagnostic-item">
                        <span class="bullet-anchor">●</span>
                        <span>{anchor}</span>
                    </div>
                    """), unsafe_allow_html=True)
            else:
                st.markdown(textwrap.dedent("""
                <div style="font-size: 0.85rem; color: #777873; padding: 0.5rem 0;">
                    Limited retention anchors found. Account is sensitive to competitive displacement or pricing shifts.
                </div>
                """), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_diag3:
            st.markdown(textwrap.dedent("""
            <div class="surface-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                    <div style="font-family: 'Newsreader', serif; font-size: 1.2rem; font-weight: 500; color: #292B2A;">
                        Recommended Action Plan
                    </div>
                    <span style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #547C78; background: #EBF2F1; padding: 0.2rem 0.5rem; border-radius: 2px;">
                        Prescriptive
                    </span>
                </div>
            """), unsafe_allow_html=True)

            if res.get("retention_actions"):
                for action in res["retention_actions"]:
                    st.markdown(textwrap.dedent(f"""
                    <div class="diagnostic-item">
                        <span class="bullet-action">→</span>
                        <span>{action}</span>
                    </div>
                    """), unsafe_allow_html=True)
            else:
                st.markdown(textwrap.dedent("""
                <div style="font-size: 0.85rem; color: #777873; padding: 0.5rem 0;">
                    Account status stable. Review periodic service satisfaction surveys.
                </div>
                """), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MODE B: BATCH PREDICTION & PORTFOLIO ANALYTICS DASHBOARD
# =============================================================================
with tab_batch:
    st.markdown('<div class="section-label">Dataset Ingestion & Batch Pipeline</div>', unsafe_allow_html=True)
    
    # Ingestion Controls Surface
    col_up1, col_up2 = st.columns([2.5, 1], gap="medium")
    
    with col_up1:
        uploaded_csv = st.file_uploader(
            "Upload Telco Customer Dataset (CSV)",
            type=["csv"],
            help="Accepts CSV with customer demographics, services, and billing columns."
        )

    with col_up2:
        st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
        if st.button("◎ Load Benchmark Sample (100 Customers)", use_container_width=True, type="secondary"):
            load_benchmark_sample()
            st.rerun()

    # Provide blank schema template download
    template_cols = predictor.REQUIRED_INPUT_COLUMNS + ["customerID", "TotalCharges"]
    sample_template_df = pd.DataFrame([{
        "customerID": "DEMO-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.00,
        "TotalCharges": 900.00,
    }])
    template_csv = sample_template_df.to_csv(index=False).encode("utf-8")

    sub_c1, sub_c2, sub_c3 = st.columns([1.5, 1, 2])
    with sub_c1:
        st.download_button(
            "↓ Download CSV Schema Template",
            data=template_csv,
            file_name="telco_customer_schema_template.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download a minimal 1-row CSV template with correct column names."
        )
    with sub_c2:
        if st.session_state["batch_results_df"] is not None:
            if st.button("Clear Dataset", use_container_width=True):
                clear_batch_state()
                st.rerun()

    # Process uploaded CSV if present
    if uploaded_csv is not None:
        try:
            raw_upload_df = pd.read_csv(uploaded_csv)
            is_valid, missing_cols, val_msgs = predictor.validate_dataset(raw_upload_df)

            if not is_valid:
                st.error("Validation Error: Uploaded file does not conform to the expected Telco customer schema.")
                for msg in val_msgs:
                    st.markdown(f"- {msg}")
            else:
                res_df, analytics = predictor.predict_batch(raw_upload_df)
                st.session_state["batch_results_df"] = res_df
                st.session_state["batch_analytics"] = analytics
                st.session_state["batch_source_name"] = uploaded_csv.name
        except Exception as e:
            st.error(f"Error processing uploaded CSV: {str(e)}")

    # Check whether batch data is available
    batch_df = st.session_state.get("batch_results_df")
    analytics = st.session_state.get("batch_analytics")
    source_name = st.session_state.get("batch_source_name")

    if batch_df is None or analytics is None:
        # Graceful Empty / Unloaded State
        st.markdown(textwrap.dedent("""
        <div class="surface-card" style="margin-top: 1.5rem; text-align: center; padding: 3.5rem 2rem;">
            <div style="font-family: 'Newsreader', Georgia, serif; font-size: 1.85rem; font-weight: 500; color: #292B2A; margin-bottom: 0.5rem;">
                No customer dataset loaded yet.
            </div>
            <div style="font-size: 0.95rem; color: #6E706B; max-width: 580px; margin: 0 auto 1.5rem auto; line-height: 1.6;">
                Upload a CSV containing active customer records, or click <strong>Load Benchmark Sample</strong>
                above to instantly view interactive portfolio visualizations, risk distributions, and prioritized retention queues.
            </div>
            <div style="display: inline-flex; gap: 0.75rem; justify-content: center; font-size: 0.78rem; color: #777873;">
                <span>✓ Automatic 18-Feature Normalization</span>
                <span>•</span>
                <span>✓ Exact StandardScaler Alignment</span>
                <span>•</span>
                <span>✓ High-Performance Inference</span>
            </div>
        </div>
        """), unsafe_allow_html=True)
    else:
        # Loaded State: Display Dashboard & Analytics
        st.markdown(textwrap.dedent(f"""
        <div style="margin-top: 1.5rem; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="section-label">Cohort Intelligence</span>
                <div style="font-family: 'Newsreader', Georgia, serif; font-size: 1.75rem; font-weight: 500; color: #292B2A;">
                    {source_name or 'Active Customer Cohort'}
                </div>
            </div>
            <div>
                <span class="badge-tag">Total Records: {analytics['total_customers']}</span>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # 1. Executive KPI Cards Row (Metric Inspired)
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        with kpi1:
            st.markdown(textwrap.dedent(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Total Analysed</div>
                    <div class="kpi-value">{analytics['total_customers']}</div>
                </div>
                <div class="kpi-badge" style="background: #EAE7DF; color: #4A4D4A;">
                    Cohort Base
                </div>
            </div>
            """), unsafe_allow_html=True)

        with kpi2:
            st.markdown(textwrap.dedent(f"""
            <div class="kpi-card" style="border-top: 3px solid #C85A46;">
                <div>
                    <div class="kpi-label">High Risk (At Risk)</div>
                    <div class="kpi-value" style="color: #C85A46;">{analytics['high_risk_count']}</div>
                </div>
                <div class="kpi-badge" style="background: #F9ECE9; color: #C85A46;">
                    {analytics['high_risk_pct']:.1f}% of cohort
                </div>
            </div>
            """), unsafe_allow_html=True)

        with kpi3:
            st.markdown(textwrap.dedent(f"""
            <div class="kpi-card" style="border-top: 3px solid #B87B3E;">
                <div>
                    <div class="kpi-label">Moderate Risk</div>
                    <div class="kpi-value" style="color: #B87B3E;">{analytics['moderate_risk_count']}</div>
                </div>
                <div class="kpi-badge" style="background: #FAF3E8; color: #B87B3E;">
                    {analytics['moderate_risk_pct']:.1f}% of cohort
                </div>
            </div>
            """), unsafe_allow_html=True)

        with kpi4:
            st.markdown(textwrap.dedent(f"""
            <div class="kpi-card" style="border-top: 3px solid #4F7959;">
                <div>
                    <div class="kpi-label">Low Risk (Safe)</div>
                    <div class="kpi-value" style="color: #4F7959;">{analytics['low_risk_count']}</div>
                </div>
                <div class="kpi-badge" style="background: #EBF2ED; color: #4F7959;">
                    {analytics['low_risk_pct']:.1f}% of cohort
                </div>
            </div>
            """), unsafe_allow_html=True)

        with kpi5:
            st.markdown(textwrap.dedent(f"""
            <div class="kpi-card" style="border-top: 3px solid #547C78;">
                <div>
                    <div class="kpi-label">Avg Churn Probability</div>
                    <div class="kpi-value" style="color: #547C78;">{analytics['avg_probability']:.1f}%</div>
                </div>
                <div class="kpi-badge" style="background: #EBF2F1; color: #547C78;">
                    Threshold: 50.0%
                </div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # 2. Altair Visualizations Row
        chart_col1, chart_col2, chart_col3 = st.columns([1, 1.2, 1], gap="medium")

        # Chart 1: Donut Chart for Risk Bands
        with chart_col1:
            st.markdown(textwrap.dedent("""
            <div class="surface-card" style="padding: 1.25rem;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.15rem; font-weight: 500; color: #292B2A; margin-bottom: 0.5rem;">
                    Risk Band Distribution
                </div>
            """), unsafe_allow_html=True)

            risk_summary_df = pd.DataFrame([
                {"Category": "Low Risk (<35%)", "Count": analytics["low_risk_count"], "Order": 1},
                {"Category": "Moderate Risk (35-65%)", "Count": analytics["moderate_risk_count"], "Order": 2},
                {"Category": "High Risk (>65%)", "Count": analytics["high_risk_count"], "Order": 3},
            ])

            donut_chart = alt.Chart(risk_summary_df).mark_arc(innerRadius=48, stroke="#FFFFFF", strokeWidth=2).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(
                    field="Category",
                    type="nominal",
                    scale=alt.Scale(
                        domain=["Low Risk (<35%)", "Moderate Risk (35-65%)", "High Risk (>65%)"],
                        range=["#4F7959", "#B87B3E", "#C85A46"]
                    ),
                    legend=alt.Legend(title=None, orient="bottom", labelFont="Plus Jakarta Sans", labelFontSize=11)
                ),
                tooltip=[
                    alt.Tooltip("Category:N", title="Risk Band"),
                    alt.Tooltip("Count:Q", title="Subscribers"),
                ]
            ).properties(height=230)

            st.altair_chart(donut_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Chart 2: Probability Distribution Histogram
        with chart_col2:
            st.markdown(textwrap.dedent("""
            <div class="surface-card" style="padding: 1.25rem;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.15rem; font-weight: 500; color: #292B2A; margin-bottom: 0.5rem;">
                    Churn Probability Distribution
                </div>
            """), unsafe_allow_html=True)

            hist_chart = alt.Chart(batch_df).mark_bar(color="#547C78", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X(
                    "churn_probability:Q",
                    bin=alt.Bin(step=0.1),
                    title="Estimated Churn Probability",
                    axis=alt.Axis(format="%", labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans")
                ),
                y=alt.Y(
                    "count():Q",
                    title="Subscribers",
                    axis=alt.Axis(labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans")
                ),
                tooltip=[
                    alt.Tooltip("count():Q", title="Count"),
                ]
            ).properties(height=230)

            # Threshold rule line at 0.50
            threshold_df = pd.DataFrame([{"threshold": 0.50}])
            rule = alt.Chart(threshold_df).mark_rule(color="#C85A46", strokeDash=[4, 4], strokeWidth=2).encode(
                x="threshold:Q"
            )

            st.altair_chart(hist_chart + rule, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Chart 3: Churn Risk by Contract Type
        with chart_col3:
            st.markdown(textwrap.dedent("""
            <div class="surface-card" style="padding: 1.25rem;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.15rem; font-weight: 500; color: #292B2A; margin-bottom: 0.5rem;">
                    Risk by Contract Term
                </div>
            """), unsafe_allow_html=True)

            contract_chart = alt.Chart(batch_df).mark_bar().encode(
                x=alt.X("Contract:N", title="Contract Term", axis=alt.Axis(labelAngle=0, labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans")),
                y=alt.Y("count():Q", title="Subscribers", stack="normalize", axis=alt.Axis(format="%", labelFont="Plus Jakarta Sans")),
                color=alt.Color(
                    "risk_band:N",
                    scale=alt.Scale(
                        domain=["Low Risk", "Moderate Risk", "High Risk"],
                        range=["#4F7959", "#B87B3E", "#C85A46"]
                    ),
                    legend=alt.Legend(title=None, orient="bottom", labelFont="Plus Jakarta Sans", labelFontSize=11)
                ),
                tooltip=["Contract:N", "risk_band:N", "count():Q"]
            ).properties(height=230)

            st.altair_chart(contract_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        # 3. High-Risk Customer Priority Queue & Filterable Table
        st.markdown('<div class="section-label">Retention Operations Queue</div>', unsafe_allow_html=True)

        col_filter1, col_filter2, col_filter3 = st.columns([1.5, 1.5, 1])

        with col_filter1:
            risk_filter = st.selectbox(
                "Filter by Risk Classification",
                options=["All Records", "High Risk (>65%)", "Moderate Risk (35–65%)", "Low Risk (<35%)"],
                index=0
            )

        with col_filter2:
            search_query = st.text_input("Search Customer ID", placeholder="Filter by ID...")

        with col_filter3:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            # Enriched CSV Download Button
            csv_export = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "↓ Export Predictions (CSV)",
                data=csv_export,
                file_name="churn_intelligence_batch_predictions.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        # Filter the dataset
        filtered_df = batch_df.copy()
        if "High Risk" in risk_filter:
            filtered_df = filtered_df[filtered_df["risk_band"] == "High Risk"]
        elif "Moderate Risk" in risk_filter:
            filtered_df = filtered_df[filtered_df["risk_band"] == "Moderate Risk"]
        elif "Low Risk" in risk_filter:
            filtered_df = filtered_df[filtered_df["risk_band"] == "Low Risk"]

        if search_query.strip():
            filtered_df = filtered_df[filtered_df["customerID"].str.contains(search_query.strip(), case=False, na=False)]

        # Sort with highest risk first
        filtered_df = filtered_df.sort_values(by="churn_probability", ascending=False)

        # Display Interactive Data Table
        display_columns = [
            "customerID", "Contract", "tenure", "MonthlyCharges",
            "churn_probability", "risk_band", "outcome_label", "recommended_action"
        ]

        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "customerID": st.column_config.TextColumn("Customer ID", width="small"),
                "Contract": st.column_config.TextColumn("Contract Term", width="small"),
                "tenure": st.column_config.NumberColumn("Tenure", format="%d mos", width="small"),
                "MonthlyCharges": st.column_config.NumberColumn("Monthly Bill", format="$%.2f", width="small"),
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn Probability",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=1.0,
                    width="medium"
                ),
                "risk_band": st.column_config.TextColumn("Risk Tier", width="small"),
                "outcome_label": st.column_config.TextColumn("Assessment", width="small"),
                "recommended_action": st.column_config.TextColumn("Prescribed Retention Action", width="large"),
            }
        )

        st.markdown(textwrap.dedent(f"""
        <div style="font-size: 0.78rem; color: #777873; margin-top: 0.5rem; display: flex; justify-content: space-between;">
            <span>Showing {len(filtered_df)} of {len(batch_df)} subscribers</span>
            <span>Sorted by descending churn probability (Highest risk prioritised)</span>
        </div>
        """), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. Model Governance & Operational Performance Section
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)

with st.expander("Model Specifications & Historical Evaluation Metrics", expanded=False):
    st.markdown(textwrap.dedent("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-family: 'Newsreader', serif; font-size: 1.25rem; font-weight: 500; color: #292B2A; margin-bottom: 0.5rem;">
            Cost-Sensitive Gradient Boosting Architecture
        </div>
        <p style="font-size: 0.88rem; line-height: 1.6; color: #595B58; margin-bottom: 1.25rem;">
            Because acquiring a replacement telecom subscriber is 5x to 7x more expensive than preserving an existing account,
            this classifier was engineered with <strong>cost-sensitive sample weights</strong> and hyperparameter-tuned
            via 5-Fold Stratified Cross-Validation to maximize <strong>Recall on Churners</strong>.
        </p>
    </div>
    """), unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(label="Churn Recall", value="80.21%", help="Catches 300 of 374 actual churners in unseen test holdout")
    with m2:
        st.metric(label="ROC-AUC Score", value="0.8476", help="High discriminative separation between churners and retainers")
    with m3:
        st.metric(label="Test Accuracy", value="75.02%", help="Overall classification accuracy on imbalanced holdout test set")
    with m4:
        st.metric(label="Precision", value="51.90%", help="Precision on minority churn class")
    with m5:
        st.metric(label="F1-Score", value="0.6303", help="Harmonic mean of precision and recall")

    st.markdown(textwrap.dedent("""
    <div style="margin-top: 1.25rem; padding: 0.85rem 1rem; background: #FAF9F5; border: 1px solid #EAE7DE; border-radius: 3px; font-size: 0.8rem; color: #777873; line-height: 1.5;">
        <strong>Methodological Disclaimer:</strong> These metrics reflect historical validation performance evaluated across an unseen 20% stratified holdout sample (N = 1,409) from the Telco Customer dataset. Predictions represent statistical risk probabilities based on historical customer patterns and are intended to inform human retention outreach, not guarantee individual customer behaviour. Prescriptive retention actions represent heuristic business guidelines derived from feature sensitivities.
    </div>
    """), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Minimal Editorial Footer
# -----------------------------------------------------------------------------
st.markdown(textwrap.dedent("""
<div style="margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid #E5E2DA; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; font-size: 0.75rem; color: #8F918C;">
    <div>
        <strong>CHURN INTELLIGENCE</strong> • Built for Executive Customer Retention Strategy
    </div>
    <div>
        Trained on Telco Customer Benchmark • Pipeline Artifacts Verified
    </div>
</div>
"""), unsafe_allow_html=True)
