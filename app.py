"""
CHURN INTELLIGENCE — Customer Retention & Predictive Analytics
Premium Streamlit Web Application

Design Philosophy: Editorial, minimal, restrained, sophisticated.
Palette: Warm ivory (#F5F3EE), White (#FFFFFF), Charcoal (#292B2A),
         Muted grey (#777873), Muted teal (#547C78), Restrained terracotta (#C85A46),
         Muted green (#4F7959).
"""

import textwrap
import streamlit as st
import pandas as pd
import numpy as np
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
    max-width: 1140px;
    margin: 0 auto;
}

/* Typography Classes */
.editorial-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.85rem;
    font-weight: 500;
    line-height: 1.15;
    letter-spacing: -0.02em;
    color: #292B2A;
    margin-bottom: 0.5rem;
}

.editorial-sub {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #6E706B;
    max-width: 680px;
    margin-bottom: 1.75rem;
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
    font-size: 1.35rem;
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
    margin-bottom: 2rem;
    border-bottom: 1px solid #E5E2DA;
}

.wordmark {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: 0.04em;
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
    font-size: 3.6rem;
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
    padding: 0.5rem 0;
    border-bottom: 1px dashed #EDEAE1;
    font-size: 0.88rem;
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

/* Streamlit Widget Styling Adjustments */
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

/* Primary Action Button */
div.stButton > button[kind="primary"] {
    background-color: #547C78 !important;
    border: 1px solid #456763 !important;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 2rem !important;
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
# 2. State Management & Benchmark Presets
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


# -----------------------------------------------------------------------------
# 3. Top Navigation & Header
# -----------------------------------------------------------------------------
st.markdown(textwrap.dedent("""
<div class="nav-bar">
    <div>
        <div class="wordmark">CHURN INTELLIGENCE</div>
        <div class="wordmark-sub">Customer Retention & Predictive Analytics</div>
    </div>
    <div style="display: flex; gap: 0.5rem; align-items: center;">
        <span class="badge-tag">Production Gradient Boosting</span>
        <span class="badge-tag">Recall: 80.2%</span>
    </div>
</div>
"""), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. Hero Section
# -----------------------------------------------------------------------------
st.markdown(textwrap.dedent("""
<div>
    <div class="section-label">Executive Predictive Toolkit</div>
    <div class="editorial-title">Understand churn before it happens.</div>
    <div class="editorial-sub">
        Explore customer behaviour, estimate individual attrition risk, and diagnose retention friction points
        using our enterprise cost-sensitive machine learning pipeline.
    </div>
</div>
"""), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Benchmark Profiles Loader (Quick Demo Bar)
# -----------------------------------------------------------------------------
st.markdown(textwrap.dedent("""
<div class="surface-card-subtle">
    <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #547C78; margin-bottom: 0.25rem;">
        Demonstration Benchmarks & Verified Test Profiles
    </div>
    <div style="font-size: 0.85rem; color: #595B58;">
        Select a pre-validated test profile from our holdout audit, or customize attributes directly in the form:
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


# -----------------------------------------------------------------------------
# 6. Customer Profile Input Form
# -----------------------------------------------------------------------------
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

    # --- GROUP 3: Contract & Billing ---
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
    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        submit_btn = st.form_submit_button("Analyze Customer", type="primary", use_container_width=True)

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


# -----------------------------------------------------------------------------
# 7. Prediction Results Section
# -----------------------------------------------------------------------------
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
                <div style="color: #6A6D69; font-size: 0.95rem; max-width: 600px;">
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
                <span>Product Band Thresholds: Low (&lt;35%) • Moderate (35–65%) • High (&gt;65%)</span>
                <span>100% (Imminent Attrition)</span>
            </div>
            <div class="risk-meter-bg">
                <div class="risk-meter-fill" style="width: {min(100.0, prob_pct)}%; background-color: {risk_color};"></div>
            </div>
        </div>
    </div>
    """
    st.markdown(textwrap.dedent(card_html), unsafe_allow_html=True)

    # Secondary Diagnostic Breakdown
    col_diag1, col_diag2 = st.columns(2, gap="large")

    with col_diag1:
        st.markdown(textwrap.dedent("""
        <div class="surface-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.25rem; font-weight: 500; color: #292B2A;">
                    Key Risk Drivers
                </div>
                <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #C85A46; background: #F9ECE9; padding: 0.2rem 0.5rem; border-radius: 2px;">
                    Friction Factors
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
            <div style="font-size: 0.88rem; color: #777873; padding: 0.5rem 0;">
                No major friction drivers identified. Account demonstrates balanced, low-risk operational patterns.
            </div>
            """), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_diag2:
        st.markdown(textwrap.dedent("""
        <div class="surface-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.25rem; font-weight: 500; color: #292B2A;">
                    Retention Anchors
                </div>
                <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #4F7959; background: #EBF2ED; padding: 0.2rem 0.5rem; border-radius: 2px;">
                    Protective Assets
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
            <div style="font-size: 0.88rem; color: #777873; padding: 0.5rem 0;">
                Limited retention anchors found. Account is sensitive to competitive displacement or pricing shifts.
            </div>
            """), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. Model Governance & Operational Performance Section
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

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
        <strong>Methodological Disclaimer:</strong> These metrics reflect historical validation performance evaluated across an unseen 20% stratified holdout sample (N = 1,409) from the Telco Customer dataset. Predictions represent statistical risk probabilities based on historical customer patterns and are intended to inform human retention outreach, not guarantee individual customer behaviour.
    </div>
    """), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 9. Minimal Editorial Footer
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
