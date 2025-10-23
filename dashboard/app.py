# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Paths
# -------------------------
DATA_PATH = r"N:\fintech_ai_project_blinkAI\data\transactions_fraud_detected.csv"
MODEL_PATH = r"N:\fintech_ai_project_blinkAI\models\rf_fraud_model.joblib"
SHAP_DIR = r"N:\fintech_ai_project_blinkAI\data\shap_outputs"

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

df = load_data()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.title("🔍 Filter Transactions")
currency = st.sidebar.multiselect("Currency", df["Currency"].unique())
channel = st.sidebar.multiselect("Channel", df["Channel"].unique())
risk_threshold = st.sidebar.slider("Risk Threshold", 0, 100, 70)

filtered = df.copy()
if currency:
    filtered = filtered[filtered["Currency"].isin(currency)]
if channel:
    filtered = filtered[filtered["Channel"].isin(channel)]

# -------------------------
# Overview KPIs
# -------------------------
st.title("💼 Fraud & AML Detection Dashboard")
st.markdown("Prototype inspired by **Blink AI Payments** (Bliink™ AML Pro + Fraud Pro).")

total_tx = len(filtered)
aml_flags = filtered["Flag_AML"].sum()
fraud_flags = filtered["Flag_Fraud"].sum()
combined_flags = ((filtered["Flag_AML"] == 1) | (filtered["Flag_Fraud"] == 1)).sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{total_tx:,}")
col2.metric("AML Flags", aml_flags)
col3.metric("Fraud Flags", fraud_flags)

st.metric("Combined Risk Alerts", combined_flags)

# -------------------------
# Risk Distribution Visualizations
# -------------------------
fig1 = px.histogram(
    filtered,
    x="AML_Risk_Score",
    color="Flag_AML",
    nbins=20,
    title="AML Risk Score Distribution",
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(
    filtered,
    x="Amount",
    y="Fraud_Score",
    color=filtered["Flag_Fraud"].map({0: "Normal", 1: "Anomalous"}),
    title="Fraud Risk vs Transaction Amount",
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# High-Risk Transactions Table
# -------------------------
st.subheader("🚨 High-Risk Transactions")
high_risk = filtered[
    (filtered["AML_Risk_Score"] >= risk_threshold) | (filtered["Flag_Fraud"] == 1)
].sort_values(by=["AML_Risk_Score", "Fraud_Score"], ascending=False)

st.dataframe(
    high_risk[
        [
            "TxId",
            "Sender",
            "Receiver",
            "Amount",
            "Currency",
            "Country",
            "Purpose",
            "Channel",
            "AML_Risk_Score",
            "Fraud_Score",
            "Flag_AML",
            "Flag_Fraud",
        ]
    ],
    use_container_width=True,
)

# -------------------------
# SHAP Explainability
# -------------------------
shap_exists = Path(MODEL_PATH).exists() and Path(SHAP_DIR).exists()
if shap_exists:
    st.subheader("🔎 SHAP Explainability")

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load global SHAP importance
    global_imp_path = os.path.join(SHAP_DIR, "global_shap_importance.csv")
    if Path(global_imp_path).exists():
        global_imp = pd.read_csv(global_imp_path)
        st.bar_chart(global_imp.set_index("feature")["mean_abs_shap"])

    # Encode features for per-row SHAP
    X_local = df[["Amount", "Country", "Purpose", "Channel", "AML_Risk_Score"]].copy()
    for col in ["Country", "Purpose", "Channel"]:
        le = LabelEncoder()
        X_local[col] = le.fit_transform(X_local[col].astype(str))

    # Load per-row SHAP values
    shap_values_path = os.path.join(SHAP_DIR, "shap_values.npy")
    if Path(shap_values_path).exists():
        shap_vals = np.load(shap_values_path)

        st.subheader("📝 Explain Selected Transaction")
        tx_selected = st.selectbox("Select TxId to explain", high_risk["TxId"].tolist())
        if tx_selected:
            idx = int(df[df["TxId"] == tx_selected].index[0])
            row_shap = shap_vals[idx]

            # Handle RandomForest shap_values shape (samples, features, classes)
            if row_shap.ndim == 2 and row_shap.shape[1] > 1:
                row_shap = row_shap[:, 1]  # positive class

            expl_df = pd.DataFrame({"feature": X_local.columns, "shap": row_shap})
            expl_df = expl_df.sort_values(by="shap", key=abs, ascending=False)
            st.table(expl_df)
else:
    st.info(
        "No explainability output found. Run `prepare_labels_and_train.py` and `shap_explain.py` to enable SHAP explainability."
    )

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    ---
    **Note:**  
    This prototype demonstrates how ISO 20022 transaction data can be analyzed in real-time using
    hybrid AI + rule-based detection for compliance and fraud prevention.
    """
)
