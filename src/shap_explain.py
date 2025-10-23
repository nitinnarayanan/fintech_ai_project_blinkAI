# src/shap_explain.py
import pandas as pd
import joblib
import shap
import numpy as np
import plotly.express as px
import os

MODEL_PATH = "../models/xgb_fraud_model.joblib"
DATA_PATH = "../data/transactions_fraud_detected.csv"
OUTPUT_DIR = "../data/shap_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load Model
# -------------------------
model = joblib.load(MODEL_PATH)

# -------------------------
# Load & Prepare Data
# -------------------------
df = pd.read_csv(DATA_PATH)
X = df[["Amount", "Country", "Purpose", "Channel", "AML_Risk_Score"]].copy()

# Encode categorical features on the fly
for col in ["Country", "Purpose", "Channel"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# -------------------------
# SHAP Explainability
# -------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# -------------------------
# Display Professional Plots
# -------------------------
top_features = 10  # top N features to display

# 1️⃣ Global Feature Importance (Bar Plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    max_display=top_features
)

# 2️⃣ SHAP Dot Summary (Per-row Contributions)
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X,
    plot_type="dot",
    max_display=top_features,
    color=plt.get_cmap("coolwarm")
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load Model & Encoders
# -------------------------
model = joblib.load(MODEL_PATH)

# shap_values_list shape: (n_samples, n_features, n_classes)
# Select positive class (index 1)
shap_values_pos = shap_values[:, :, 1]  # shape now (200, 5)

# Compute mean absolute SHAP per feature
mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)  # shape (5,)

# Create DataFrame
importance_df = pd.DataFrame({
    "feature": X.columns.tolist(),
    "mean_abs_shap": mean_abs_shap
})

# Save CSV
csv_path = os.path.join(OUTPUT_DIR, "global_shap_importance.csv")
importance_df.to_csv(csv_path, index=False)
print(f"✅ Saved global SHAP importance CSV → {csv_path}")
