# src/prepare_labels_and_train_rf.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

DATA_PATH = r"N:\fintech_ai_project_blinkAI\data\transactions_fraud_detected.csv"
MODEL_PATH = r"N:\fintech_ai_project_blinkAI\models\rf_fraud_model.joblib"

def load_and_encode(path=DATA_PATH):
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Features for fraud detection
    features = ["Amount", "Country", "Purpose", "Channel", "AML_Risk_Score"]
    df_sub = df[features].copy()

    # Label encode categorical columns
    encoders = {}
    for col in ["Country", "Purpose", "Channel"]:
        le = LabelEncoder()
        df_sub[col] = le.fit_transform(df_sub[col].astype(str))
        encoders[col] = le

    return df, df_sub, encoders

def create_labels(df):
    # Use existing Flag_Fraud as proxy labels
    if "Flag_Fraud" in df.columns:
        labels = df["Flag_Fraud"].copy()
    else:
        labels = pd.Series(0, index=df.index)

    # Flip a small subset to simulate corrections / feedback
    rng = np.random.default_rng(42)
    flip_idx = rng.choice(df.index, size=int(0.01 * len(df)), replace=False)
    labels.loc[flip_idx] = 1 - labels.loc[flip_idx]

    return labels

def train_and_save(X, y, model_path=MODEL_PATH):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print("✅ Trained and saved Random Forest model to", model_path)

    return model

if __name__ == "__main__":
    df, X, encoders = load_and_encode()
    y = create_labels(df)
    model = train_and_save(X, y)
