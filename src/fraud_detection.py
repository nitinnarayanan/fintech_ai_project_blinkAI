import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

def load_and_prepare(path="../data/transactions_aml_screened.csv"):
    df = pd.read_csv(path)

    # Select useful features
    features = ["Amount", "Country", "Purpose", "Channel", "AML_Risk_Score"]
    df_sub = df[features].copy()

    # Encode categorical fields numerically
    for col in ["Country", "Purpose", "Channel"]:
        df_sub[col] = LabelEncoder().fit_transform(df_sub[col])

    return df, df_sub

def train_fraud_model(X):
    """Train an unsupervised Isolation Forest model."""
    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,   # ≈5 % anomalies
        random_state=42
    )
    model.fit(X)
    return model

def detect_anomalies(df, X, model):
    df["Fraud_Score"] = -model.decision_function(X) * 100   # higher = riskier
    df["Flag_Fraud"]  = (model.predict(X) == -1).astype(int)
    return df

if __name__ == "__main__":
    df, X = load_and_prepare()
    model = train_fraud_model(X)
    results = detect_anomalies(df, X, model)

    results.to_csv("../data/transactions_fraud_detected.csv", index=False)
    print("✅ Fraud detection complete.")
    print("Flagged fraudulent transactions:", results["Flag_Fraud"].sum())
    print(results.head(10))
