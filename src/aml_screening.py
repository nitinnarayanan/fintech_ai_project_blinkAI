import pandas as pd
import random

def load_data(tx_path="../data/transactions_parsed.csv",
              sanctions_path="../data/sanctioned_accounts.csv"):
    tx_df = pd.read_csv(tx_path)
    sanc_df = pd.read_csv(sanctions_path)
    sanctioned = set(sanc_df["Account"])
    return tx_df, sanctioned

def aml_screening(tx_df, sanctioned):
    """Simple rule-based AML screening + risk scoring."""
    results = tx_df.copy()

    def calc_risk(row):
        risk = 0
        # 1️⃣  Sanctions match
        if row["Sender"] in sanctioned or row["Receiver"] in sanctioned:
            risk += 70
        # 2️⃣  High-value transaction
        if row["Amount"] > 3000:
            risk += 15
        # 3️⃣  High-risk country (demo list)
        if row["Country"] in ["IR", "KP", "SY", "CU", "RU"]:
            risk += 10
        # 4️⃣  Suspicious purpose keywords
        if any(k in row["Purpose"].lower() for k in ["crypto", "refund", "loan"]):
            risk += 5
        return min(risk, 100)

    results["AML_Risk_Score"] = results.apply(calc_risk, axis=1)
    results["Flag_AML"] = results["AML_Risk_Score"].apply(lambda x: 1 if x >= 70 else 0)
    return results

if __name__ == "__main__":
    tx_df, sanctioned = load_data()
    screened = aml_screening(tx_df, sanctioned)
    screened.to_csv("../data/transactions_aml_screened.csv", index=False)

    print("✅ AML screening complete.")
    print("Flagged transactions:", screened["Flag_AML"].sum())
    print(screened.head(10))
