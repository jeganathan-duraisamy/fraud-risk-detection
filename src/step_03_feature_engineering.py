import pandas as pd
import numpy as np
import os

DATA_PATH = r"data/raw/AIML Dataset.csv"

print("Feature Engineering (Fraud-specific)")
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

print("\n--- STEP 1: AMOUNT BEHAVIOUR FEATURES ---")
# 1.Feature Engineering - Amount behavior Features 

# 1.1) Amount relative to origin starting balance (protect divide-by-zero)
df["amount_to_oldbalanceOrg"] = np.where(
    df["oldbalanceOrg"] > 0,
    df["amount"] / df["oldbalanceOrg"],
    0
)

# 1.2) Flag: origin account becomes zero after transaction (common fraud pattern)
df["origin_zero_after_txn"] = (df["newbalanceOrig"] == 0).astype(int)

# 1.3) Flag: looks like a "drain" (amount roughly equals old balance)
# Using small tolerance so floating errors don't break the rule
tolerance = 1e-6
df["origin_drained_flag"] = (
    (df["oldbalanceOrg"] > 0) &
    (np.abs(df["oldbalanceOrg"] - df["amount"]) <= tolerance) &
    (df["newbalanceOrig"] == 0)
).astype(int)

# 1.4) Log transform for amount (helps with heavy skew)
df["log_amount"] = np.log1p(df["amount"])

print("Created features:",
      ["amount_to_oldbalanceOrg", "origin_zero_after_txn", "origin_drained_flag", "log_amount"])

print(df[["amount", "oldbalanceOrg", "newbalanceOrig",
          "amount_to_oldbalanceOrg", "origin_zero_after_txn",
          "origin_drained_flag", "log_amount"]].head(5))



#2.Time Behaviour Features

print("\n--- STEP 2: TIME BEHAVIOUR FEATURES ---")

# In this dataset, 'step' is a time index (often hours since start)
# We’ll extract useful “time patterns” from it.

# 2.1) Hour of day proxy (if step is hourly, this creates a repeating 0–23 cycle)
df["hour_of_day"] = (df["step"] % 24).astype(int)

# 2.2) Day index proxy (groups steps into days)
df["day_index"] = (df["step"] // 24).astype(int)

# 2.3) Weekend flag proxy (every 7 days)
df["is_weekend_proxy"] = (df["day_index"] % 7 >= 5).astype(int)

print("Created features:", ["hour_of_day", "day_index", "is_weekend_proxy"])
print(df[["step", "hour_of_day", "day_index", "is_weekend_proxy"]].head(5))



#3.Interaction Features

print("\n--- STEP 3: INTERACTION FEATURES ---")

# 1) Transaction type vs amount (some types are more fraud-prone at high amounts)
# We'll create "amount by type" using one-hot encoding + multiplication.

type_dummies = pd.get_dummies(df["type"], prefix="type")
for col in type_dummies.columns:
    df[col] = type_dummies[col]

    # Interaction: amount * type flag
    df[f"{col}_x_amount"] = df[col] * df["amount"]

created_interactions = [c for c in df.columns if c.endswith("_x_amount")]
print("Created interaction features (sample):", created_interactions[:5])
print(df[["type", "amount"] + created_interactions[:3]].head(5))


#4.Signal Amplification Features

print("\n--- STEP 4 : SIGNAL AMPLIFICATION FEATURES ---")

# 4.1) High amount flag (relative, not fixed threshold)
# We'll use percentile so it adapts to dataset scale.
high_amt_threshold = df["amount"].quantile(0.99)
df["high_amount_flag"] = (df["amount"] >= high_amt_threshold).astype(int)

# 4.2) Origin / Dest balance mismatch flags (from reconciliation logic idea)
# These capture “impossible” balance changes that are often fraud/data issues.

tolerance = 1e-6
df["origin_recon_mismatch_flag"] = (
    np.abs(df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]) > tolerance
).astype(int)

df["dest_recon_mismatch_flag"] = (
    np.abs(df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]) > tolerance
).astype(int)

# 4.3) Combine strong suspicious indicators into a single flag (simple amplification)
df["strong_suspicion_flag"] = (
    (df["origin_drained_flag"] == 1) |
    (df["high_amount_flag"] == 1) |
    (df["origin_recon_mismatch_flag"] == 1)
).astype(int)

print("High amount threshold (99th percentile):", high_amt_threshold)
print("Created features:", ["high_amount_flag", "origin_recon_mismatch_flag",
                          "dest_recon_mismatch_flag", "strong_suspicion_flag"])

print(df[["amount", "high_amount_flag", "origin_recon_mismatch_flag",
          "dest_recon_mismatch_flag", "strong_suspicion_flag"]].head(5))


print("\n--- VERIFICATION: ENGINEERED COLUMNS + FULL COLUMN LIST ---")

expected_cols = [
    # Step 1: Amount behaviour
    "amount_to_oldbalanceOrg", "origin_zero_after_txn", "origin_drained_flag", "log_amount",
    # Step 2: Time behaviour
    "hour_of_day", "day_index", "is_weekend_proxy",
    # Step 3: Interaction features (created dynamically, so we check pattern)
    # Step 4: Signal amplification
    "high_amount_flag", "origin_recon_mismatch_flag", "dest_recon_mismatch_flag", "strong_suspicion_flag"
]

missing = [c for c in expected_cols if c not in df.columns]
print("Missing engineered columns:", missing if missing else "None ✅")

# Confirm interaction columns exist (pattern-based check)
interaction_cols = [c for c in df.columns if c.startswith("type_") and c.endswith("_x_amount")]
print(f"Interaction columns created: {len(interaction_cols)}")
print("Sample interaction columns:", interaction_cols[:10])

print("\n--- B) FULL COLUMN LIST ---")
print(f"Total columns now: {df.shape[1]}\n")
for col in df.columns:
    print(col)

print("\n--- C) QUICK SANITY CHECK (ENGINEERED FEATURES SUMMARY) ---")
sanity_cols = expected_cols + interaction_cols[:10]  # include a sample of interactions
print(df[sanity_cols].describe(include="all").T)

from pathlib import Path

# Project root = one level above src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Create processed folder inside data
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Final output file
OUTPUT_PATH = OUTPUT_DIR / "model_ready_transactions_step3.csv"

# Save file
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved model-ready dataset to: {OUTPUT_PATH}")

