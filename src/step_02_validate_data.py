import pandas as pd

DATA_PATH = r"data/raw/AIML Dataset.csv"

print("STEP 2 — Data Validation & Quality Checks (read-only)")
print("Loading raw dataset...")

df = pd.read_csv(DATA_PATH)

print("Loaded.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- STEP 2.1: SCHEMA CHECK (COLUMNS + DTYPES) ---")

expected_cols = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"
]

missing_cols = [c for c in expected_cols if c not in df.columns]
extra_cols = [c for c in df.columns if c not in expected_cols]

print("Missing expected columns:", missing_cols if missing_cols else "None")
print("Unexpected extra columns:", extra_cols if extra_cols else "None")

print("\nDtypes (how Python interprets each column):")
print(df.dtypes)

print("\n--- STEP 2.2: MISSING VALUES & DATA QUALITY CHECKS ---")

# 1️⃣ Null counts
null_counts = df.isnull().sum()
null_percent = (null_counts / len(df)) * 100

print("\nNull values per column:")
print(null_counts)

print("\nPercentage missing per column:")
print(null_percent.round(4))

# 2️⃣ Balance logic validation - we have to do it for new baalnce as well 
print("\n--- Balance Logic Checks ---")

negative_amounts = (df["amount"] < 0).sum()
negative_origin_balance = (df["oldbalanceOrg"] < 0).sum()
negative_dest_balance = (df["oldbalanceDest"] < 0).sum()

print("Negative amounts:", negative_amounts)
print("Negative origin balances:", negative_origin_balance)
print("Negative destination balances:", negative_dest_balance)

# 3️⃣ Fraud class imbalance
print("\n--- Fraud Class Distribution ---")

fraud_counts = df["isFraud"].value_counts()
fraud_percent = df["isFraud"].value_counts(normalize=True) * 100

print("Fraud counts:")
print(fraud_counts)

print("\nFraud percentage:")
print(fraud_percent.round(4))

# 4. Create Reconciliation Difference Columns & count Mismatches 
print("\n--- STEP 2.3: BALANCE RECONCILIATION CHECK ---")

# Origin balance difference
df["origin_balance_diff"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]

# Destination balance difference
df["dest_balance_diff"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

# Absolute differences (easier to interpret)
df["origin_balance_abs_diff"] = df["origin_balance_diff"].abs()
df["dest_balance_abs_diff"] = df["dest_balance_diff"].abs()

print("Created reconciliation difference features.")

tolerance = 1e-6

origin_mismatch = (df["origin_balance_abs_diff"] > tolerance).sum()
dest_mismatch = (df["dest_balance_abs_diff"] > tolerance).sum()

print("Origin balance mismatches:", origin_mismatch)
print("Destination balance mismatches:", dest_mismatch)



