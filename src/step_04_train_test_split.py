import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = r"data/processed/model_ready_transactions_step3.csv"
OUT_DIR = r"data/processed"


os.makedirs(OUT_DIR, exist_ok=True)

print("STEP 04 — Train/Test Split & Leakage Prevention")
print("Loading:", INPUT_PATH)

df = pd.read_csv(INPUT_PATH)
print("Loaded shape:", df.shape)

# -----------------------------
# Target (Y)
# -----------------------------
if "isFraud" not in df.columns:
    raise ValueError("Target column 'isFraud' not found in dataset.")

y = df["isFraud"]

# -----------------------------
# Drop columns we must NOT use as features (X)
# -----------------------------
drop_cols = [
    "isFraud",          # target
    "nameOrig",         # identifier
    "nameDest",         # identifier
    "isFlaggedFraud",   # rule-based flag (leakage risk)
    "step"              # use only for time split; not as feature in baseline
]

# Drop only columns that exist (prevents errors if a column isn't present)
drop_cols_existing = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols_existing)

# If any non-numeric columns remain (e.g., raw 'type'), remove them safely
non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
if non_numeric:
    print("Dropping non-numeric columns from X:", non_numeric)
    X = X.drop(columns=non_numeric)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

# Safety check: make sure identifiers aren't in X
for bad in ["nameOrig", "nameDest", "isFraud", "isFlaggedFraud"]:
    if bad in X.columns:
        raise ValueError(f"Leakage column still in X: {bad}")

# -----------------------------
# 80/20 Stratified Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nSplit complete.")
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("Fraud rate train:", float(y_train.mean()))
print("Fraud rate test :", float(y_test.mean()))

# -----------------------------
# Save Step 04 outputs
# -----------------------------
X_train_path = os.path.join(OUT_DIR, "X_train_step4.csv")
X_test_path  = os.path.join(OUT_DIR, "X_test_step4.csv")
y_train_path = os.path.join(OUT_DIR, "y_train_step4.csv")
y_test_path  = os.path.join(OUT_DIR, "y_test_step4.csv")

X_train.to_csv(X_train_path, index=False)
X_test.to_csv(X_test_path, index=False)
y_train.to_csv(y_train_path, index=False)
y_test.to_csv(y_test_path, index=False)

print("\nSaved:")
print(" -", X_train_path)
print(" -", X_test_path)
print(" -", y_train_path)
print(" -", y_test_path)
