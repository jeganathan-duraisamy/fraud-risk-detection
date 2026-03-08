import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
DATA_DIR   = r"data/processed"
OUTPUT_DIR = r"outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# LOAD STEP 4 OUTPUTS
# -------------------------------------------------------
print("Loading train/test splits...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_step4.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test_step4.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_step4.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test_step4.csv")).squeeze()

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"Fraud rate train: {y_train.mean():.4f} | test: {y_test.mean():.4f}")

# -------------------------------------------------------
# SCALE FEATURES (needed for Logistic Regression baseline)
# -------------------------------------------------------
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# -------------------------------------------------------
# MODEL 1 — Logistic Regression (Baseline)
# -------------------------------------------------------
print("\nTraining Logistic Regression baseline...")
lr = LogisticRegression(
    class_weight="balanced",   # handles class imbalance
    max_iter=1000,
    random_state=42
)
lr.fit(X_train_sc, y_train)
lr_probs = lr.predict_proba(X_test_sc)[:, 1]
lr_preds = lr.predict(X_test_sc)

lr_ap    = average_precision_score(y_test, lr_probs)
lr_roc   = roc_auc_score(y_test, lr_probs)
print(f"  LR  | PR-AUC: {lr_ap:.4f} | ROC-AUC: {lr_roc:.4f}")

# -------------------------------------------------------
# MODEL 2 — Random Forest (Primary Model)
# -------------------------------------------------------
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",   # handles 0.13% fraud rate
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)

rf_ap  = average_precision_score(y_test, rf_probs)
rf_roc = roc_auc_score(y_test, rf_probs)
print(f"  RF  | PR-AUC: {rf_ap:.4f} | ROC-AUC: {rf_roc:.4f}")

# -------------------------------------------------------
# CLASSIFICATION REPORT
# -------------------------------------------------------
print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, rf_preds, target_names=["Legit", "Fraud"]))

# -------------------------------------------------------
# PORTFOLIO VISUAL — PR-AUC Curve + Confusion Matrix
# -------------------------------------------------------
fig = plt.figure(figsize=(16, 6), facecolor="#0f0f0f")
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# --- Colour palette (matches your dark Power BI theme) ---
GOLD    = "#f0c040"
BLUE    = "#4da6ff"
WHITE   = "#e0e0e0"
GREY    = "#555555"
BG      = "#0f0f0f"
PANEL   = "#1a1a1a"

# ── LEFT: PR-AUC Curve ──────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(PANEL)

# Random Forest curve
rf_prec, rf_rec, _ = precision_recall_curve(y_test, rf_probs)
ax1.plot(rf_rec, rf_prec, color=GOLD, lw=2.5,
         label=f"Random Forest  (AP = {rf_ap:.3f})")

# Logistic Regression curve
lr_prec, lr_rec, _ = precision_recall_curve(y_test, lr_probs)
ax1.plot(lr_rec, lr_prec, color=BLUE, lw=1.8, linestyle="--",
         label=f"Logistic Reg.  (AP = {lr_ap:.3f})")

# Baseline (random classifier)
baseline = y_test.mean()
ax1.axhline(baseline, color=GREY, linestyle=":", lw=1.2,
            label=f"Baseline (fraud rate = {baseline:.4f})")

ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.02])
ax1.set_xlabel("Recall",    color=WHITE, fontsize=11)
ax1.set_ylabel("Precision", color=WHITE, fontsize=11)
ax1.set_title("Precision-Recall Curve\n(Imbalanced Fraud Detection)",
              color=WHITE, fontsize=13, fontweight="bold", pad=12)
ax1.tick_params(colors=WHITE)
for spine in ax1.spines.values():
    spine.set_edgecolor(GREY)
ax1.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9, loc="upper right")

# Annotation box
ax1.annotate(
    f"PR-AUC = {rf_ap:.3f}\nROC-AUC = {rf_roc:.3f}",
    xy=(0.55, 0.75), xycoords="axes fraction",
    fontsize=10, color=GOLD,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#222222", edgecolor=GOLD, alpha=0.85)
)

# ── RIGHT: Confusion Matrix ──────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(PANEL)

cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Legit", "Fraud"])
disp.plot(ax=ax2, colorbar=False,
          cmap="YlOrBr",          # warm palette visible on dark bg
          values_format=",.0f")

ax2.set_title("Confusion Matrix — Random Forest\n(Threshold = 0.5)",
              color=WHITE, fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Predicted Label", color=WHITE, fontsize=11)
ax2.set_ylabel("True Label",      color=WHITE, fontsize=11)
ax2.tick_params(colors=WHITE)
for spine in ax2.spines.values():
    spine.set_edgecolor(GREY)
ax2.xaxis.label.set_color(WHITE)
ax2.yaxis.label.set_color(WHITE)
plt.setp(ax2.get_xticklabels(), color=WHITE)
plt.setp(ax2.get_yticklabels(), color=WHITE)

# Fix colourbar text if present
im = ax2.images[0] if ax2.images else None

# ── Super title ─────────────────────────────────────────
fig.suptitle(
    "Production-Oriented Fraud Risk Detection  |  6.3M+ Transactions",
    color=WHITE, fontsize=14, fontweight="bold", y=1.02
)

OUT_PATH = os.path.join(OUTPUT_DIR, "fraud_model_evaluation.png")
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\nPortfolio visual saved → {OUT_PATH}")

# -------------------------------------------------------
# FEATURE IMPORTANCE (top 15) — bonus portfolio snippet
# -------------------------------------------------------
fig2, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
ax.set_facecolor(PANEL)

feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns)
top15    = feat_imp.nlargest(15).sort_values()

bars = ax.barh(top15.index, top15.values, color=GOLD, edgecolor="none", height=0.6)
ax.set_title("Top 15 Feature Importances — Random Forest",
             color=WHITE, fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Importance", color=WHITE, fontsize=11)
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_edgecolor(GREY)
ax.set_facecolor(PANEL)
fig2.patch.set_facecolor(BG)

FI_PATH = os.path.join(OUTPUT_DIR, "fraud_feature_importance.png")
plt.savefig(FI_PATH, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Feature importance visual saved → {FI_PATH}")