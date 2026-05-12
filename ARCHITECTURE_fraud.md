# Architecture: Fraud Risk Detection Pipeline

## Overview

A production-oriented, leakage-controlled ML pipeline for detecting financial fraud across 6.3M+ transactions. Built with modularity, explainability, and real-world deployment constraints in mind.

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────┐
│              RAW DATA                               │
│  AIML Dataset.csv — 6.3M+ transactions             │
│  0.13% fraud rate (severely imbalanced)            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         STEP 1 — DATA INGESTION                     │
│  step_01_ingest_raw_data.py                         │
│  - Load raw CSV                                     │
│  - Verify shape, dtypes, nulls                      │
│  - Confirm transaction volume                       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         STEP 2 — DATA VALIDATION                    │
│  step_02_validate_data.py                           │
│  - Schema checks                                    │
│  - Null analysis                                    │
│  - Balance reconciliation                           │
│  - Fraud rate verification                          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         STEP 3 — FEATURE ENGINEERING                │
│  step_03_feature_engineering.py                     │
│                                                     │
│  Amount Behaviour:                                  │
│  - amount_to_oldBalanceOrg                          │
│  - origin_zero_after_txn                            │
│  - origin_drained_flag                              │
│  - log_amount                                       │
│                                                     │
│  Time Behaviour:                                    │
│  - hour_of_day                                      │
│  - day_index                                        │
│  - is_weekend_proxy                                 │
│                                                     │
│  Interaction Features:                              │
│  - type_x_amount                                    │
│                                                     │
│  Signal Amplification:                              │
│  - high_amount_flag                                 │
│  - origin_recon_mismatch_flag                       │
│  - dest_recon_mismatch_flag                         │
│  - strong_suspicion_flag                            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         STEP 4 — LEAKAGE-CONTROLLED SPLIT           │
│  step_04_train_test_split.py                        │
│                                                     │
│  Excluded columns (would leak at prediction time):  │
│  - isFlaggedFraud (label leakage)                   │
│  - nameOrig, nameDest (no signal, privacy risk)     │
│  - oldBalanceOrg (used only for split logic)        │
│                                                     │
│  Stratified 80/20 split                             │
│  Preserves 0.13% fraud rate in both sets            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         STEP 5 — MODEL DEVELOPMENT                  │
│  step_05_model_development.py                       │
│                                                     │
│  Baseline: Logistic Regression                      │
│  - class_weight="balanced"                          │
│  - StandardScaler applied                           │
│  - PR-AUC: 0.980                                    │
│                                                     │
│  Final: Random Forest                               │
│  - 200 estimators, max depth 12                     │
│  - class_weight="balanced"                          │
│  - No scaling required                              │
│  - PR-AUC: 0.999                                    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              OUTPUTS                                │
│  fraud_model_evaluation.png                         │
│  fraud_feature_importance.png                       │
└─────────────────────────────────────────────────────┘
```

---

## Leakage Prevention Strategy

| Column | Why Excluded |
|--------|-------------|
| `isFlaggedFraud` | Direct label leak - system flag set after fraud confirmed |
| `nameOrig` | No predictive signal, privacy risk |
| `nameDest` | No predictive signal, privacy risk |
| `oldBalanceOrg` | Used only for time-based split logic, not available at real-time prediction |

**Stratified split** ensures class distribution is preserved — critical when fraud rate is 0.13%.

---

## Class Imbalance Strategy

| Approach | Decision |
|----------|----------|
| Oversampling (SMOTE) | Not used - risk of leakage if applied before split |
| Undersampling | Not used - loses legitimate transaction information |
| `class_weight="balanced"` | ✅ Used - penalises misclassification of minority class proportionally |
| PR-AUC as metric | ✅ Used - accuracy is misleading at 0.13% fraud rate |

---

## Model Selection Rationale

| Model | PR-AUC | Reason for inclusion |
|-------|--------|---------------------|
| Logistic Regression | 0.980 | Interpretable baseline, fast to train |
| Random Forest | 0.999 | Handles non-linear interactions, feature importance output |

Random Forest chosen as final model for:
- Higher PR-AUC (0.999 vs 0.980)
- Native feature importance scores for compliance explainability
- No scaling requirement - simpler production deployment
- Robust to outliers in amount distribution

---

## Feature Importance Design

Features were engineered in 4 deliberate categories to capture different fraud signatures:

| Category | Signal Captured |
|----------|----------------|
| Amount behaviour | Drain patterns, disproportionate transfers |
| Time behaviour | Off-hours transactions, day-of-week patterns |
| Interaction features | High-risk type+amount combinations |
| Signal amplification | Composite flags for strongest combined signals |

This design makes the model **defensible to compliance teams** - each feature has a clear business rationale.

---

## Production Considerations

- **Modular pipeline** - each step independently runnable and auditable
- **No data leakage** - explicitly excluded columns unavailable at prediction time
- **Stratified split** - fraud rate preserved across train/test
- **Explainable outputs** - feature importances exportable for regulatory review
- **Scalable** - designed for 6.3M+ transaction volume
