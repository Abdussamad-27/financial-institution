# TASK DELIVERABLE: INTEGRATION CHECK (TB-L07)
Project: Loan Approval Prediction System


--------------------------------------------------------------------------------
1. DATA QUALITY & SANITIZATION HANDOFF
--------------------------------------------------------------------------------
• Dataset Source: Loan_Dataset_1000_rows-v2.csv (1000 samples, 11 attributes).
• Null Values: 0 missing/null entries detected; no imputation required.
• High Cardinality Feature Pruning: 'Applicant_Name' contains unique textual 
  identifiers that carry zero generalized credit risk signal. 
  -> Confirmed: Dropped in both KNN and Decision Tree pipelines before feature/target split.

--------------------------------------------------------------------------------
2. TARGET VARIABLE DISTRIBUTION & SPLIT PROTOCOL
--------------------------------------------------------------------------------
• Target Column: 'Loan_Approval_Status' (Binary: 0 = Not Approved, 1 = Approved).
• Class Imbalance:
  - Class 0: 760 rows (76.0%)
  - Class 1: 240 rows (24.0%)
• Stratification Rule:
  - train_test_split must enforce `stratify=y` with `test_size=0.20` and `random_state=42`.
  - Result: Both training (800 rows) and testing (200 rows) preserve the exact 76:24 ratio.
• Evaluation Impact:
  - Accuracy alone is not sufficient; evaluation must report Precision, Recall, and 
    F1-score for Class 1 (Approved).

--------------------------------------------------------------------------------
3. CATEGORICAL ENCODING & DIMENSIONALITY ALIGNMENT
--------------------------------------------------------------------------------
• Categorical Columns Identified: ['Gender', 'Loan_Purpose', 'Loan_Type'].
• Transformation Method: One-hot encoding using `pd.get_dummies(drop_first=True)` 
  to prevent the dummy variable trap (multicollinearity).
• Encoded Feature Space: Expands from 9 raw predictive attributes to 13 numerical columns.
• Inference Alignment Artifact:
  - To prevent 'ValueError: feature mismatch' when predicting on a single user in GUI, 
    `model_columns.pkl` has been established as the reference schema.

--------------------------------------------------------------------------------
4. MODEL-SPECIFIC PREPROCESSING RULES
--------------------------------------------------------------------------------
• Scale Discrepancy Observed in EDA:
  - Annual_Income & Loan_Amount_Requested span up to millions.
  - Credit_Score spans 300 to 900.
  - Loan_Term spans 12 to 240 months.

• KNN Model Requirements:
  - Distance Metric: Euclidean distance.
  - Action Required: Features must be standardized using `StandardScaler`.
  - Data Leakage Prevention: `scaler.fit_transform()` executed strictly on X_train; 
    X_test and new applicant inputs transformed using `scaler.transform()`.
  - Artifact Saved: `scaler.pkl`.

• Decision Tree Requirements:
  - Splitting Logic: Axis-aligned threshold splits (Credit_Score, Income, etc.).
  - Action Required: Invariant to monotonic scaling; unscaled X_train and X_test 
    passed directly without StandardScaler.

--------------------------------------------------------------------------------
5. CORRELATION & FEATURE IMPORTANCE CONFIRMATION
--------------------------------------------------------------------------------
• EDA Correlation Findings: Credit_Score, Loan_Amount_Requested, and Annual_Income 
  show the strongest statistical correlation with loan approval.
• Decision Tree Validation:
  - The trained Decision Tree's `feature_importances_` validates this finding:
    1. Credit_Score: ~30.4% importance
    2. Loan_Amount_Requested: ~17.7% importance
    3. Annual_Income: ~16.3% importance

--------------------------------------------------------------------------------
SIGN-OFF & READINESS:
- Data Pipeline Integrator: Approved
- ML Model Developer (KNN & DT): Approved
- Pipeline Status: Fully integrated without data leakage; ready for GUI and evaluation.