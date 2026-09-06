import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv("Loan_Dataset_1000_rows-v2.csv")

print("--- 1. DATA INTEGRITY CHECK ---")
print(f"Total rows and columns: {df.shape}")
print(f"Missing values count: {df.isnull().sum().sum()}")

# 2. Identifier Removal
df_clean = df.drop("Applicant_Name", axis=1)
print(f"Columns after dropping ID: {df_clean.shape[1]}")

# 3. Target Distribution Check
target_counts = df_clean["Loan_Approval_Status"].value_counts(normalize=True) * 100
print("\n--- 2. CLASS BALANCE CHECK ---")
print(f"Class 0 (Not Approved): {target_counts[0]:.2f}%")
print(f"Class 1 (Approved): {target_counts[1]:.2f}%")

# 4. Feature Encoding Check
X = df_clean.drop("Loan_Approval_Status", axis=1)
X_encoded = pd.get_dummies(X, drop_first=True)
print("\n--- 3. FEATURE SPACE ENCODING ---")
print(f"Feature count after one-hot encoding: {X_encoded.shape[1]}")
print(f"Encoded feature columns: {list(X_encoded.columns)}")

# 5. Numerical Range Check (Why KNN needs scaling)
print("\n--- 4. SCALE VARIANCE (JUSTIFICATION FOR KNN SCALER) ---")
print(f"Annual Income Range: [{X['Annual_Income'].min()}, {X['Annual_Income'].max()}]")
print(f"Credit Score Range: [{X['Credit_Score'].min()}, {X['Credit_Score'].max()}]")
print(f"Loan Term Range: [{X['Loan_Term'].min()}, {X['Loan_Term'].max()}]")