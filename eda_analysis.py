import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset present in your folder
df = pd.read_csv('Loan_Dataset_1000_rows.csv')
df.columns = df.columns.str.strip()

print("Columns found in dataset:")
print(df.columns.tolist())

# 2. Target Distribution Plot
target_candidates = [c for c in df.columns if 'status' in c.lower() or 'approved' in c.lower()]
if target_candidates:
    target_col = target_candidates[0]
    plt.figure(figsize=(6, 4))
    df[target_col].value_counts().plot(kind='bar', color=['#0284c7', '#ef4444'], edgecolor='black')
    plt.title('Loan Approval Target Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Loan Status')
    plt.ylabel('Applicant Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('loan_target_distribution.png')
    plt.close()

# 3. Numeric Correlation Heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])
if not numeric_df.empty:
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

print("EDA analysis completed. Plots saved.")