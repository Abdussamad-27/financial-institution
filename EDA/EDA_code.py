import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load cleaned dataset
df = pd.read_csv('Loan_Dataset_1000_rows-v2.csv')

# 2. Target Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan_Approval_Status', data=df, palette='Set2')
plt.title('Target Class Distribution (Cleaned Data)')
plt.xlabel('Loan Approval Status')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('loan_target_distribution.png')
plt.close()

# 3. Correlation Heatmap (only numeric columns)
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap (Cleaned Data)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

print("EDA plots regenerated successfully.")
