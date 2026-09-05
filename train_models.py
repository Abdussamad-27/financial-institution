import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load dataset (update filename if needed)
df = pd.read_csv('Loan_Dataset_1000_rows-v2.csv')

# 2. Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# 3. Separate features (X) and target (y)
X = df.drop('Loan_Approval_Status', axis=1)
y = df['Loan_Approval_Status']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Scale features (required for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("=== KNN Performance ===")
print(classification_report(y_test, y_pred_knn))
print("ROC-AUC:", roc_auc_score(y_test, knn.predict_proba(X_test_scaled)[:, 1]))

# 7. Train Decision Tree (balanced for 76/24 imbalance)
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n=== Decision Tree Performance ===")
print(classification_report(y_test, y_pred_dt))
print("ROC-AUC:", roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1]))

# 8. Save best model and scaler for your Tkinter UI
joblib.dump(dt, 'best_loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel saved successfully as best_loan_model.pkl!")