import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------------------------------------------------------
# 1. LOAD DATASET & GENERATE THE IDENTICAL TEST SPLIT
# ---------------------------------------------------------
df = pd.read_csv("Loan_Dataset_1000_rows-v2.csv")
df = df.drop("Applicant_Name", axis=1)

X = df.drop("Loan_Approval_Status", axis=1)
y = df["Loan_Approval_Status"]

# One-hot encode using training pipeline configuration
X = pd.get_dummies(X, drop_first=True)

# Identical random_state and stratify to guarantee test set consistency
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 2. LOAD SAVED ARTIFACTS
# ---------------------------------------------------------
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("decision_tree_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Reindex test set columns to guarantee alignment
X_test_aligned = X_test.reindex(columns=model_columns, fill_value=0)

# Scale features specifically for KNN
X_test_scaled = scaler.transform(X_test_aligned)

# ---------------------------------------------------------
# 3. GENERATE PREDICTIONS & PROBABILITIES
# ---------------------------------------------------------
# KNN Inference
y_pred_knn = knn_model.predict(X_test_scaled)
y_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

# Decision Tree Inference
y_pred_dt = dt_model.predict(X_test_aligned)
y_prob_dt = dt_model.predict_proba(X_test_aligned)[:, 1]

# ---------------------------------------------------------
# 4. COMPUTE PERFORMANCE METRICS
# ---------------------------------------------------------
metrics = {
    "Metric": [
        "Accuracy",
        "Precision (Approved - Class 1)",
        "Recall (Approved - Class 1)",
        "F1-Score (Approved - Class 1)",
        "ROC-AUC Score"
    ],
    "KNN Model (k=8)": [
        accuracy_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_knn, pos_label=1),
        recall_score(y_test, y_pred_knn, pos_label=1),
        f1_score(y_test, y_pred_knn, pos_label=1),
        roc_auc_score(y_test, y_prob_knn)
    ],
    "Decision Tree Model": [
        accuracy_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_dt, pos_label=1),
        recall_score(y_test, y_pred_dt, pos_label=1),
        f1_score(y_test, y_pred_dt, pos_label=1),
        roc_auc_score(y_test, y_prob_dt)
    ]
}

comparison_df = pd.DataFrame(metrics)
comparison_df["KNN Model (k=8)"] = comparison_df["KNN Model (k=8)"].apply(lambda v: f"{v * 100:.2f}%" if "%" not in str(v) else v)
comparison_df["Decision Tree Model"] = comparison_df["Decision Tree Model"].apply(lambda v: f"{v * 100:.2f}%" if "%" not in str(v) else v)

print("\n=======================================================")
print("            MODEL PERFORMANCE COMPARISON               ")
print("=======================================================")
print(comparison_df.to_string(index=False))
print("=======================================================\n")

# ---------------------------------------------------------
# 5. SIDE-BY-SIDE CONFUSION MATRIX PLOT
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# KNN Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=["Not Approved", "Approved"])
disp_knn.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("KNN Confusion Matrix (k=8)")

# Decision Tree Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=["Not Approved", "Approved"])
disp_dt.plot(ax=axes[1], cmap="Oranges", colorbar=False)
axes[1].set_title("Decision Tree Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrices_comparison.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 6. COMBINED ROC CURVE PLOT
# ---------------------------------------------------------
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

auc_knn = roc_auc_score(y_test, y_prob_knn)
auc_dt = roc_auc_score(y_test, y_prob_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color="#1f77b4", lw=2, label=f"KNN (AUC = {auc_knn:.3f})")
plt.plot(fpr_dt, tpr_dt, color="#ff7f0e", lw=2, label=f"Decision Tree (AUC = {auc_dt:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.500)")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.title("Receiver Operating Characteristic (ROC) Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig("roc_curve_comparison.png", dpi=300)
plt.show()