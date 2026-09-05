# Machine Learning Model Evaluation & Comparison

## Overview
This module evaluates predictive models to determine loan approval suitability based on historical financial records (`Loan_Dataset_1000_rows-v2.csv`).

---

## Class Imbalance Context
* **Distribution:** ~76% Rejected (0) vs ~24% Approved (1).
* **Metric Choice:** Standard accuracy is misleading in imbalanced scenarios. Models were prioritized based on **Class 1 F1-Score** and **ROC-AUC**.

---

## Comparison Summary

| Model | Hyperparameters / Preprocessing | Accuracy | Class 1 Recall | Class 1 F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Nearest Neighbors** | `n_neighbors=5`, `StandardScaler` | 76% | 0.15 | 0.23 | 0.6083 |
| **Decision Tree** | `max_depth=5`, `class_weight='balanced'` | **78%** | **0.71** | **0.60** | **0.8289** |

---

## Final Production Selection
* **Selected Model:** **Decision Tree**
* **Reason:** Overcomes the 76/24 rejection skew, achieving a **0.71 recall** and **0.60 F1-score** on approvals compared to KNN's 0.15 recall.
* **Artifact:** Saved as `best_loan_model.pkl` and connected to `loan_prediction_exact_gui.py`.