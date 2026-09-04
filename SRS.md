# Multi-Loan Approval Prediction System (INR ₹)
A machine learning classification project evaluating **K-Nearest Neighbors (KNN)** and **Decision Tree** models with a **Tkinter GUI** front-end.

---

## 1. Dataset Specification
- **Source:** Kaggle — *Loan Approval Dataset* (Arbaaz Tamboli)
- **Problem Type:** Supervised Binary Classification (`Approved`: 1 / `Rejected`: 0)
- **Currency Standard:** Indian Rupees (₹ INR / Lakhs)
- **Credit Benchmark:** CIBIL Score (300 to 900)

---

## 2. Dataset Schema & GUI Mapping

| Feature Name | Data Type | Standard / Unit (₹) | Tkinter GUI Component |
| :--- | :--- | :--- | :--- |
| `Loan_Purpose` | Categorical | Home, Personal, Business, Study, Gold, Card | Dropdown: Loan Category |
| `Loan_Type` | Categorical | Secured / Unsecured | Auto-selected / Toggle |
| `Annual_Income` | Numeric | ₹3,00,000 to ₹50,00,000+ | Entry: Annual Income (₹) |
| `Loan_Amount_Requested` | Numeric | ₹50,000 to ₹1,50,00,000 | Entry: Requested Loan (₹) |
| `Loan_Term` | Numeric | 12 to 240 Months | Entry / Slider: Tenure |
| `Credit_Score` | Numeric | 300 to 900 | Entry: CIBIL Score |
| `Monthly_Expenses` | Numeric | Monthly Obligations (₹) | Entry: Monthly Expenses (₹) |
| `Outstanding_Debt` | Numeric | Current Debt (₹) | Entry: Existing Liabilities (₹) |
| `Loan_Approval_Status` | Target | `Approved` (1) / `Rejected` (0) | Color Status Banner (Green / Red) |

---

## 3. System Requirements

### Functional Requirements
- **Input Validation:** Reject non-numeric entries, enforce positive rupee values, and bound CIBIL between 300 and 900.
- **Data Pipeline:** One-hot encode categorical features; fit and save `StandardScaler` for numeric scaling.
- **Dual Inference:** Allow users to predict using KNN, Decision Tree, or run a simultaneous comparison.
- **Probability Scores:** Display approval confidence percentage via `predict_proba()`.
- **Model Persistence:** Save offline trained pipelines as `scaler.pkl`, `knn_model.pkl`, and `dt_model.pkl`.

### Non-Functional Requirements
- **Inference Latency:** GUI response time under 200 ms per prediction.
- **Distance Metric Integrity:** Strict scaling required to prevent high rupee values from dominating KNN distance.
- **Reproducibility:** Seed fixed at `random_state=42` across all splits and cross-validations.
- **Fault Tolerance:** Invalid inputs display clean Tkinter message dialogs without crashing.

---

## 4. Algorithm Comparison Architecture

| Evaluation Metric | K-Nearest Neighbors (KNN) | Decision Tree Classifier |
| :--- | :--- | :--- |
| **Scaling Sensitivity** | Mandatory (`StandardScaler`) | Invariant (No scaling required) |
| **Decision Logic** | Euclidean distance to nearest applicant vectors | Orthogonal cutoff rules (IF-ELSE nodes) |
| **Inference Latency** | $\mathcal{O}(nd)$ distance computations | $\mathcal{O}(\text{depth})$ tree traversal |
| **Hyperparameters** | `n_neighbors`, `weights`, `metric` | `criterion`, `max_depth`, `min_samples_split` |
