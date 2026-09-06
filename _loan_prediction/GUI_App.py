import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import pickle

# =========================================================
# 1. LOAD PRE-TRAINED ARTIFACTS (PICKLE FILES)
# =========================================================

try:
    with open("decision_tree_model.pkl", "rb") as f:
        decision_tree_model = pickle.load(f)

    with open("knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("model_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    # Recorded test accuracy scores from the trained models
    # (KNN k=8 scored 81.5%, Decision Tree with depth tuning scored 80.0%)
    knn_accuracy = 81.5
    dt_accuracy = 80.0

except FileNotFoundError as e:
    messagebox.showerror(
        "File Missing",
        f"Could not find required pickle file:\n{e}\n\n"
        f"Make sure decision_tree_model.pkl, knn_model.pkl, scaler.pkl, "
        f"and model_columns.pkl are in this folder!"
    )
    exit()


# =========================================================
# 2. PREDICTION FUNCTION
# =========================================================

def predict_loan():

    try:

        # -----------------------------------------
        # Get user values from GUI
        # -----------------------------------------

        gender = gender_var.get()
        purpose = purpose_var.get()
        loan_type = loan_type_var.get()
        selected_model = model_var.get()

        annual_income = float(annual_income_entry.get())
        loan_amount = float(loan_amount_entry.get())
        loan_term = float(loan_term_entry.get())
        credit_score = float(credit_score_entry.get())
        monthly_expenses = float(monthly_expenses_entry.get())
        outstanding_debt = float(outstanding_debt_entry.get())

        # -----------------------------------------
        # Validate dropdown values
        # -----------------------------------------

        if gender == "":
            messagebox.showwarning("Input Error", "Please select Gender.")
            return

        if purpose == "":
            messagebox.showwarning("Input Error", "Please select Loan Purpose.")
            return

        if loan_type == "":
            messagebox.showwarning("Input Error", "Please select Loan Type.")
            return

        if selected_model == "":
            messagebox.showwarning("Input Error", "Please select a prediction model.")
            return

        # -----------------------------------------
        # Create applicant dataframe
        # -----------------------------------------

        applicant = pd.DataFrame({
            "Gender": [gender],
            "Loan_Purpose": [purpose],
            "Loan_Type": [loan_type],
            "Annual_Income": [annual_income],
            "Loan_Amount_Requested": [loan_amount],
            "Loan_Term": [loan_term],
            "Credit_Score": [credit_score],
            "Monthly_Expenses": [monthly_expenses],
            "Outstanding_Debt": [outstanding_debt]
        })

        # -----------------------------------------
        # Encode categorical values & Align columns
        # -----------------------------------------

        applicant = pd.get_dummies(applicant)

        # Match training schema using pickled columns
        applicant = applicant.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # =====================================================
        # MODEL PREDICTIONS
        # =====================================================

        if selected_model == "KNN":
            # Scale features with pickled StandardScaler
            applicant_scaled = scaler.transform(applicant)
            prediction = knn_model.predict(applicant_scaled)[0]
            model_accuracy = knn_accuracy

        else:
            # Decision Tree works directly on unscaled aligned features
            prediction = decision_tree_model.predict(applicant)[0]
            model_accuracy = dt_accuracy

        # -----------------------------------------
        # Explain prediction
        # -----------------------------------------

        reasons = []
        positive_factors = []

        if credit_score < 600:
            reasons.append(f"⚠ Credit Score is low ({credit_score:.0f})")
        elif credit_score >= 750:
            positive_factors.append(f"✓ Good Credit Score ({credit_score:.0f})")

        if outstanding_debt > annual_income * 0.30:
            reasons.append("⚠ Outstanding Debt is relatively high compared with Annual Income")
        elif outstanding_debt <= annual_income * 0.10:
            positive_factors.append("✓ Low Outstanding Debt")

        if loan_amount > annual_income * 0.50:
            reasons.append("⚠ Requested Loan Amount is high compared with Annual Income")
        elif loan_amount <= annual_income * 0.30:
            positive_factors.append("✓ Loan Amount is reasonable compared with Annual Income")

        if monthly_expenses > annual_income / 12 * 0.50:
            reasons.append("⚠ Monthly Expenses are relatively high")
        elif monthly_expenses <= annual_income / 12 * 0.30:
            positive_factors.append("✓ Monthly Expenses are relatively low")

        # -----------------------------------------
        # Display result
        # -----------------------------------------

        if prediction == 1:
            result_label.config(
                text="LOAN APPROVED ✅",
                fg="green"
            )

            if positive_factors:
                explanation = "Positive factors:\n" + "\n".join(positive_factors)
            else:
                explanation = "Positive factors: Applicant meets the model's learned pattern."

        else:
            result_label.config(
                text="LOAN NOT APPROVED ❌",
                fg="red"
            )

            if reasons:
                explanation = "Possible reasons:\n" + "\n".join(reasons)
            else:
                explanation = (
                    "Possible reasons:\n"
                    "⚠ The applicant's overall feature pattern was classified as not approved by the model."
                )

        model_result_label.config(
            text=(
                f"Model Used: {selected_model}\n"
                f"Model Accuracy: {model_accuracy:.2f}%\n\n"
                f"{explanation}"
            ),
            justify="left"
        )

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numerical values."
        )


# =========================================================
# 3. CLEAR FUNCTION
# =========================================================

def clear_fields():

    gender_var.set("")
    purpose_var.set("")
    loan_type_var.set("")
    model_var.set("")

    annual_income_entry.delete(0, tk.END)
    loan_amount_entry.delete(0, tk.END)
    loan_term_entry.delete(0, tk.END)
    credit_score_entry.delete(0, tk.END)
    monthly_expenses_entry.delete(0, tk.END)
    outstanding_debt_entry.delete(0, tk.END)

    result_label.config(text="")
    model_result_label.config(text="")


# =========================================================
# 4. CREATE TKINTER WINDOW
# =========================================================

root = tk.Tk()

root.title("Loan Approval Prediction System")
root.geometry("650x760")
root.resizable(True, True)


# =========================================================
# TITLE
# =========================================================

title_label = tk.Label(
    root,
    text="Loan Approval Prediction System",
    font=("Arial", 18, "bold")
)
title_label.pack(pady=(10, 2))

subtitle_label = tk.Label(
    root,
    text="KNN and Decision Tree",
    font=("Arial", 11)
)
subtitle_label.pack(pady=(0, 6))


# =========================================================
# MAIN FORM
# =========================================================

form_frame = tk.Frame(root)
form_frame.pack(pady=4)


# GENDER
tk.Label(form_frame, text="Gender:", font=("Arial", 10)).grid(row=0, column=0, padx=10, pady=4, sticky="w")
gender_var = tk.StringVar()
gender_combo = ttk.Combobox(form_frame, textvariable=gender_var, values=["Male", "Female"], state="readonly", width=28)
gender_combo.grid(row=0, column=1, pady=4)


# LOAN PURPOSE
tk.Label(form_frame, text="Loan Purpose:", font=("Arial", 10)).grid(row=1, column=0, padx=10, pady=4, sticky="w")
purpose_var = tk.StringVar()
purpose_combo = ttk.Combobox(
    form_frame,
    textvariable=purpose_var,
    values=["Business", "Card", "Gold", "Home", "Personal", "Study"],
    state="readonly",
    width=28
)
purpose_combo.grid(row=1, column=1, pady=4)


# LOAN TYPE
tk.Label(form_frame, text="Loan Type:", font=("Arial", 10)).grid(row=2, column=0, padx=10, pady=4, sticky="w")
loan_type_var = tk.StringVar()
loan_type_combo = ttk.Combobox(form_frame, textvariable=loan_type_var, values=["Secured", "Unsecured"], state="readonly", width=28)
loan_type_combo.grid(row=2, column=1, pady=4)


# ANNUAL INCOME
tk.Label(form_frame, text="Annual Income:", font=("Arial", 10)).grid(row=3, column=0, padx=10, pady=4, sticky="w")
annual_income_entry = tk.Entry(form_frame, width=31)
annual_income_entry.grid(row=3, column=1, pady=4)


# LOAN AMOUNT
tk.Label(form_frame, text="Loan Amount Requested:", font=("Arial", 10)).grid(row=4, column=0, padx=10, pady=4, sticky="w")
loan_amount_entry = tk.Entry(form_frame, width=31)
loan_amount_entry.grid(row=4, column=1, pady=4)


# LOAN TERM
tk.Label(form_frame, text="Loan Term (Months):", font=("Arial", 10)).grid(row=5, column=0, padx=10, pady=4, sticky="w")
loan_term_entry = tk.Entry(form_frame, width=31)
loan_term_entry.grid(row=5, column=1, pady=4)


# CREDIT SCORE
tk.Label(form_frame, text="Credit Score:", font=("Arial", 10)).grid(row=6, column=0, padx=10, pady=4, sticky="w")
credit_score_entry = tk.Entry(form_frame, width=31)
credit_score_entry.grid(row=6, column=1, pady=4)


# MONTHLY EXPENSES
tk.Label(form_frame, text="Monthly Expenses:", font=("Arial", 10)).grid(row=7, column=0, padx=10, pady=4, sticky="w")
monthly_expenses_entry = tk.Entry(form_frame, width=31)
monthly_expenses_entry.grid(row=7, column=1, pady=4)


# OUTSTANDING DEBT
tk.Label(form_frame, text="Outstanding Debt:", font=("Arial", 10)).grid(row=8, column=0, padx=10, pady=4, sticky="w")
outstanding_debt_entry = tk.Entry(form_frame, width=31)
outstanding_debt_entry.grid(row=8, column=1, pady=4)


# MODEL SELECTION
tk.Label(form_frame, text="Select Model:", font=("Arial", 10, "bold")).grid(row=9, column=0, padx=10, pady=4, sticky="w")
model_var = tk.StringVar()
model_combo = ttk.Combobox(
    form_frame,
    textvariable=model_var,
    values=["KNN", "Decision Tree"],
    state="readonly",
    width=28
)
model_combo.grid(row=9, column=1, pady=4)


# =========================================================
# BUTTONS
# =========================================================

button_frame = tk.Frame(root)
button_frame.pack(pady=8)

predict_button = tk.Button(
    button_frame,
    text="Check Loan Status",
    command=predict_loan,
    font=("Arial", 11, "bold"),
    width=18,
    height=1
)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_fields,
    font=("Arial", 11),
    width=10,
    height=1
)
clear_button.grid(row=0, column=1, padx=10)


# =========================================================
# RESULT
# =========================================================

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=(8, 2))

model_result_label = tk.Label(root, text="", font=("Arial", 10))
model_result_label.pack(pady=(0, 10))


# =========================================================
# START APPLICATION
# =========================================================

root.mainloop()