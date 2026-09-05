import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

# Load trained model
model = joblib.load('best_loan_model.pkl')

# Feature names used during training
FEATURES = list(model.feature_names_in_)

def predict_status():
    try:
        # Collect raw values from GUI entries
        data = {
            'Annual_Income': float(entry_income.get()),
            'Monthly_Expenses': float(entry_expenses.get()),
            'Outstanding_Debt': float(entry_debt.get()),
            'Credit_Score': float(entry_cibil.get()),
            'Loan_Amount_Requested': float(entry_amount.get()),
            'Loan_Term': float(entry_term.get())
        }
        
        # Create input dataframe and align columns
        input_df = pd.DataFrame([data])
        for col in FEATURES:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[FEATURES]
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        if prediction == 1:
            lbl_result.config(text="Loan Status: APPROVED", fg="#1b7f3b")
        else:
            lbl_result.config(text="Loan Status: REJECTED", fg="#b3261e")
            
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

# Build UI window
root = tk.Tk()
root.title("Loan Approval Prediction System")
root.geometry("420x480")
root.resizable(False, False)

fields = [
    ("Annual Income (₹):", "entry_income"),
    ("Monthly Expenses (₹):", "entry_expenses"),
    ("Outstanding Debt (₹):", "entry_debt"),
    ("Credit Score (300-850):", "entry_cibil"),
    ("Loan Amount Requested (₹):", "entry_amount"),
    ("Loan Term (Months):", "entry_term")
]

entries = {}
for idx, (label_text, var_name) in enumerate(fields):
    lbl = tk.Label(root, text=label_text, font=("Arial", 10, "bold"))
    lbl.pack(pady=(10 if idx == 0 else 4, 0))
    entry = tk.Entry(root, font=("Arial", 10), justify="center")
    entry.pack(pady=(0, 4))
    entries[var_name] = entry

entry_income = entries["entry_income"]
entry_expenses = entries["entry_expenses"]
entry_debt = entries["entry_debt"]
entry_cibil = entries["entry_cibil"]
entry_amount = entries["entry_amount"]
entry_term = entries["entry_term"]

btn_predict = tk.Button(
    root, text="Check Approval", font=("Arial", 11, "bold"),
    bg="#0284c7", fg="white", padx=10, pady=5, command=predict_status
)
btn_predict.pack(pady=20)

lbl_result = tk.Label(root, text="Loan Status: Awaiting Input", font=("Arial", 12, "bold"))
lbl_result.pack(pady=10)

root.mainloop()