import pandas as pd
import pickle
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset to get encoders and scaler
df = pd.read_csv("personal_tutoring_dataset.csv")

categorical_columns = ["Gender", "Country", "State", "City", "Parent Occupation",
                       "Earning Class", "Course Name", "Material Name"]
special_integer_dropdowns = ["Material Level", "Level of Student", "Level of Course"]

# Encode categorical values and store mappings
label_encoders = {}
category_mappings = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    category_mappings[col] = le.classes_

scaler = StandardScaler()
X = df.drop(columns=["Name", "Assessment Score"])
scaler.fit(X)

# ------------------- GUI ------------------- #
root = tk.Tk()
root.title("Personal Tutoring Prediction Dashboard")
root.geometry("800x800")
root.configure(bg="#dcdcdc")

# Style
style = ttk.Style()
style.theme_use("clam")

style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10))
style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
style.configure("TEntry", padding=4)
style.configure("TCombobox", padding=4)

# Header
header_frame = ttk.Frame(root, padding=20)
header_frame.pack(fill="x")
ttk.Label(header_frame, text="📊 Personal Tutoring Prediction Dashboard", font=("Segoe UI", 16, "bold")).pack()

# Center Frame for form, result and button
center_frame = tk.Frame(root, bg="#dcdcdc")
center_frame.pack(pady=20, expand=True)

# Form
form_frame = ttk.LabelFrame(center_frame, text="Input Student Details", padding=20)
form_frame.pack(padx=20, pady=10)

entries = {}
row = 0
for col in X.columns:
    ttk.Label(form_frame, text=f"{col}:").grid(row=row, column=0, sticky="w", pady=6, padx=5)

    if col in categorical_columns:
        entries[col] = tk.StringVar()
        combo = ttk.Combobox(form_frame, textvariable=entries[col], values=list(category_mappings[col]), width=35, state="readonly")
        combo.grid(row=row, column=1, pady=6, padx=5, sticky="w")
    elif col in special_integer_dropdowns:
        entries[col] = tk.StringVar(value="1")
        combo = ttk.Combobox(form_frame, textvariable=entries[col], values=[str(i) for i in range(1, 13)], width=35, state="readonly")
        combo.grid(row=row, column=1, pady=6, padx=5, sticky="w")
    else:
        entries[col] = ttk.Entry(form_frame, width=38)
        entries[col].grid(row=row, column=1, pady=6, padx=5, sticky="w")
    row += 1

# Prediction Result
result_var = tk.StringVar()
ttk.Label(center_frame, textvariable=result_var, foreground="green", font=("Segoe UI", 12, "bold")).pack(pady=10)

# Predict Button
btn_frame = ttk.Frame(center_frame)
btn_frame.pack(pady=10)
ttk.Button(btn_frame, text="Predict Assessment Score", command=lambda: predict()).pack()

# Predict function
def predict():
    try:
        user_input = {}
        for col in X.columns:
            if col in categorical_columns:
                selected = entries[col].get()
                if selected not in category_mappings[col]:
                    messagebox.showerror("Invalid Input", f"Please select a valid option for {col}.")
                    return
                user_input[col] = label_encoders[col].transform([selected])[0]
            elif col in special_integer_dropdowns:
                user_input[col] = int(entries[col].get())
            else:
                user_input[col] = float(entries[col].get())

        user_df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(user_df)
        prediction = model.predict(scaled_input)[0]
        result_var.set(f"🎯 Predicted Assessment Score: {prediction:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

# Footer
footer = ttk.Label(root, text="© 2025 • Personal Tutoring AI System", font=("Segoe UI", 9), foreground="#555", background="#dcdcdc")
footer.pack(side=tk.BOTTOM, pady=15)

root.mainloop()
