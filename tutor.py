from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

# Load dataset
dataset_path = "personal_tutoring_dataset.csv"
df = pd.read_csv("personal_tutoring_dataset.csv")

# Encode categorical columns
categorical_columns = ["Gender", "Country", "State", "City", "Parent Occupation", 
                       "Earning Class", "Course Name", "Material Name"]

label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
   df[col] = label_encoders[col].fit_transform(df[col])


# Define features and target variable
X = df.drop(columns=["Name", "Assessment Score"])  
y = df["Assessment Score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train optimized Random Forest model
optimized_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
optimized_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = optimized_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

import pickle

# Save the trained model
with open("trained_model.pkl", "wb") as file:
    pickle.dump(optimized_model, file)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
