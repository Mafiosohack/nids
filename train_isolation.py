import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Load dataset
data = pd.read_csv("data/raw/KDDTrain+.txt", header=None)

# Label column is second last
label_column = data.iloc[:, -2]

# Keep only normal traffic
normal_data = data[label_column == "normal"]

# Drop label and difficulty columns
X = normal_data.iloc[:, :-2]

print("Raw training shape:", X.shape)

# Categorical columns in NSL-KDD
categorical_cols = [1, 2, 3]  # protocol_type, service, flag

# Separate categorical and numeric
X_cat = X.iloc[:, categorical_cols]
X_num = X.drop(columns=categorical_cols)

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(X_cat)

# Combine numeric + encoded categorical
import numpy as np
X_final = np.hstack((X_num.values, X_cat_encoded))

print("Final training shape:", X_final.shape)

# Train Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_final)

# Save model and encoder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/isolation_forest.pkl")
joblib.dump(encoder, "models/encoder.pkl")

print("Model and encoder saved.")
