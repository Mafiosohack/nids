import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load training dataset
data = pd.read_csv("data/raw/KDDTrain+.txt", header=None)

# Labels (second last column)
labels = data.iloc[:, -2]

# Convert to binary
y = np.where(labels == "normal", 0, 1)

# Features only
X = data.iloc[:, :-2]

# Categorical columns
categorical_cols = [1, 2, 3]

X_cat = X.iloc[:, categorical_cols]
X_num = X.drop(columns=categorical_cols)

# Encode categorical
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(X_cat)

X_final = np.hstack((X_num.values, X_cat_encoded))

print("Final feature shape:", X_final.shape)

# Split validation
X_train, X_val, y_train, y_val = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest.pkl")
joblib.dump(encoder, "models/rf_encoder.pkl")

print("Random Forest model saved.")
