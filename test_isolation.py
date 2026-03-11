import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report

# Load model and encoder
model = joblib.load("models/isolation_forest.pkl")
encoder = joblib.load("models/encoder.pkl")

# Load test data
data = pd.read_csv("data/raw/KDDTest+.txt", header=None)

# Extract labels (second last column)
labels = data.iloc[:, -2]

# Features only
X = data.iloc[:, :-2]

# Same categorical columns
categorical_cols = [1, 2, 3]

X_cat = X.iloc[:, categorical_cols]
X_num = X.drop(columns=categorical_cols)

# Transform using SAME encoder
X_cat_encoded = encoder.transform(X_cat)

X_final = np.hstack((X_num.values, X_cat_encoded))

# Predict
preds = model.predict(X_final)

# IsolationForest: -1 = anomaly, 1 = normal
pred_labels = np.where(preds == -1, "attack", "normal")

# Convert real labels to binary
true_labels = np.where(labels == "normal", "normal", "attack")

print(classification_report(true_labels, pred_labels))
