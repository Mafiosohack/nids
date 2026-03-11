import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

# ----------------------------
# 1. Load Dataset
# ----------------------------

data = pd.read_csv("data/raw/KDDTrain+.txt", header=None)

labels = data.iloc[:, -2]
y = np.where(labels == "normal", 0, 1)

X = data.iloc[:, :-2]

print("Raw training shape:", X.shape)

# ----------------------------
# 2. Separate categorical + numerical
# ----------------------------

categorical_cols = [1, 2, 3]

X_cat = X.iloc[:, categorical_cols]
X_num = X.drop(columns=categorical_cols)

# ----------------------------
# 3. Encode categorical features
# ----------------------------

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat_encoded = encoder.fit_transform(X_cat)

# Combine
X_final = np.hstack((X_num.values, X_cat_encoded))

print("Final feature shape:", X_final.shape)

# ----------------------------
# 4. Train / Validation Split
# ----------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 5. Train Improved Random Forest
# ----------------------------

model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ----------------------------
# 6. Validation Performance
# ----------------------------

y_pred = model.predict(X_val)

print("\nValidation Results:")
print(classification_report(y_val, y_pred))

# ----------------------------
# 7. Feature Importance (Top 15)
# ----------------------------

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 15 Important Features:")
for i in indices[:15]:
    print(f"Feature {i} Importance: {importances[i]:.6f}")

# ----------------------------
# 8. Save Model + Encoder
# ----------------------------

joblib.dump(model, "models/random_forest_v2.pkl")
joblib.dump(encoder, "models/rf_encoder_v2.pkl")

print("\nModel and encoder saved.")
