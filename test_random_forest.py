import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report

# Load model and encoder
model = joblib.load("models/random_forest.pkl")
encoder = joblib.load("models/rf_encoder.pkl")

# Load test dataset
data = pd.read_csv("data/raw/KDDTest+.txt", header=None)

labels = data.iloc[:, -2]
y_true = np.where(labels == "normal", 0, 1)

X = data.iloc[:, :-2]

categorical_cols = [1, 2, 3]

X_cat = X.iloc[:, categorical_cols]
X_num = X.drop(columns=categorical_cols)

X_cat_encoded = encoder.transform(X_cat)

X_final = np.hstack((X_num.values, X_cat_encoded))

y_pred = model.predict(X_final)

print(classification_report(y_true, y_pred))
