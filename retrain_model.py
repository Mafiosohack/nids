"""
retrain_model.py — NIDS ML Model Retraining Script

When to run this:
  - live_ids_v2.py shows pred='0' on everything, even during active attacks
  - You've captured new labelled traffic and want the model to learn from it
  - You want to evaluate why the current model isn't alerting

What this does:
  1. Downloads NSL-KDD dataset (or uses a local copy)
  2. Trains a Random Forest classifier
  3. Saves model.pkl and encoder.pkl to models/
  4. Prints classification report so you can see if it's actually any good

Requirements:
  pip install scikit-learn pandas numpy joblib requests

Run:
  python3 retrain_model.py
"""

import os
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR   = Path("models")
MODEL_PATH   = OUTPUT_DIR / "random_forest_v2.pkl"
ENCODER_PATH = OUTPUT_DIR / "rf_encoder_v2.pkl"

# NSL-KDD column names (41 features + label + difficulty)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# These 3 are the categorical features fed to the encoder
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# All numeric features (everything except categoricals, label, difficulty)
NUMERIC_COLS = [c for c in NSL_KDD_COLUMNS if c not in CATEGORICAL_COLS + ["label", "difficulty"]]

# NSL-KDD download URLs (public dataset)
TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

# ─────────────────────────────────────────────
#  LABEL MAPPING
# ─────────────────────────────────────────────
# NSL-KDD has many attack subcategories. Map them to binary: 0=normal, 1=attack.
# If you want multi-class, change this to keep original labels.
def binarize_label(label: str) -> int:
    return 0 if label.strip().lower() == "normal" else 1

# ─────────────────────────────────────────────
#  DOWNLOAD / LOAD DATA
# ─────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    local_train = Path("KDDTrain+.txt")
    local_test  = Path("KDDTest+.txt")

    frames = []

    for path, url, name in [
        (local_train, TRAIN_URL, "Train"),
        (local_test,  TEST_URL,  "Test"),
    ]:
        if path.exists():
            print(f"[DATA] Loading {name} from local file: {path}")
            df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
        else:
            print(f"[DATA] Downloading NSL-KDD {name} set from GitHub...")
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    content = r.read().decode("utf-8")
                df = pd.read_csv(io.StringIO(content), header=None, names=NSL_KDD_COLUMNS)
                # Save locally so you don't need to re-download
                path.write_text(content)
                print(f"[DATA] Saved to {path}")
            except Exception as e:
                print(f"[DATA] Failed to download {name}: {e}")
                print("[DATA] Place KDDTrain+.txt and KDDTest+.txt in the current directory.")
                raise SystemExit(1)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"[DATA] Total samples: {len(df)}")
    return df

# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print("[PREP] Preprocessing...")

    # Binary label
    df["label_bin"] = df["label"].apply(binarize_label)

    print(f"[PREP] Class distribution:")
    print(f"       Normal:  {(df['label_bin'] == 0).sum()}")
    print(f"       Attack:  {(df['label_bin'] == 1).sum()}")

    # Encode categorical columns
    # OrdinalEncoder handles unknown categories gracefully at inference time
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )
    cat_encoded = encoder.fit_transform(df[CATEGORICAL_COLS])
    cat_df = pd.DataFrame(cat_encoded, columns=CATEGORICAL_COLS, index=df.index)

    # Numeric features
    num_df = df[NUMERIC_COLS].astype(np.float64)

    # Final feature matrix: numeric first, then encoded categoricals
    # This matches the order in live_ids_v2.py: np.hstack((numeric, cat_encoded))
    X = np.hstack((num_df.values, cat_df.values))
    y = df["label_bin"].values

    return X, y, encoder

# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
def train(X, y):
    print("[TRAIN] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[TRAIN] Training samples: {len(X_train)}")
    print(f"[TRAIN] Test samples:     {len(X_test)}")
    print("[TRAIN] Fitting RandomForestClassifier (100 trees)...")
    print("        This may take a minute...")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",   # handles class imbalance
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    clf.fit(X_train, y_train)

    print("[TRAIN] Evaluating on test set...")
    y_pred = clf.predict(X_test)

    print("\n" + "─"*60)
    print("CLASSIFICATION REPORT")
    print("─"*60)
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal (0)", "Attack (1)"]
    ))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives  (correct normal):  {tn}")
    print(f"False Positives (false alarms):     {fp}  ← you want this low")
    print(f"False Negatives (missed attacks):   {fn}  ← you want this low")
    print(f"True Positives  (caught attacks):   {tp}")
    print("─"*60 + "\n")

    return clf

# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def print_top_features(clf, n=10):
    feature_names = NUMERIC_COLS + CATEGORICAL_COLS
    importances = clf.feature_importances_
    top = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:n]
    print("TOP 10 MOST IMPORTANT FEATURES:")
    for name, score in top:
        bar = "█" * int(score * 200)
        print(f"  {name:<35} {score:.4f}  {bar}")
    print()

# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────
def save(clf, encoder):
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(clf,     MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"[SAVE] Model   → {MODEL_PATH}")
    print(f"[SAVE] Encoder → {ENCODER_PATH}")
    print("[SAVE] Done. Restart live_ids_v2.py to use the new model.")

# ─────────────────────────────────────────────
#  VERIFY ENCODER COMPATIBILITY
# ─────────────────────────────────────────────
def verify_compatibility(encoder):
    """
    Quick sanity check: make sure the encoder output shape matches
    what live_ids_v2.py expects.
    """
    sample = np.array([["tcp", "http", "SF"]])
    encoded = encoder.transform(sample)
    total_features = len(NUMERIC_COLS) + encoded.shape[1]
    print(f"[CHECK] Numeric features:    {len(NUMERIC_COLS)}")
    print(f"[CHECK] Categorical encoded: {encoded.shape[1]}")
    print(f"[CHECK] Total feature vector size: {total_features}")
    print("[CHECK] Encoder compatible with live_ids_v2.py ✓\n")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("NIDS MODEL RETRAINING")
    print("=" * 60 + "\n")

    df      = load_dataset()
    X, y, encoder = preprocess(df)
    clf     = train(X, y)

    print_top_features(clf)
    verify_compatibility(encoder)
    save(clf, encoder)
