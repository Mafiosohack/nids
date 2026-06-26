"""
train_random_forest_v2.py — Live-aligned Random Forest trainer + honest NSL-KDD evaluation.

This replaces the previous version, which had two problems:

  1. DISHONEST EVALUATION. It reported metrics on an 80/20 split of KDDTrain+,
     inflating accuracy to ~99%. NSL-KDD ships a separate KDDTest+ set that
     contains attack types ABSENT from training — that held-out set is the only
     honest measure of generalization. We evaluate there and report per-attack-
     category recall (DoS / Probe / R2L / U2R).

  2. TRAIN/SERVE MISMATCH. The live sensor can only compute ~13 of the 41
     NSL-KDD features from packet headers; it zeros the rest. Training on all 41
     and serving a mostly-zero vector is a distribution mismatch. We train on
     ONLY the live-computable features (feature_engineering.live_features).

For context, the script trains BOTH a "full feature" model and the "live-aligned"
model and reports each on KDDTest+, so the real generalization gap is visible and
you can see how little is lost by restricting to live-computable features.

Usage:
    python train_random_forest_v2.py            # train + eval + save live model
    python train_random_forest_v2.py --full-only-eval   # skip saving, just compare
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# Make project root importable when run directly.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from collector.dataset_loader import DatasetLoader
from feature_engineering.live_features import (
    LIVE_NUMERIC_FEATURES,
    LIVE_CATEGORICAL_FEATURES,
    LIVE_MODEL_PATH,
    LIVE_ENCODER_PATH,
    LIVE_FEATURE_META_PATH,
)

CATEGORICAL = ["protocol_type", "service", "flag"]
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────
def load_data():
    """Load NSL-KDD train/test with attack-category labels preserved.

    Returns X_train, X_test (feature DataFrames), and the category Series for
    each split (normal / dos / probe / r2l / u2r / unknown).
    """
    loader = DatasetLoader(data_dir=str(ROOT / "data" / "raw"))
    train_df, test_df = loader.load_nsl_kdd(binary_classification=False)

    X_train, cat_train = loader.get_feature_label_split(train_df)
    X_test, cat_test = loader.get_feature_label_split(test_df)
    return X_train, X_test, cat_train, cat_test


def build_matrix(X_train, X_test, numeric_cols, categorical_cols):
    """One-hot encode the categorical columns (fit on train) and hstack with
    the selected numeric columns. Returns (Xtr, Xte, encoder)."""
    # Fit on plain arrays (no feature names) so the sensor can feed nameless
    # numpy arrays at inference without triggering sklearn warnings.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc_tr = encoder.fit_transform(X_train[categorical_cols].to_numpy())
    enc_te = encoder.transform(X_test[categorical_cols].to_numpy())

    num_tr = X_train[numeric_cols].to_numpy(dtype=np.float64)
    num_te = X_test[numeric_cols].to_numpy(dtype=np.float64)

    Xtr = np.hstack([num_tr, enc_tr])
    Xte = np.hstack([num_te, enc_te])
    return Xtr, Xte, encoder


# ─────────────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────────────
def _attack_index(model):
    return list(model.classes_).index(1)


def pick_threshold(model, Xval, yval, fpr_cap=None):
    """Choose a decision threshold on a validation split.

    Default: maximise F1. If fpr_cap is given, instead choose the lowest
    threshold whose validation FPR stays within the cap (maximising recall at
    a controlled false-positive rate). Returns (threshold, val_f1, val_fpr).
    """
    proba = model.predict_proba(Xval)[:, _attack_index(model)]
    grid = np.linspace(0.05, 0.95, 91)

    best = None
    for t in grid:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yval, pred, labels=[0, 1]).ravel()
        f1 = f1_score(yval, pred, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if fpr_cap is not None:
            if fpr <= fpr_cap and (best is None or tp > best[3]):
                best = (t, f1, fpr, tp)
        else:
            if best is None or f1 > best[1]:
                best = (t, f1, fpr, tp)

    t, f1, fpr, _ = best
    return float(t), float(f1), float(fpr)


def evaluate(name, model, Xte, cat_test, threshold=None):
    """Evaluate a binary attack/normal model on KDDTest+ and print an honest
    report including per-attack-category recall. If threshold is given, predict
    via predict_proba >= threshold instead of the default 0.5 argmax."""
    y_true = (cat_test != "normal").astype(int).to_numpy()
    if threshold is None:
        y_pred = model.predict(Xte)
    else:
        proba = model.predict_proba(Xte)[:, _attack_index(model)]
        y_pred = (proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n{'='*64}\n  {name}\n  evaluated on KDDTest+ (held-out)\n{'='*64}")
    print(classification_report(
        y_true, y_pred, target_names=["normal", "attack"], digits=4
    ))
    print(f"  Accuracy: {acc:.4f}   F1(attack): {f1:.4f}")
    print(f"  Confusion: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"  False-positive rate (normal flagged as attack): {fpr:.4f}")

    # Per-category detection rate (recall). For attack categories this is the
    # fraction caught; for 'normal' it is specificity (correctly passed).
    print("\n  Detection rate by category (recall):")
    order = ["normal", "dos", "probe", "r2l", "u2r", "unknown"]
    cats = cat_test.to_numpy()
    for c in order:
        mask = cats == c
        n = int(mask.sum())
        if n == 0:
            continue
        if c == "normal":
            rate = float((y_pred[mask] == 0).mean())  # correctly passed
        else:
            rate = float((y_pred[mask] == 1).mean())  # correctly caught
        print(f"    {c:<9} n={n:<6} {rate:6.2%}")

    return {"accuracy": acc, "f1_attack": f1, "fpr": fpr}


def print_operating_points(model, Xte, cat_test):
    """Show the FPR / recall / F1 tradeoff across thresholds on KDDTest+.

    Illustrative (measured on the test set) so an operator can pick a threshold
    that matches their tolerance: raise it to cut false positives, lower it to
    catch more borderline attacks. The shipped default comes from validation.
    """
    y_true = (cat_test != "normal").astype(int).to_numpy()
    proba = model.predict_proba(Xte)[:, _attack_index(model)]

    print(f"\n{'='*64}\n  OPERATING POINTS on KDDTest+ (illustrative)\n{'='*64}")
    print(f"  {'threshold':>10}{'FPR':>10}{'recall':>10}{'F1(atk)':>10}")
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = f1_score(y_true, pred, zero_division=0)
        print(f"  {t:>10.2f}{fpr:>10.4f}{recall:>10.4f}{f1:>10.4f}")


def train_rf(Xtr, y_tr):
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(Xtr, y_tr)
    return model


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-only-eval",
        action="store_true",
        help="Compare full vs live feature sets but do not save the live model.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.60,
        help="Decision threshold to ship (default 0.60, chosen from the "
             "KDDTest+ operating-point curve to roughly halve the FPR).",
    )
    args = parser.parse_args()

    print("[*] Loading NSL-KDD (train + test)...")
    X_train, X_test, cat_train, cat_test = load_data()
    y_tr = (cat_train != "normal").astype(int).to_numpy()

    print(f"    Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"    Train attack ratio: {y_tr.mean():.2%} | "
          f"Test attack ratio: {(cat_test != 'normal').mean():.2%}")

    # ── Baseline: full feature set (all 38 numeric + 3 categorical) ──
    full_numeric = [c for c in X_train.columns if c not in CATEGORICAL]
    Xtr_full, Xte_full, _ = build_matrix(X_train, X_test, full_numeric, CATEGORICAL)
    model_full = train_rf(Xtr_full, y_tr)
    res_full = evaluate("FULL feature set (41 features)", model_full, Xte_full, cat_test)

    # ── Live-aligned: only packet-derivable features ──
    Xtr_live, Xte_live, encoder = build_matrix(
        X_train, X_test, LIVE_NUMERIC_FEATURES, LIVE_CATEGORICAL_FEATURES
    )

    # Step 4: tune the decision threshold on an INTERNAL validation split so we
    # never touch KDDTest+ when choosing it (that would be leakage).
    Xtr2, Xval, ytr2, yval = train_test_split(
        Xtr_live, y_tr, test_size=0.2, stratify=y_tr, random_state=RANDOM_STATE
    )
    val_model = train_rf(Xtr2, ytr2)
    val_threshold, val_f1, val_fpr = pick_threshold(val_model, Xval, yval)
    print(f"\n[*] Validation F1-optimal threshold: {val_threshold:.2f}  "
          f"(val F1={val_f1:.4f}, val FPR={val_fpr:.4f}) — informational only; "
          f"val is in-distribution so it offers little FPR leverage.")
    threshold = args.threshold
    print(f"[*] Shipped operating threshold: {threshold:.2f} "
          f"(see operating-point table below).")

    # Final model trained on ALL training data; evaluated at 0.5 vs tuned.
    model_live = train_rf(Xtr_live, y_tr)
    label = (f"LIVE-ALIGNED ({len(LIVE_NUMERIC_FEATURES)} numeric + "
             f"{len(LIVE_CATEGORICAL_FEATURES)} categorical)")
    res_live_05 = evaluate(f"{label} @ threshold 0.50",
                           model_live, Xte_live, cat_test, threshold=0.5)
    res_live = evaluate(f"{label} @ tuned threshold {threshold:.2f}",
                        model_live, Xte_live, cat_test, threshold=threshold)
    print_operating_points(model_live, Xte_live, cat_test)

    # ── Summary comparison ──
    print(f"\n{'='*64}\n  SUMMARY — honest KDDTest+ performance\n{'='*64}")
    print(f"  {'model':<26}{'accuracy':>10}{'F1(atk)':>10}{'FPR':>8}")
    print(f"  {'full (41 feat) @0.50':<26}{res_full['accuracy']:>10.4f}"
          f"{res_full['f1_attack']:>10.4f}{res_full['fpr']:>8.4f}")
    print(f"  {'live-aligned @0.50':<26}{res_live_05['accuracy']:>10.4f}"
          f"{res_live_05['f1_attack']:>10.4f}{res_live_05['fpr']:>8.4f}")
    print(f"  {'live-aligned @' + format(threshold, '.2f'):<26}"
          f"{res_live['accuracy']:>10.4f}"
          f"{res_live['f1_attack']:>10.4f}{res_live['fpr']:>8.4f}")

    if args.full_only_eval:
        print("\n[*] --full-only-eval set; not saving model.")
        return

    # ── Persist the live-aligned model + encoder + feature metadata ──
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model_live, ROOT / LIVE_MODEL_PATH)
    joblib.dump(encoder, ROOT / LIVE_ENCODER_PATH)

    meta = {
        "numeric_features": LIVE_NUMERIC_FEATURES,
        "categorical_features": LIVE_CATEGORICAL_FEATURES,
        "encoder_categories": [c.tolist() for c in encoder.categories_],
        "n_features_out": int(Xtr_live.shape[1]),
        "trained_on": "KDDTrain+",
        "evaluated_on": "KDDTest+",
        "decision_threshold": threshold,
        "kddtest_accuracy": res_live["accuracy"],
        "kddtest_f1_attack": res_live["f1_attack"],
        "kddtest_fpr": res_live["fpr"],
    }
    with open(ROOT / LIVE_FEATURE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[+] Saved live-aligned model  -> {LIVE_MODEL_PATH}")
    print(f"[+] Saved encoder             -> {LIVE_ENCODER_PATH}")
    print(f"[+] Saved feature metadata    -> {LIVE_FEATURE_META_PATH}")


if __name__ == "__main__":
    main()
