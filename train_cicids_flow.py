"""
train_cicids_flow.py — Model A: dedicated CIC-IDS2017 flow classifier.

Trains a MULTICLASS HistGradientBoosting model on the full ~78 flow-statistics
features (the features/ set), predicting the specific attack type per flow
(BENIGN / DDoS / PortScan / DoS Hulk / Bot / Web Attack ... / Heartbleed ...).

Why HistGradientBoosting: it consumes CIC-IDS's NaN (from Inf'd rate columns)
natively, scales to ~2.8M rows in low memory, and is the model family this
project standardised on (see compare_models.py). Class imbalance — BENIGN is
80% of rows, Heartbleed is 11 rows — is handled with balanced sample weights.

Honesty notes:
  * CIC-IDS ships no canonical train/test split, so we use a stratified 75/25
    hold-out with a fixed seed. This is in-distribution (same capture week), so
    metrics here are an optimistic upper bound relative to a fresh-network
    deployment — reported plainly rather than dressed up.
  * The three rarest classes (Heartbleed, Web Attack - Sql Injection,
    Infiltration) have double-digit support; their per-class recall is noisy and
    labelled as low-support in the report.

Usage:
    python train_cicids_flow.py               # full data, train + eval + save
    python train_cicids_flow.py --nrows 200000  # quick subsample run
    python train_cicids_flow.py --no-save     # evaluate only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from collector.cicids_loader import load_flow_features

RANDOM_STATE = 42
MODEL_PATH = "models/cicids_flow.pkl"
META_PATH = "models/cicids_flow_meta.json"
LOW_SUPPORT_THRESHOLD = 100  # classes with fewer test samples flagged as noisy


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.1,
        max_iter=300,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )


def per_class_report(y_true, y_pred, classes) -> list[dict]:
    """Per-class precision/recall/f1/support, most-supported first."""
    rep = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    rows = []
    for c in classes:
        r = rep[c]
        rows.append({
            "class": c,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1-score"], 4),
            "support": int(r["support"]),
            "low_support": int(r["support"]) < LOW_SUPPORT_THRESHOLD,
        })
    rows.sort(key=lambda d: d["support"], reverse=True)
    return rows


def top_permutation_importances(model, X_test, y_test, feature_names,
                                sample=15000, top_n=12) -> list[dict]:
    """Permutation importance on a subsample (HGB exposes no native importances).
    Scaled so the top feature is 100 for the dashboard."""
    n = min(sample, len(X_test))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), n, replace=False)
    Xs, ys = X_test.iloc[idx], y_test.iloc[idx]
    r = permutation_importance(
        model, Xs, ys, n_repeats=3, random_state=RANDOM_STATE,
        scoring="f1_macro", n_jobs=-1,
    )
    order = np.argsort(r.importances_mean)[::-1][:top_n]
    top = float(r.importances_mean[order[0]]) or 1.0
    return [
        {"name": feature_names[i], "pct": round(100.0 * r.importances_mean[i] / top, 1)}
        for i in order if r.importances_mean[i] > 0
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nrows", type=int, default=None,
                    help="Subsample N rows for a quick run (default: all ~2.8M).")
    ap.add_argument("--no-save", action="store_true", help="Evaluate only.")
    ap.add_argument("--skip-importance", action="store_true",
                    help="Skip permutation importance (saves ~1-2 min).")
    args = ap.parse_args()

    print("[*] Loading CIC-IDS2017 flow features (features/ set)...")
    t0 = time.time()
    X, y_multi, y_bin, feature_names = load_flow_features(nrows=args.nrows)
    print(f"    Loaded {len(X):,} flows x {len(feature_names)} features "
          f"in {time.time()-t0:.1f}s")
    classes = list(y_multi.cat.categories)
    print(f"    Classes ({len(classes)}): {classes}")
    print(f"    Attack ratio: {y_bin.mean():.2%}")

    y = y_multi.astype(str)
    # Stratified split needs >=2 samples per class. Full data satisfies this
    # (min class = 11); only tiny --nrows subsamples can drop below it, so guard.
    vc = y.value_counts()
    too_rare = vc[vc < 2].index.tolist()
    if too_rare:
        keep = ~y.isin(too_rare)
        print(f"    [warn] dropping {len(too_rare)} class(es) with <2 rows in this "
              f"sample: {too_rare}")
        X, y = X[keep], y[keep]
        classes = [c for c in classes if c not in too_rare]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    print(f"    Train: {len(X_tr):,} | Test: {len(X_te):,}")

    print("[*] Training HistGradientBoosting (balanced sample weights)...")
    t0 = time.time()
    sw = compute_sample_weight("balanced", y_tr)
    model = build_model()
    model.fit(X_tr, y_tr, sample_weight=sw)
    print(f"    Trained in {time.time()-t0:.1f}s")

    # ── Evaluation ──
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    weighted_f1 = f1_score(y_te, y_pred, average="weighted")

    # Binary view (any-attack detection) derived from the multiclass prediction.
    yb_true = (y_te != "BENIGN").astype(int)
    yb_pred = (y_pred != "BENIGN").astype(int)
    tn, fp, fn, tp = confusion_matrix(yb_true, yb_pred, labels=[0, 1]).ravel()
    bin_recall = tp / (tp + fn) if (tp + fn) else 0.0
    bin_fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n{'='*64}\n  MODEL A — CIC-IDS flow classifier (multiclass)\n"
          f"  stratified 25% hold-out (in-distribution)\n{'='*64}")
    print(f"  Accuracy:            {acc:.4f}")
    print(f"  Balanced accuracy:   {bal_acc:.4f}")
    print(f"  Macro F1:            {macro_f1:.4f}")
    print(f"  Weighted F1:         {weighted_f1:.4f}")
    print(f"  Binary (any-attack): recall={bin_recall:.4f}  FPR={bin_fpr:.4f}")
    print(f"\n{classification_report(y_te, y_pred, zero_division=0, digits=4)}")

    rows = per_class_report(y_te, y_pred, classes)
    print("  Per-class (sorted by support):")
    for r in rows:
        tag = "  [low-support]" if r["low_support"] else ""
        print(f"    {r['class']:<28} recall={r['recall']:.4f} "
              f"f1={r['f1']:.4f} n={r['support']}{tag}")

    importances = []
    if not args.skip_importance:
        print("\n[*] Computing permutation importance (subsample)...")
        importances = top_permutation_importances(model, X_te, y_te, feature_names)
        for f in importances[:8]:
            print(f"    {f['name']:<32} {f['pct']:.1f}")

    if args.no_save:
        print("\n[*] --no-save set; model not persisted.")
        return

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / MODEL_PATH)
    meta = {
        "model_type": "HistGradientBoostingClassifier",
        "task": "multiclass",
        "dataset": "CIC-IDS2017 (MachineLearningCSV / features set)",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "classes": classes,
        "train_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "eval_note": "stratified 25% hold-out, in-distribution (same capture week)",
        "metrics": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "binary_recall": round(float(bin_recall), 4),
            "binary_fpr": round(float(bin_fpr), 4),
        },
        "per_class": rows,
        "feature_importance": importances,
    }
    with open(ROOT / META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[+] Saved model    -> {MODEL_PATH}")
    print(f"[+] Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
