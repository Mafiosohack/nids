"""
train_cicids_live.py — Model B: CIC-IDS2017 retrained onto the LIVE schema.

Trains a MULTICLASS model using only the 10 numeric + 3 categorical features
the live sensor (live_ids_v2.py) can actually compute from packet headers. The
CIC-IDS -> live mapping lives in collector.cicids_loader.load_live_mapped and
mirrors the sensor's runtime feature derivation, so this model is drop-in
servable by a live-schema sensor with no train/serve skew.

The point of Model B: retrain the live detector on modern 2017 attack traffic
(DDoS, PortScan, Web attacks, Bot, patator brute-force) instead of 1999-era
NSL-KDD, WITHOUT changing the sensor's feature contract.

It is saved to NEW artifact paths (models/cicids_live*.pkl) and does NOT touch
the shipped NSL-KDD model (models/rf_live.pkl). Wiring the live sensor + the
dashboard over to this multiclass model is a deliberate follow-up (the current
sensor is binary attack/normal); this script only produces the model + metadata.

Honesty notes:
  * The live schema is deliberately lossy — collapsing 78 flow statistics to 13
    packet-derivable features throws away most of CIC-IDS's signal. Expect
    materially lower per-class recall than Model A. That gap IS the cost of
    being servable from packet headers, and is reported plainly.
  * CIC-IDS's flag counts are coarse (whole-flow aggregates), so the derived
    protocol_type/service/flag/serror/rerror are approximations of the sensor's
    per-packet state machine. See load_live_mapped for specifics.

Usage:
    python train_cicids_live.py                # full data, train + eval + save
    python train_cicids_live.py --nrows 200000 # quick subsample run
    python train_cicids_live.py --no-save      # evaluate only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from collector.cicids_loader import load_live_mapped

RANDOM_STATE = 42
MODEL_PATH = "models/cicids_live.pkl"
ENCODER_PATH = "models/cicids_live_encoder.pkl"
META_PATH = "models/cicids_live_meta.json"
LOW_SUPPORT_THRESHOLD = 100


def build_matrix(X_tr, X_te, numeric_cols, categorical_cols):
    """One-hot the categoricals (fit on train), hstack with numeric — the same
    layout train_random_forest_v2.build_matrix uses, so the live-serving code
    can feed the identical [numeric | one-hot] vector."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc_tr = encoder.fit_transform(X_tr[categorical_cols].to_numpy())
    enc_te = encoder.transform(X_te[categorical_cols].to_numpy())
    num_tr = X_tr[numeric_cols].to_numpy(dtype=np.float64)
    num_te = X_te[numeric_cols].to_numpy(dtype=np.float64)
    return np.hstack([num_tr, enc_tr]), np.hstack([num_te, enc_te]), encoder


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


def live_feature_importance(model, encoder, X_df, y, numeric_features,
                            categorical_features, sample=15000, n_repeats=3,
                            top_n=13) -> list[dict]:
    """Grouped permutation importance over the 13 ORIGINAL live features.

    HGB exposes no native importances, and the model sees one-hot columns; we
    permute each original feature (re-encoding categoricals as a group) and
    measure the macro-F1 drop, so a categorical counts as one feature, not N
    dummies. Scaled so the top feature is 100 for the dashboard."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = min(sample, len(X_df))
    idx = rng.choice(len(X_df), n, replace=False)
    Xs = X_df.iloc[idx].reset_index(drop=True)
    ys = y.iloc[idx].reset_index(drop=True)

    def encode(df):
        num = df[numeric_features].to_numpy(dtype=np.float64)
        cat = encoder.transform(df[categorical_features].to_numpy())
        return np.hstack([num, cat])

    base = f1_score(ys, model.predict(encode(Xs)), average="macro")
    scores = {}
    for feat in numeric_features + categorical_features:
        drops = []
        for _ in range(n_repeats):
            Xp = Xs.copy()
            Xp[feat] = rng.permutation(Xp[feat].to_numpy())
            drops.append(base - f1_score(ys, model.predict(encode(Xp)),
                                         average="macro"))
        scores[feat] = float(np.mean(drops))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
    return [{"name": k, "pct": round(100.0 * v / top, 1)}
            for k, v in ranked[:top_n] if v > 0]


def per_class_report(y_true, y_pred, classes) -> list[dict]:
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    print("[*] Loading CIC-IDS2017 mapped onto the live schema (labelled/ set)...")
    t0 = time.time()
    X, y_multi, y_bin, numeric_features, categorical_features = load_live_mapped(
        nrows=args.nrows
    )
    print(f"    Loaded {len(X):,} flows -> {len(numeric_features)} numeric + "
          f"{len(categorical_features)} categorical in {time.time()-t0:.1f}s")
    classes = list(y_multi.cat.categories)
    print(f"    Classes ({len(classes)}): {classes}")
    print(f"    Attack ratio: {y_bin.mean():.2%}")

    y = y_multi.astype(str)
    # Stratified split needs >=2 samples per class (full data min = 11; guard
    # only bites on tiny --nrows subsamples).
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
    Xtr, Xte, encoder = build_matrix(
        X_tr, X_te, numeric_features, categorical_features
    )
    print(f"    Train: {len(X_tr):,} | Test: {len(X_te):,} | "
          f"encoded width: {Xtr.shape[1]}")

    print("[*] Training HistGradientBoosting (balanced sample weights)...")
    t0 = time.time()
    sw = compute_sample_weight("balanced", y_tr)
    model = build_model()
    model.fit(Xtr, y_tr, sample_weight=sw)
    print(f"    Trained in {time.time()-t0:.1f}s")

    y_pred = model.predict(Xte)
    acc = accuracy_score(y_te, y_pred)
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    weighted_f1 = f1_score(y_te, y_pred, average="weighted")

    yb_true = (y_te != "BENIGN").astype(int)
    yb_pred = (y_pred != "BENIGN").astype(int)
    tn, fp, fn, tp = confusion_matrix(yb_true, yb_pred, labels=[0, 1]).ravel()
    bin_recall = tp / (tp + fn) if (tp + fn) else 0.0
    bin_fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n{'='*64}\n  MODEL B — CIC-IDS on LIVE schema (multiclass, 13 features)\n"
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

    print("\n[*] Computing live-feature importance (grouped permutation)...")
    importances = live_feature_importance(
        model, encoder, X_te, y_te, numeric_features, categorical_features
    )
    for f in importances[:8]:
        print(f"    {f['name']:<20} {f['pct']:.1f}")

    if args.no_save:
        print("\n[*] --no-save set; model not persisted.")
        return

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / MODEL_PATH)
    joblib.dump(encoder, ROOT / ENCODER_PATH)
    meta = {
        "model_type": "HistGradientBoostingClassifier",
        "task": "multiclass",
        "dataset": "CIC-IDS2017 (GeneratedLabelledFlows / labelled set) -> live schema",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "encoder_categories": [c.tolist() for c in encoder.categories_],
        "n_features_out": int(Xtr.shape[1]),
        "classes": classes,
        "train_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "eval_note": "stratified 25% hold-out, in-distribution; live schema is "
                     "lossy (13 packet-derivable features) — see module docstring",
        "serving_note": "drop-in for a live-schema sensor; current live_ids_v2 is "
                        "binary and loads models/rf_live.pkl — wiring is a follow-up",
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
    print(f"[+] Saved encoder  -> {ENCODER_PATH}")
    print(f"[+] Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
