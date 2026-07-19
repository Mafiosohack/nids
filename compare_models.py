"""
compare_models.py — Honest algorithm bake-off for the live-aligned NIDS model.

Question we are answering: "is there a more accurate ML algorithm than the
Random Forest we currently ship?"

Method (no cherry-picking):
  - Reuse the EXACT live-aligned feature pipeline from train_random_forest_v2.py
    (same NSL-KDD split, same one-hot encoder, same held-out KDDTest+ eval).
  - Train each candidate on KDDTrain+, tune its decision threshold on an INTERNAL
    validation split (never on the test set), then score every model on the
    held-out KDDTest+ set: accuracy / precision / recall / F1(attack) / AUC / FPR.
  - AUC is threshold-independent, so it is the cleanest "which model is better"
    signal; F1 at the tuned threshold shows practical operating performance.

Candidates:
  - Random Forest              (current shipped model — the baseline)
  - HistGradientBoosting       (scikit-learn's gradient-boosting; LightGBM-class,
                                already installed — no extra dependency)
  - ExtraTrees                 (a cheap second opinion)

Usage:
    python compare_models.py             # train + compare, DO NOT touch saved model
    python compare_models.py --promote   # if a challenger wins, save it as the live
                                         # model + refresh models/rf_live_features.json
                                         # (so the live sensor + dashboard use it)

Promotion rule: a challenger replaces the Random Forest ONLY if it wins on
F1(attack) at its tuned threshold AND does not have a worse AUC. Otherwise the
Random Forest stays. Honest by construction.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Reuse the honest pipeline pieces so this is apples-to-apples with production.
from train_random_forest_v2 import (
    load_data, build_matrix, evaluate, pick_threshold, _attack_index, RANDOM_STATE,
)
from feature_engineering.live_features import (
    LIVE_NUMERIC_FEATURES, LIVE_CATEGORICAL_FEATURES,
    LIVE_MODEL_PATH, LIVE_ENCODER_PATH, LIVE_FEATURE_META_PATH,
)


# ─────────────────────────────────────────────────────────────────────
#  CANDIDATE MODELS
# ─────────────────────────────────────────────────────────────────────
def make_candidates():
    """Return {display_name: (estimator, short_note)}. All use balanced class
    weights so the (roughly balanced) attack/normal split is handled the same."""
    return {
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1),
            "baseline (current shipped)",
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                l2_regularization=1.0, early_stopping=True,
                class_weight="balanced", random_state=RANDOM_STATE),
            "gradient boosting (sklearn)",
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(
                n_estimators=400, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1),
            "randomised trees",
        ),
    }


# ─────────────────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE  (works for tree ensembles AND boosting)
# ─────────────────────────────────────────────────────────────────────
def importances_for(model, encoder, Xval, yval, top_n=8):
    """Per-original-feature importance, mapped back from the one-hot layout.

    Tree ensembles expose feature_importances_; HistGradientBoosting does not,
    so we fall back to permutation importance (model-agnostic, honest) on a
    validation subset.
    """
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:
        # Permutation importance on a capped sample keeps this quick.
        n = min(4000, Xval.shape[0])
        idx = np.random.RandomState(RANDOM_STATE).choice(Xval.shape[0], n, replace=False)
        r = permutation_importance(
            model, Xval[idx], yval[idx], n_repeats=5,
            random_state=RANDOM_STATE, n_jobs=-1, scoring="roc_auc")
        imp = np.clip(r.importances_mean, 0, None)

    names = list(LIVE_NUMERIC_FEATURES)
    values = list(imp[:len(LIVE_NUMERIC_FEATURES)])
    offset = len(LIVE_NUMERIC_FEATURES)
    for cat_name, cats in zip(LIVE_CATEGORICAL_FEATURES, encoder.categories_):
        width = len(cats)
        names.append(cat_name)
        values.append(float(imp[offset:offset + width].sum()))
        offset += width

    ranked = sorted(zip(names, values), key=lambda kv: kv[1], reverse=True)[:top_n]
    top = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
    return [{"name": n, "pct": round(100.0 * v / top, 1)} for n, v in ranked]


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--promote", action="store_true",
                    help="Save the winning challenger as the live model if it "
                         "beats the Random Forest.")
    args = ap.parse_args()

    print("[*] Loading NSL-KDD and building the live-aligned feature matrix "
          "(identical to train_random_forest_v2.py)...")
    X_train, X_test, cat_train, cat_test = load_data()
    y_tr = (cat_train != "normal").astype(int).to_numpy()

    Xtr, Xte, encoder = build_matrix(
        X_train, X_test, LIVE_NUMERIC_FEATURES, LIVE_CATEGORICAL_FEATURES)

    # Internal validation split for HONEST per-model threshold tuning (no test leak).
    Xtr2, Xval, ytr2, yval = train_test_split(
        Xtr, y_tr, test_size=0.2, stratify=y_tr, random_state=RANDOM_STATE)

    results = {}
    fitted = {}
    thresholds = {}
    for name, (est, note) in make_candidates().items():
        print(f"\n[*] Training {name} ({note})...")
        # Tune threshold on a model trained on the train-portion only...
        est.fit(Xtr2, ytr2)
        t, vf1, vfpr = pick_threshold(est, Xval, yval)
        thresholds[name] = t
        # ...then refit on ALL training data for the final scored model.
        est_full = est.__class__(**est.get_params())
        est_full.fit(Xtr, y_tr)
        fitted[name] = est_full
        res = evaluate(f"{name} @ tuned {t:.2f}", est_full, Xte, cat_test, threshold=t)
        results[name] = res
        print(f"    (validation-tuned threshold={t:.2f}, val F1={vf1:.4f})")

    # ── Head-to-head summary ──
    print(f"\n{'='*74}\n  BAKE-OFF — honest KDDTest+ performance (higher is better, FPR lower)\n{'='*74}")
    print(f"  {'model':<22}{'thr':>6}{'acc':>9}{'prec':>9}{'recall':>9}"
          f"{'F1':>9}{'AUC':>9}{'FPR':>8}")
    for name, res in results.items():
        print(f"  {name:<22}{thresholds[name]:>6.2f}{res['accuracy']:>9.4f}"
              f"{res['precision']:>9.4f}{res['recall']:>9.4f}{res['f1_attack']:>9.4f}"
              f"{res['auc']:>9.4f}{res['fpr']:>8.4f}")

    baseline = "Random Forest"
    # Rank by AUC first: it is threshold-independent, so it measures model quality
    # without depending on a validation-tuned cut-off that may not transfer to
    # KDDTest+'s novel attacks. F1 is the tiebreak / practical sanity check.
    winner = max(results, key=lambda n: (results[n]["auc"], results[n]["f1_attack"]))
    base_f1, base_auc = results[baseline]["f1_attack"], results[baseline]["auc"]
    win_f1, win_auc = results[winner]["f1_attack"], results[winner]["auc"]

    print(f"\n  Baseline Random Forest: F1={base_f1:.4f}  AUC={base_auc:.4f}")
    print(f"  Best challenger:        {winner}  F1={win_f1:.4f}  AUC={win_auc:.4f}")

    # Promote only if the challenger is better on AUC AND does not regress F1 —
    # i.e. a strictly better model, not one trading false positives for recall.
    genuine_win = (winner != baseline
                   and win_auc > base_auc + 1e-4
                   and win_f1 >= base_f1 - 1e-4)
    if winner == baseline:
        print("  → Random Forest still wins. No change recommended.")
    elif genuine_win:
        print(f"  → {winner} is genuinely better (higher AUC, F1 not worse).")
    else:
        print(f"  → {winner} leads on AUC but regresses F1; "
              f"keeping Random Forest is the safe, honest call.")

    if not args.promote:
        print("\n[*] Report-only mode. Re-run with --promote to adopt a winner.")
        return

    if not genuine_win:
        print("\n[*] --promote set, but no challenger cleanly beat the Random "
              "Forest. Leaving the shipped model unchanged.")
        return

    # ── Promote the winner: save model + refresh metadata the dashboard reads ──
    win_model = fitted[winner]
    win_thr = thresholds[winner]
    joblib.dump(win_model, ROOT / LIVE_MODEL_PATH)
    joblib.dump(encoder, ROOT / LIVE_ENCODER_PATH)   # unchanged, but keep in lockstep

    def row(name, res, active, note):
        return {"name": f"{name} (live)", "active": active, "note": note,
                "accuracy": round(res["accuracy"], 4), "precision": round(res["precision"], 4),
                "recall": round(res["recall"], 4), "f1": round(res["f1_attack"], 4),
                "auc": round(res["auc"], 4), "fpr": round(res["fpr"], 4)}

    models_block = [row(winner, results[winner], True,
                        f"live features @ {win_thr:.2f} (promoted by bake-off)")]
    for name in results:
        if name != winner:
            models_block.append(
                row(name, results[name], False, f"live features @ {thresholds[name]:.2f}"))

    meta = {
        "numeric_features": LIVE_NUMERIC_FEATURES,
        "categorical_features": LIVE_CATEGORICAL_FEATURES,
        "encoder_categories": [c.tolist() for c in encoder.categories_],
        "n_features_out": int(Xtr.shape[1]),
        "trained_on": "KDDTrain+", "evaluated_on": "KDDTest+",
        "train_samples": int(len(X_train)), "test_samples": int(len(X_test)),
        "decision_threshold": win_thr,
        "models": models_block,
        "feature_importance": importances_for(win_model, encoder, Xval, yval),
        "kddtest_accuracy": results[winner]["accuracy"],
        "kddtest_f1_attack": results[winner]["f1_attack"],
        "kddtest_fpr": results[winner]["fpr"],
    }
    with open(ROOT / LIVE_FEATURE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[+] PROMOTED {winner} → {LIVE_MODEL_PATH}")
    print(f"[+] Refreshed metadata → {LIVE_FEATURE_META_PATH}")
    print(f"[+] Live sensor (live_ids_v2.py) will use it on next start "
          f"(threshold {win_thr:.2f}).")


if __name__ == "__main__":
    main()
