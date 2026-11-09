# Legacy model: NSL-KDD (the starting point)

The NIDS started on the **NSL-KDD** dataset, from the first commit
(2026-02-19). The live-aligned, honestly evaluated version below dates from
2026-06-26. It was served until **2026-07-19**, when it was replaced by the CIC-IDS2017 models
(`models/cicids_live.pkl`, `models/cicids_flow.pkl`). This page records what it
was, how well it did, and why it was retired, so the history survives after the
dataset files were removed from the working tree on 2026-11-09.

## What it was

| | |
|---|---|
| Dataset | NSL-KDD (`KDDTrain+.txt` 19.2 MB, `KDDTest+.txt` 3.5 MB) |
| Trained on | `KDDTrain+`: 125,973 records |
| Evaluated on | `KDDTest+`: 22,544 records, held out, including attack types absent from training |
| Task | Binary (attack vs. normal) |
| Features | 13 "live-computable" features out of NSL-KDD's 41: 10 numeric (`duration`, `src_bytes`, `dst_bytes`, `land`, `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `srv_rerror_rate`) + 3 categorical (`protocol_type`, `service`, `flag`), one-hot to 94 columns |
| Served model | HistGradientBoosting, decision threshold 0.56 (won the bake-off in `compare_models.py`) |
| Artifacts | `models/rf_live.pkl`, `models/rf_live_encoder.pkl`, `models/rf_live_features.json` (gitignored; regenerable) |

## Results on the held-out KDDTest+ set

Each threshold was tuned on an internal validation split of the training data,
never on the test set.

| Model | Accuracy | Precision | Recall | F1 (attack) | AUC | FPR |
|---|---|---|---|---|---|---|
| **HistGradientBoosting (served)** | **0.818** | **0.964** | 0.707 | 0.815 | **0.970** | **3.5%** |
| Random Forest | 0.794 | 0.921 | 0.698 | 0.794 | 0.936 | 7.9% |
| ExtraTrees | 0.816 | 0.917 | 0.744 | 0.822 | 0.917 | 8.9% |

Feature importance (served model, relative): `src_bytes` 100, `count` 13.4,
`service` 5.3, `dst_bytes` 1.2, `duration` 0.7. In practice it was a
`src_bytes` detector.

## Two lessons that shaped the later work

1. **Evaluation honesty.** The first trainer scored itself on an 80/20 split of
   `KDDTrain+` and reported **~99% accuracy**. On the proper held-out
   `KDDTest+` set, which contains attack types never seen in training, the
   number fell to the ~80% above. That gap is why every later model is reported
   on held-out data with FPR alongside accuracy.
2. **Train/serve mismatch.** The live sensor can only compute about 13 of the 41
   NSL-KDD features from packet headers. Training on all 41 and then serving a
   mostly-zero vector is a silent distribution mismatch. The fix was to train
   only on the live-computable subset (`feature_engineering/live_features.py`).
   The CIC-IDS2017 live model keeps that same 13-feature schema.

## Why it was retired

- **Old traffic.** NSL-KDD is a cleaned-up version of KDD Cup '99, whose
  traffic dates from 1998. Its services and attacks do not resemble a modern lab.
- **Recall.** It missed ~29% of attacks on the held-out set.
- **Binary only.** CIC-IDS2017 provides 15 labelled classes (DoS variants,
  PortScan, brute force, web attacks, Bot and others), so the replacement can
  say *what* it saw and not just *that* it saw something.

The replacement, `models/cicids_live.pkl`, uses the same 13-feature schema and
scores 96.4% accuracy, 99.96% binary recall and 4.4% binary FPR. That figure is
on a stratified in-distribution hold-out, which is an easier test than
KDDTest+'s unseen attack types, so the two sets of numbers are not directly
comparable. See `models/cicids_live_meta.json`.

## The code is still here; the data is not

The training and comparison scripts are kept as a record:
`train_random_forest_v2.py`, `compare_models.py`, `retrain_model.py`,
`train_random_forest.py`, `train_isolation.py`, `test_random_forest.py`,
`test_isolation.py`, `collector/dataset_loader.py`,
`feature_engineering/preprocessor.py`, `feature_engineering/feature_definitions.py`,
`models/random_forest.py`, `models/isolation_forest.py`, `config/config.yaml`,
`utils/config.py`, `utils/logger.py`. **None of them is imported by the live
sensor or the server.**

They need `data/raw/KDDTrain+.txt` and `KDDTest+.txt`, which were deleted on
2026-11-09 to save space. To run them again, restore the files from git history
(this writes them to disk without staging them, and `data/raw/` is gitignored,
so they won't be re-committed by accident):

```
mkdir -p data/raw
git show 9c20132:data/raw/KDDTrain+.txt > data/raw/KDDTrain+.txt
git show 9c20132:data/raw/KDDTest+.txt  > data/raw/KDDTest+.txt
```

or download them from the public NSL-KDD mirror that `collector/dataset_loader.py`
uses (`https://raw.githubusercontent.com/defcom17/NSL_KDD/master/`).
