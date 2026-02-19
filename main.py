"""
NIDS — Network Intrusion Detection System
Main entry point for training and detection.

Usage:
  python main.py train  --dataset nsl-kdd --model isolation_forest --binary
  python main.py detect --model-path models/isolation_forest.pkl ...
  python main.py info
"""

import click
import numpy as np
from pathlib import Path


@click.group()
def cli():
    """Network Intrusion Detection System"""
    pass


@cli.command()
@click.option('--dataset',    default='nsl-kdd',
              help='Dataset: nsl-kdd')
@click.option('--model',      default='isolation_forest',
              help='Model: isolation_forest | random_forest')
@click.option('--binary',     is_flag=True,
              help='Binary classification (normal vs attack)')
@click.option('--output-dir', default='models/',
              help='Directory to save trained models')
def train(dataset, model, binary, output_dir):
    """Train intrusion detection model."""

    from utils.logger import setup_logging
    setup_logging(log_level="INFO", log_file="logs/nids.log")
    from loguru import logger

    logger.info("="*70)
    logger.info("NIDS TRAINING MODE")
    logger.info("="*70)
    logger.info(f"Dataset        : {dataset}")
    logger.info(f"Model          : {model}")
    logger.info(
        f"Classification : {'Binary' if binary else 'Multi-class'}"
    )

    # ── Step 1: Load dataset ──────────────────────────────────
    logger.info("\n[1/4] Loading dataset...")
    from collector.dataset_loader import load_dataset
    train_df, test_df = load_dataset(dataset, binary=binary)

    # ── Step 2: Preprocess ────────────────────────────────────
    logger.info("\n[2/4] Preprocessing features...")
    from feature_engineering.preprocessor import preprocess_dataset

    categorical_features = ['protocol_type', 'service', 'flag']

    # Make sure folders exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = preprocess_dataset(
        train_df,
        test_df,
        categorical_features=categorical_features,
        scaling_method='standard',
        encoding_method='onehot',
        save_preprocessor=True,
        preprocessor_path=f"{output_dir}/preprocessor.pkl"
    )

    # ── Step 3: Train ─────────────────────────────────────────
    logger.info(f"\n[3/4] Training {model}...")

    if model == 'isolation_forest':
        from models.isolation_forest import IsolationForestDetector
        detector = IsolationForestDetector(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        detector.fit(X_train)
        model_path = f"{output_dir}/isolation_forest.pkl"

    elif model == 'random_forest':
        from models.random_forest import RandomForestDetector
        detector = RandomForestDetector(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        detector.fit(X_train, y_train)
        model_path = f"{output_dir}/random_forest.pkl"

    else:
        click.echo(f"ERROR: Unknown model '{model}'")
        return

    # ── Step 4: Evaluate ──────────────────────────────────────
    logger.info("\n[4/4] Evaluating model...")
    metrics = detector.evaluate(X_test, y_test)

    # ── Save model ────────────────────────────────────────────
    detector.save(model_path)

    # ── Summary ───────────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Model saved      : {model_path}")
    logger.info(
        f"Preprocessor     : {output_dir}/preprocessor.pkl"
    )
    logger.info("\nPerformance Metrics:")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  F1-Score  : {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc'):
        logger.info(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    logger.success("\nTraining pipeline completed successfully!")


@cli.command()
@click.option('--model-path',       required=True,
              help='Path to trained model')
@click.option('--preprocessor-path',required=True,
              help='Path to preprocessor')
@click.option('--data-path',        required=True,
              help='CSV data file to analyze')
@click.option('--output-path',      default='results/detections.csv',
              help='Output path for results')
def detect(model_path, preprocessor_path, data_path, output_path):
    """Run detection on a dataset file."""

    from utils.logger import setup_logging
    setup_logging(log_level="INFO", log_file="logs/nids.log")
    from loguru import logger
    import pandas as pd

    logger.info("="*70)
    logger.info("NIDS DETECTION MODE")
    logger.info("="*70)

    # Load model and preprocessor
    logger.info("Loading model and preprocessor...")
    from models.isolation_forest import IsolationForestDetector
    from feature_engineering.preprocessor import FeaturePreprocessor

    detector     = IsolationForestDetector.load(model_path)
    preprocessor = FeaturePreprocessor.load(preprocessor_path)

    # Load data
    logger.info(f"Loading data from {data_path}...")
    from collector.dataset_loader import DatasetLoader
    columns = DatasetLoader.NSL_KDD_COLUMNS
    data = pd.read_csv(data_path, names=columns)
    data = data.drop(
        ['difficulty', 'label'], axis=1, errors='ignore'
    )

    # Preprocess
    logger.info("Preprocessing...")
    X = preprocessor.transform(data)

    # Detect
    logger.info("Running detection...")
    predictions    = detector.predict(X)
    anomaly_scores = detector.predict_proba(X)

    # Save results
    results = data.copy()
    results['prediction'] = [
        'anomaly' if p == -1 else 'normal'
        for p in predictions
    ]
    results['anomaly_score'] = anomaly_scores

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    # Summary
    num_anomalies = (predictions == -1).sum()
    rate = num_anomalies / len(predictions) * 100

    logger.info("\n" + "="*70)
    logger.info("DETECTION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total samples    : {len(predictions)}")
    logger.info(
        f"Anomalies found  : {num_anomalies} ({rate:.2f}%)"
    )
    logger.info(f"Results saved    : {output_path}")
    logger.success("Detection completed successfully!")


@cli.command()
def info():
    """Display system information."""
    print("\n" + "="*70)
    print("NIDS — Network Intrusion Detection System")
    print("="*70)
    print("\nAvailable Models:")
    print("  isolation_forest  — Unsupervised anomaly detection")
    print("  random_forest     — Supervised classification")
    print("\nSupported Datasets:")
    print("  nsl-kdd           — NSL-KDD dataset")
    print("\nCommands:")
    print(
        "  python main.py train "
        "--dataset nsl-kdd "
        "--model isolation_forest --binary"
    )
    print(
        "  uvicorn api.main:app --reload  "
        "(start API server)"
    )
    print("\n" + "="*70)


if __name__ == '__main__':
    cli()