"""
Isolation Forest Anomaly Detection Model

WHY ISOLATION FOREST:
- No labeled data required (unsupervised)
- Fast training: O(n log n) complexity
- Works well with high-dimensional data
- Built-in anomaly scoring
- Designed specifically for anomaly detection

HOW IT WORKS:
- Builds random decision trees on random feature subsets
- Anomalies are easier to isolate (need fewer splits)
- Normal points require more splits to isolate
- Score = average path length across all trees
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
from loguru import logger
import time


class IsolationForestDetector:
    """Isolation Forest for network intrusion detection."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            n_estimators  : Number of trees (more = more accurate, slower)
            max_samples   : Samples per tree (256 is fast and effective)
            contamination : Expected % of anomalies (0.1 = 10%)
            random_state  : Seed for reproducibility
            n_jobs        : -1 uses all CPU cores
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )
        self.contamination = contamination
        self.is_fitted = False
        self.training_time = None

    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """
        Train the Isolation Forest on normal traffic.

        Args:
            X: Training features (numpy array)

        Returns:
            self (for method chaining)
        """
        logger.info(
            f"Training Isolation Forest on {X.shape[0]} samples..."
        )
        start_time = time.time()
        self.model.fit(X)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        logger.success(
            f"Isolation Forest trained in {self.training_time:.2f}s"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in traffic.

        Returns:
            Array of 1 (normal) or -1 (anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores normalized to [0, 1].
        Higher score = more anomalous = more suspicious.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Raw scores: lower (more negative) = more anomalous
        raw_scores = -self.model.score_samples(X)

        # Normalize to [0, 1]
        min_s = raw_scores.min()
        max_s = raw_scores.max()
        if max_s - min_s > 0:
            return (raw_scores - min_s) / (max_s - min_s)
        return np.zeros_like(raw_scores)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """
        Evaluate model performance with full metrics.

        Args:
            X      : Test features
            y_true : True labels (string or numeric)

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating Isolation Forest...")

        start_time = time.time()
        y_pred_raw = self.predict(X)
        inference_time = time.time() - start_time

        # Convert: -1 (anomaly) -> 1 (attack), 1 (normal) -> 0
        y_pred = (y_pred_raw == -1).astype(int)

        # Convert string labels to binary
        if isinstance(y_true[0], str):
            y_binary = (y_true != 'normal').astype(int)
        else:
            y_binary = y_true

        metrics = {
            'accuracy': accuracy_score(y_binary, y_pred),
            'precision': precision_score(
                y_binary, y_pred, zero_division=0
            ),
            'recall': recall_score(
                y_binary, y_pred, zero_division=0
            ),
            'f1_score': f1_score(
                y_binary, y_pred, zero_division=0
            ),
            'confusion_matrix': confusion_matrix(
                y_binary, y_pred
            ).tolist(),
            'inference_time_ms': inference_time * 1000,
            'samples_per_second': (
                len(X) / inference_time if inference_time > 0 else 0
            )
        }

        # ROC-AUC score
        try:
            anomaly_scores = self.predict_proba(X)
            metrics['roc_auc'] = roc_auc_score(y_binary, anomaly_scores)
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            metrics['roc_auc'] = None

        # Log results
        logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
        logger.info(f"Precision : {metrics['precision']:.4f}")
        logger.info(f"Recall    : {metrics['recall']:.4f}")
        logger.info(f"F1-Score  : {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            logger.info(f"ROC-AUC   : {metrics['roc_auc']:.4f}")

        return metrics

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'IsolationForestDetector':
        """Load model from disk."""
        detector = IsolationForestDetector()
        detector.model = joblib.load(filepath)
        detector.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
        return detector