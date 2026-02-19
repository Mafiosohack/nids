"""
Random Forest Classifier for Intrusion Detection

WHY RANDOM FOREST:
- High accuracy on labeled data (85-95%)
- Classifies attack TYPES (DoS, Probe, R2L, U2R)
- Feature importance reveals what matters most
- Handles imbalanced classes automatically
- Robust and resistant to overfitting

HOW IT WORKS:
- Builds many decision trees on random data subsets
- Each tree votes on the final classification
- Majority vote = final prediction
- Feature importance from information gain at splits
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time


class RandomForestDetector:
    """Random Forest for supervised intrusion detection."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: str = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest classifier.

        Args:
            n_estimators     : Number of trees
            max_depth        : Max depth per tree (None = unlimited)
            min_samples_split: Min samples to split a node
            min_samples_leaf : Min samples in a leaf node
            class_weight     : 'balanced' handles imbalanced classes
            random_state     : Seed for reproducibility
            n_jobs           : -1 uses all CPU cores
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )
        self.is_fitted = False
        self.training_time = None
        self.feature_importance_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'RandomForestDetector':
        """
        Train the Random Forest model.

        Args:
            X: Training features
            y: Training labels

        Returns:
            self
        """
        logger.info(
            f"Training Random Forest on {X.shape[0]} samples..."
        )
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.feature_importance_ = self.model.feature_importances_
        self.is_fitted = True
        logger.success(
            f"Random Forest trained in {self.training_time:.2f}s"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """
        Comprehensive model evaluation with all metrics.

        Args:
            X      : Test features
            y_true : True labels

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating Random Forest...")

        start_time = time.time()
        y_pred = self.predict(X)
        inference_time = time.time() - start_time

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            'recall': recall_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            'f1_score': f1_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            'inference_time_ms': inference_time * 1000,
            'samples_per_second': (
                len(X) / inference_time if inference_time > 0 else 0
            )
        }

        # Classification report
        report = classification_report(
            y_true, y_pred, zero_division=0
        )
        logger.info(f"\nClassification Report:\n{report}")

        logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
        logger.info(f"Precision : {metrics['precision']:.4f}")
        logger.info(f"Recall    : {metrics['recall']:.4f}")
        logger.info(f"F1-Score  : {metrics['f1_score']:.4f}")

        return metrics

    def get_top_features(
        self,
        feature_names: Optional[List[str]] = None,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get the most important features for detection.

        Args:
            feature_names: Names of features
            top_n        : How many top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if feature_names is None:
            feature_names = [
                f"feature_{i}"
                for i in range(len(self.feature_importance_))
            ]

        pairs = list(zip(feature_names, self.feature_importance_))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_n]

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'RandomForestDetector':
        """Load model from disk."""
        detector = RandomForestDetector()
        detector.model = joblib.load(filepath)
        detector.is_fitted = True
        detector.feature_importance_ = detector.model.feature_importances_
        logger.info(f"Model loaded from {filepath}")
        return detector