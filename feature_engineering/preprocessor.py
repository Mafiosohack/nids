"""
Feature Engineering and Preprocessing Pipeline
Handles encoding, scaling, and transformation of network features.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
import joblib
from pathlib import Path
from loguru import logger


class FeaturePreprocessor:
    """
    Preprocess features for machine learning models.
    Handles categorical encoding and feature scaling.
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        scaling_method: str = 'standard',
        encoding_method: str = 'onehot'
    ):
        self.categorical_features = categorical_features or []
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.label_encoders = {}
        self.onehot_encoder = None
        self.scaler = None
        self.numerical_features = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> 'FeaturePreprocessor':
        """Fit preprocessor on training data."""
        logger.info("Fitting feature preprocessor...")
        X_processed = X.copy()

        # Get numerical feature names
        self.numerical_features = [
            col for col in X.columns
            if col not in self.categorical_features
        ]

        # Encode categoricals
        if self.categorical_features:
            if self.encoding_method == 'onehot':
                cat_data = X_processed[self.categorical_features]
                self.onehot_encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore'
                )
                self.onehot_encoder.fit(cat_data)
            elif self.encoding_method == 'label':
                for col in self.categorical_features:
                    le = LabelEncoder()
                    le.fit(X_processed[col].astype(str))
                    self.label_encoders[col] = le

        # Fit scaler on numerical features
        num_data = X_processed[self.numerical_features]
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        self.scaler.fit(num_data)

        self.is_fitted = True
        logger.success("Feature preprocessor fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X_processed = X.copy()

        # Scale numerical features
        num_data = X_processed[self.numerical_features].values
        num_scaled = self.scaler.transform(num_data)

        # Encode categorical features
        if self.categorical_features:
            if self.encoding_method == 'onehot':
                cat_data = X_processed[self.categorical_features]
                cat_encoded = self.onehot_encoder.transform(cat_data)
                result = np.hstack([num_scaled, cat_encoded])
            elif self.encoding_method == 'label':
                cat_cols = []
                for col in self.categorical_features:
                    le = self.label_encoders[col]
                    encoded = X_processed[col].astype(str).apply(
                        lambda x: le.transform([x])[0]
                        if x in le.classes_ else -1
                    ).values.reshape(-1, 1)
                    cat_cols.append(encoded)
                cat_encoded = np.hstack(cat_cols)
                result = np.hstack([num_scaled, cat_encoded])
        else:
            result = num_scaled

        return result

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def save(self, filepath: str) -> None:
        """Save preprocessor to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'FeaturePreprocessor':
        """Load preprocessor from disk."""
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def preprocess_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_features: List[str],
    scaling_method: str = 'standard',
    encoding_method: str = 'onehot',
    save_preprocessor: bool = True,
    preprocessor_path: str = 'models/preprocessor.pkl'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for train and test datasets.
    Returns: (X_train, X_test, y_train, y_test)
    """
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label'].values
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label'].values

    preprocessor = FeaturePreprocessor(
        categorical_features=categorical_features,
        scaling_method=scaling_method,
        encoding_method=encoding_method
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if save_preprocessor:
        preprocessor.save(preprocessor_path)

    logger.info(
        f"Preprocessing complete: "
        f"Train={X_train_processed.shape}, "
        f"Test={X_test_processed.shape}"
    )

    return X_train_processed, X_test_processed, y_train, y_test