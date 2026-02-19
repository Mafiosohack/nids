"""
Dataset Loader Module
Handles loading and preprocessing of intrusion detection datasets.
Supports: NSL-KDD, CICIDS2017, UNSW-NB15
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import urllib.request
from loguru import logger


class DatasetLoader:
    """Load and preprocess intrusion detection datasets."""

    NSL_KDD_COLUMNS = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    ATTACK_CATEGORIES = {
        'normal': 'normal',
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
        'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos',
        'udpstorm': 'dos', 'processtable': 'dos', 'mailbomb': 'dos',
        'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe',
        'portsweep': 'probe', 'mscan': 'probe', 'saint': 'probe',
        'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l',
        'phf': 'r2l', 'multihop': 'r2l', 'warezmaster': 'r2l',
        'warezclient': 'r2l', 'spy': 'r2l', 'xlock': 'r2l',
        'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l',
        'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r',
        'perl': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
    }

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_nsl_kdd(self) -> None:
        """Download NSL-KDD dataset if not present."""
        base_url = (
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
        )
        files = ['KDDTrain+.txt', 'KDDTest+.txt']

        for filename in files:
            filepath = self.data_dir / filename
            if filepath.exists():
                logger.info(f"{filename} already exists, skipping download")
                continue
            logger.info(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(base_url + filename, filepath)
                logger.success(f"Downloaded {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                raise

    def load_nsl_kdd(
        self,
        train_file: str = "KDDTrain+.txt",
        test_file: str = "KDDTest+.txt",
        binary_classification: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load NSL-KDD dataset."""
        logger.info("Loading NSL-KDD dataset...")
        self.download_nsl_kdd()

        train_df = pd.read_csv(
            self.data_dir / train_file,
            names=self.NSL_KDD_COLUMNS
        )
        test_df = pd.read_csv(
            self.data_dir / test_file,
            names=self.NSL_KDD_COLUMNS
        )

        train_df = train_df.drop('difficulty', axis=1)
        test_df = test_df.drop('difficulty', axis=1)

        train_df = self._process_labels(train_df, binary_classification)
        test_df = self._process_labels(test_df, binary_classification)

        logger.success(
            f"Loaded NSL-KDD: Train={len(train_df)}, Test={len(test_df)}"
        )
        return train_df, test_df

    def _process_labels(
        self,
        df: pd.DataFrame,
        binary: bool = False
    ) -> pd.DataFrame:
        """Process and clean dataset labels."""
        df['label'] = df['label'].str.replace('.', '', regex=False)
        if binary:
            df['label'] = df['label'].apply(
                lambda x: 'normal' if x == 'normal' else 'attack'
            )
        else:
            df['label'] = df['label'].apply(
                lambda x: self.ATTACK_CATEGORIES.get(x.lower(), 'unknown')
            )
        return df

    def get_feature_label_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into features and labels."""
        X = df.drop('label', axis=1)
        y = df['label']
        return X, y

    def print_dataset_summary(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """Print dataset summary statistics."""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"\nTraining samples : {len(train_df)}")
        print(f"Test samples     : {len(test_df)}")
        print(f"Features         : {len(train_df.columns) - 1}")
        print(f"\nLabel distribution (train):")
        print(train_df['label'].value_counts())
        print("="*60)


def load_dataset(
    dataset_type: str = "nsl-kdd",
    data_dir: str = "data/raw",
    binary: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load intrusion detection dataset."""
    loader = DatasetLoader(data_dir)
    if dataset_type.lower() == "nsl-kdd":
        return loader.load_nsl_kdd(binary_classification=binary)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")