"""
data_loader.py
--------------
PTB-XL dataset loading and preprocessing utilities.

Dataset: PTB-XL — A large publicly available electrocardiography dataset
Source:  https://physionet.org/content/ptb-xl/1.0.3/
"""

import os
import glob
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


# ── Constants ─────────────────────────────────────────────────────────────────

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_metadata(base_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load PTB-XL metadata and SCP statement definitions.

    Args:
        base_path: Root directory of the PTB-XL dataset

    Returns:
        df: Patient/ECG metadata DataFrame
        scp: SCP diagnostic code definitions DataFrame
    """
    df = pd.read_csv(os.path.join(base_path, 'ptbxl_database.csv'))
    df['scp_codes'] = df['scp_codes'].apply(eval)

    scp = pd.read_csv(os.path.join(base_path, 'scp_statements.csv'), index_col=0)

    return df, scp


def annotate_dataframe(df: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'is_normal' and 'superclass' columns to the metadata DataFrame.

    Args:
        df:  Patient metadata DataFrame
        scp: SCP diagnostic definitions DataFrame

    Returns:
        df with new columns: is_normal, superclass
    """
    norm_codes = scp[scp['diagnostic_class'] == 'NORM'].index.tolist()

    def is_normal(codes):
        return any(c in norm_codes for c in codes)

    def get_superclass(codes):
        for code in codes:
            if code in scp.index:
                sc = scp.loc[code, 'diagnostic_class']
                if pd.notna(sc):
                    return sc
        return 'UNKNOWN'

    df = df.copy()
    df['is_normal'] = df['scp_codes'].apply(is_normal)
    df['superclass'] = df['scp_codes'].apply(get_superclass)
    df = df[df['age'].notna() & (df['age'] > 0) & (df['age'] < 110)]

    return df


def build_record_lookup(base_path: str, sampling_rate: int = 100) -> dict:
    """
    Build a dictionary mapping ECG record stems to their full file paths.

    Args:
        base_path:     Root directory of PTB-XL dataset
        sampling_rate: 100 (lr) or 500 (hr). Default: 100

    Returns:
        dict mapping stem (e.g. '00001_lr') to full path without extension
    """
    folder = 'records100' if sampling_rate == 100 else 'records500'
    pattern = os.path.join(base_path, folder, '**', '*.hea')

    lookup = {}
    for path in glob.glob(pattern, recursive=True):
        stem = os.path.splitext(os.path.basename(path))[0]
        lookup[stem] = path[:-4]

    print(f"Found {len(lookup)} records at {sampling_rate}Hz")
    return lookup


# ── Dataset Class ─────────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL ECG records.

    Each sample returns:
        ecg (torch.Tensor): Shape (12, T) — normalized 12-lead ECG
        age (torch.Tensor): Scalar — chronological age in years

    Args:
        df:            DataFrame with ECG metadata (must have 'filename_lr' and 'age')
        record_lookup: Dict mapping filename stem to full path
        sampling_rate: 100 or 500. Default: 100
    """
    def __init__(
        self,
        df: pd.DataFrame,
        record_lookup: dict,
        sampling_rate: int = 100
    ):
        self.df = df.reset_index(drop=True)
        self.lookup = record_lookup
        self.col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        ecg = self._load_ecg(row)
        age = np.float32(row['age'])
        return torch.tensor(ecg), torch.tensor(age)

    def _load_ecg(self, row: pd.Series) -> np.ndarray:
        stem = os.path.basename(row[self.col])
        path = self.lookup.get(stem)
        if path is None:
            raise FileNotFoundError(f"Record not found: {stem}")

        record = wfdb.rdrecord(path)
        sig = record.p_signal.T  # (12, T)

        # Per-lead z-score normalization
        sig = (sig - sig.mean(axis=1, keepdims=True)) / (
            sig.std(axis=1, keepdims=True) + 1e-8
        )
        return sig.astype(np.float32)


# ── DataLoader Factory ────────────────────────────────────────────────────────

def get_dataloaders(
    df: pd.DataFrame,
    record_lookup: dict,
    batch_size: int = 64,
    val_split: float = 0.2,
    num_workers: int = 0,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Split normal ECGs into train/val and return DataLoaders.

    Args:
        df:            Full annotated DataFrame
        record_lookup: Record path lookup dictionary
        batch_size:    Batch size for training. Default: 64
        val_split:     Fraction for validation. Default: 0.2
        num_workers:   DataLoader workers. Default: 0 (Colab-safe)
        random_state:  Random seed. Default: 42

    Returns:
        train_loader, val_loader
    """
    normal_df = df[df['is_normal']].copy()
    train_df, val_df = train_test_split(
        normal_df, test_size=val_split, random_state=random_state
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_loader = DataLoader(
        ECGDataset(train_df, record_lookup),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        ECGDataset(val_df, record_lookup),
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_loader, val_loader
