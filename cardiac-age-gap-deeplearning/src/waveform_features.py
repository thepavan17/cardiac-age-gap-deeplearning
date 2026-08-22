"""
waveform_features.py
--------------------
Per-lead ECG waveform feature extraction and correlation analysis.

Features extracted per lead:
    - Amplitude range (max - min): proxy for QRS amplitude
    - Standard deviation: overall signal variability
    - Signal energy (sum of squares): total power
    - Skewness: waveform asymmetry (key finding: dominant correlate of age gap)
"""

import os
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def extract_features_single(path: str) -> Optional[dict]:
    """
    Extract waveform features from a single ECG record.

    Args:
        path: Full path to record (without extension)

    Returns:
        Dict of feature_name -> value, or None if record not found
    """
    try:
        record = wfdb.rdrecord(path)
        sig = record.p_signal  # (T, 12)
    except Exception:
        return None

    features = {}
    for i, lead in enumerate(LEADS):
        s = sig[:, i]
        features[f'{lead}_amplitude_range'] = s.max() - s.min()
        features[f'{lead}_std']             = s.std()
        features[f'{lead}_energy']          = float(np.sum(s ** 2))
        features[f'{lead}_skewness']        = float(pd.Series(s).skew())

    return features


def extract_features_dataset(
    result_df: pd.DataFrame,
    record_lookup: dict
) -> pd.DataFrame:
    """
    Extract waveform features for all ECGs in the dataset.

    Args:
        result_df:     DataFrame with patient metadata and age_adjusted_gap
        record_lookup: Dict mapping filename stem to full path

    Returns:
        feat_df: DataFrame with features joined to metadata
    """
    feature_list = []
    total = len(result_df)

    for i, (idx, row) in enumerate(result_df.iterrows()):
        if i % 1000 == 0:
            print(f"  Processing {i}/{total}...")

        stem = os.path.basename(row['filename_lr'])
        path = record_lookup.get(stem)
        if path is None:
            continue

        f = extract_features_single(path)
        if f:
            f['idx'] = idx
            feature_list.append(f)

    feat_df = pd.DataFrame(feature_list).set_index('idx')
    feat_df = feat_df.join(
        result_df[['age_adjusted_gap', 'superclass', 'sex', 'age']]
    )

    print(f"Done. Features extracted for {len(feat_df)} records.")
    return feat_df


def correlate_features_with_gap(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation of each waveform feature with age-adjusted gap.

    Args:
        feat_df: DataFrame with waveform features and 'age_adjusted_gap'

    Returns:
        Sorted DataFrame with feature, r, p-value, n
    """
    feature_cols = [
        c for c in feat_df.columns
        if any(x in c for x in ['amplitude_range', 'std', 'energy', 'skewness'])
    ]

    results = []
    for col in feature_cols:
        sub = feat_df[[col, 'age_adjusted_gap']].dropna()
        r, p = stats.pearsonr(sub[col], sub['age_adjusted_gap'])
        results.append({'feature': col, 'r': r, 'p_value': p, 'n': len(sub)})

    return (
        pd.DataFrame(results)
        .sort_values('r', key=abs, ascending=False)
        .reset_index(drop=True)
    )


def clean_feature_name(name: str) -> str:
    """Convert raw column name to readable label."""
    name = name.replace('_amplitude_range', ' Amplitude Range')
    name = name.replace('_skewness', ' Skewness')
    name = name.replace('_std', ' Std Dev')
    name = name.replace('_energy', ' Energy')
    return name


def plot_waveform_correlations(
    corr_df: pd.DataFrame,
    top_n: int = 12,
    save_path: str = None
):
    """
    Horizontal bar chart of top waveform feature correlations.

    Args:
        corr_df:   Output of correlate_features_with_gap()
        top_n:     Number of top features to display
        save_path: Optional path to save figure
    """
    top = corr_df.head(top_n).copy()
    top['clean'] = top['feature'].apply(clean_feature_name)
    top = top.sort_values('r')

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_colors = ['#e74c3c' if r > 0 else '#3498db' for r in top['r']]
    bars = ax.barh(top['clean'], top['r'], color=bar_colors, alpha=0.85, edgecolor='white')

    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Pearson r with Age-Adjusted ECG Age Gap', fontsize=11)
    ax.set_title('Waveform Features Most Correlated with\nCardiac Biological Age Gap', fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    for bar, r in zip(bars, top['r']):
        offset = 0.007 if r > 0 else -0.007
        ha = 'left' if r > 0 else 'right'
        ax.text(r + offset, bar.get_y() + bar.get_height() / 2,
                f'r={r:.3f}', va='center', ha=ha, fontsize=8.5, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.85, label='Positive correlation'),
        Patch(facecolor='#3498db', alpha=0.85, label='Negative correlation')
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
