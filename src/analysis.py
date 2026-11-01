"""
analysis.py
-----------
Statistical analyses for ECG biological age gap study.

Includes:
    - Group comparisons (t-test, Cohen's d)
    - Sex-stratified analysis
    - Summary statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple


# ── Group Comparisons ─────────────────────────────────────────────────────────

def compare_groups(
    result_df: pd.DataFrame,
    gap_col: str = 'age_adjusted_gap',
    reference: str = 'NORM',
    groups: List[str] = ['MI', 'STTC', 'CD', 'HYP']
) -> pd.DataFrame:
    """
    Compare age-adjusted gap between reference and pathological groups.

    Args:
        result_df: DataFrame with superclass and gap columns
        gap_col:   Column name for gap values
        reference: Reference group label
        groups:    Pathological groups to compare against reference

    Returns:
        DataFrame with mean, std, n, t-statistic, p-value, Cohen's d
    """
    ref = result_df[result_df['superclass'] == reference][gap_col].dropna()
    results = []

    for group in groups:
        g = result_df[result_df['superclass'] == group][gap_col].dropna()
        t, p = stats.ttest_ind(g, ref)
        d = (g.mean() - ref.mean()) / np.sqrt((g.std()**2 + ref.std()**2) / 2)

        results.append({
            'group':    group,
            'n':        len(g),
            'mean':     round(g.mean(), 2),
            'std':      round(g.std(), 2),
            'p_value':  p,
            'cohens_d': round(d, 2)
        })

    return pd.DataFrame(results)


def sex_stratified_comparison(
    result_df: pd.DataFrame,
    gap_col: str = 'age_adjusted_gap',
    groups: List[str] = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
) -> pd.DataFrame:
    """
    Compare gap between males and females within each diagnostic group.

    Args:
        result_df: DataFrame with sex (0=female, 1=male) and gap columns
        gap_col:   Column for gap values
        groups:    Diagnostic groups to analyze

    Returns:
        DataFrame with male/female means and sex-difference statistics
    """
    results = []
    for group in groups:
        m = result_df[(result_df['sex'] == 1) & (result_df['superclass'] == group)][gap_col].dropna()
        f = result_df[(result_df['sex'] == 0) & (result_df['superclass'] == group)][gap_col].dropna()
        t, p = stats.ttest_ind(m, f)
        d = (m.mean() - f.mean()) / np.sqrt((m.std()**2 + f.std()**2) / 2)

        results.append({
            'group':         group,
            'female_mean':   round(f.mean(), 2),
            'female_std':    round(f.std(), 2),
            'female_n':      len(f),
            'male_mean':     round(m.mean(), 2),
            'male_std':      round(m.std(), 2),
            'male_n':        len(m),
            'p_value':       p,
            'cohens_d':      round(d, 2)
        })

    return pd.DataFrame(results)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_age_gap_by_category(result_df: pd.DataFrame, save_path: str = None):
    """
    Dot plot of age-adjusted gap with 95% CI by diagnostic category.
    """
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    labels  = ['Normal', 'Myocardial\nInfarction', 'ST/T\nChange',
               'Conduction\nDisorder', 'Hypertrophy']
    colors  = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6', '#3498db']

    means = [result_df[result_df['superclass'] == c]['age_adjusted_gap'].mean() for c in classes]
    sems  = [result_df[result_df['superclass'] == c]['age_adjusted_gap'].sem()  for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (m, s, col) in enumerate(zip(means, sems, colors)):
        ax.errorbar(i, m, yerr=s * 1.96, fmt='o', color=col,
                    markersize=12, capsize=6, capthick=2, linewidth=2)

    ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Age-Adjusted ECG Age Gap\nMean ± 95% CI (years)', fontsize=11)
    ax.set_title('ECG Biological Age Gap by Diagnostic Category\n(PTB-XL, n=21,837)', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sex_stratified(result_df: pd.DataFrame, save_path: str = None):
    """
    Dot plot of age-adjusted gap by sex across diagnostic categories.
    """
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    labels  = ['Normal', 'Myocardial\nInfarction', 'ST/T\nChange',
               'Conduction\nDisorder', 'Hypertrophy']

    x = np.arange(len(classes))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    for sex, label, offset, marker, color in [
        (0, 'Female', -width/2, 'o', '#e91e8c'),
        (1, 'Male',    width/2, 's', '#1565c0')
    ]:
        means = [result_df[(result_df['sex'] == sex) & (result_df['superclass'] == c)]
                 ['age_adjusted_gap'].mean() for c in classes]
        sems  = [result_df[(result_df['sex'] == sex) & (result_df['superclass'] == c)]
                 ['age_adjusted_gap'].sem() for c in classes]
        for i, (m, s) in enumerate(zip(means, sems)):
            ax.errorbar(i + offset, m, yerr=s * 1.96, fmt=marker, color=color,
                        markersize=10, capsize=5, capthick=2, linewidth=2,
                        label=label if i == 0 else '')

    ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Age-Adjusted ECG Age Gap\nMean ± 95% CI (years)', fontsize=11)
    ax.set_title('ECG Biological Age Gap by Sex and Diagnostic Category', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import sys
    result_df = pd.read_csv('results/age_gap_results.csv')

    print("=== Group Comparisons ===")
    print(compare_groups(result_df).to_string(index=False))

    print("\n=== Sex-Stratified Analysis ===")
    print(sex_stratified_comparison(result_df).to_string(index=False))

    plot_age_gap_by_category(result_df, save_path='figures/fig1_age_gap_by_category.png')
    plot_sex_stratified(result_df, save_path='figures/fig2_sex_stratified.png')
