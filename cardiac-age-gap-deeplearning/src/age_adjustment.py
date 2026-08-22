"""
age_adjustment.py
-----------------
Age-adjustment of ECG biological age gap.

Rationale:
    Raw age gap is confounded by chronological age — older patients tend
    to have more negative gaps due to model regression-to-mean. We correct
    for this by fitting a linear regression of gap ~ age in the normal cohort
    and computing residuals for all subjects.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_age_adjusted_gap(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute age-adjusted ECG biological age gap.

    Fits a linear model: age_gap ~ chronological_age on normal ECGs only,
    then computes residuals for all subjects. This removes the age-driven
    component of the gap, isolating the pathology-driven signal.

    Args:
        result_df: DataFrame with columns ['age', 'age_gap', 'superclass']

    Returns:
        result_df with new column 'age_adjusted_gap'
    """
    norm_sub = result_df[result_df['superclass'] == 'NORM'][['age', 'age_gap']].dropna()

    lr = LinearRegression()
    lr.fit(norm_sub[['age']], norm_sub['age_gap'])

    print(f"Age-gap regression (NORM): slope={lr.coef_[0]:.4f}, intercept={lr.intercept_:.4f}")

    result_df = result_df.copy()
    result_df['age_adjusted_gap'] = (
        result_df['age_gap'] - lr.predict(result_df[['age']])
    )

    return result_df


def summarize_adjusted_gap(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary statistics of age-adjusted gap by superclass.

    Args:
        result_df: DataFrame with 'superclass' and 'age_adjusted_gap'

    Returns:
        Summary DataFrame with mean, std, count per superclass
    """
    return (
        result_df
        .groupby('superclass')['age_adjusted_gap']
        .agg(['mean', 'std', 'count'])
        .round(2)
    )
