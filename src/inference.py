"""
inference.py
------------
Compute ECG biological age gap for all subjects in PTB-XL.

Usage:
    python src/inference.py --base_path ./data/ptbxl --model_path best_model.pt
"""

import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from model import ECGAgeNet
from data_loader import load_metadata, annotate_dataframe, build_record_lookup, ECGDataset
from age_adjustment import compute_age_adjusted_gap


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load data ─────────────────────────────────────────────────────────────
    df, scp = load_metadata(args.base_path)
    df = annotate_dataframe(df, scp)
    record_lookup = build_record_lookup(args.base_path, sampling_rate=100)

    # ── Load model ────────────────────────────────────────────────────────────
    model = ECGAgeNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # ── Inference on full dataset ──────────────────────────────────────────────
    all_loader = DataLoader(
        ECGDataset(df, record_lookup),
        batch_size=128,
        num_workers=0
    )

    all_preds = []
    with torch.no_grad():
        for ecg, _ in all_loader:
            all_preds += model(ecg.to(device)).cpu().tolist()

    # ── Compute age gap ───────────────────────────────────────────────────────
    result_df = df.copy().reset_index(drop=True)
    result_df['predicted_age'] = all_preds
    result_df['age_gap'] = result_df['predicted_age'] - result_df['age']

    # ── Age-adjust the gap ────────────────────────────────────────────────────
    result_df = compute_age_adjusted_gap(result_df)

    # ── Save results ──────────────────────────────────────────────────────────
    result_df[['age', 'sex', 'superclass', 'predicted_age', 'age_gap', 'age_adjusted_gap']]\
        .to_csv('results/age_gap_results.csv', index=False)

    print(f"Results saved to results/age_gap_results.csv")
    print("\nAge-adjusted gap by superclass:")
    print(result_df.groupby('superclass')['age_adjusted_gap'].agg(['mean','std','count']).round(2))

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ECG Age Gap Inference')
    parser.add_argument('--base_path',  type=str, default='./data/ptbxl')
    parser.add_argument('--model_path', type=str, default='best_model.pt')
    args = parser.parse_args()
    run_inference(args)
