"""
train.py
--------
Training loop for the ECG biological age prediction model.

Usage:
    python src/train.py --base_path ./data/ptbxl --epochs 30 --batch_size 64
"""

import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error

from model import ECGAgeNet
from data_loader import load_metadata, annotate_dataframe, build_record_lookup, get_dataloaders


def train(args):
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    df, scp = load_metadata(args.base_path)
    df = annotate_dataframe(df, scp)
    record_lookup = build_record_lookup(args.base_path, sampling_rate=100)

    train_loader, val_loader = get_dataloaders(
        df, record_lookup,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ECGAgeNet(dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.L1Loss()

    best_mae = float('inf')

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for ecg, age in train_loader:
            ecg, age = ecg.to(device), age.to(device)
            pred = model(ecg)
            loss = criterion(pred, age)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ecg, age in val_loader:
                preds += model(ecg.to(device)).cpu().tolist()
                trues += age.tolist()

        mae = mean_absolute_error(trues, preds)
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.2f} | "
              f"Val MAE: {mae:.2f} yrs")

        # Save best model
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  --> Best model saved (MAE: {best_mae:.2f})")

    print(f"\nTraining complete. Best Val MAE: {best_mae:.2f} years")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ECG Biological Age Model')
    parser.add_argument('--base_path',   type=str,   default='./data/ptbxl')
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int,   default=0)
    args = parser.parse_args()
    train(args)
