"""
model.py
--------
1D ResNet architecture for ECG biological age prediction.

Architecture:
    Input: (batch, 12, 1000) — 12-lead ECG at 100Hz
    Output: (batch,) — predicted age in years
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual block with two 1D convolutional layers.
    Uses skip connection to preserve gradient flow.
    """
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class ECGAgeNet(nn.Module):
    """
    1D ResNet for age regression from 12-lead ECG waveforms.

    Trained on normal ECGs only to learn age-normative ECG patterns.
    Applied to all ECGs to compute the ECG biological age gap.

    Args:
        dropout (float): Dropout rate in the regression head. Default: 0.3

    Input shape:
        (batch_size, 12, 1000) — 12 leads, 1000 timepoints at 100Hz

    Output shape:
        (batch_size,) — predicted age in years
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            # Initial convolution
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.ReLU(),

            # Block 1
            ResBlock(32),

            # Downsample
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            # Block 2
            ResBlock(64),

            # Downsample
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            # Global average pooling
            nn.AdaptiveAvgPool1d(1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ECGAgeNet()
    x = torch.randn(8, 12, 1000)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {count_parameters(model):,}")
