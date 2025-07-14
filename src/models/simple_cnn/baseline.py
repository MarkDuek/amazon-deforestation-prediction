import torch
import torch.nn as nn


class SimpleBaselineB(nn.Module):
    def __init__(self, in_channels: int, time_steps: int):
        super().__init__()
        # kernel size = (T,1,1) so the conv itself collapses the time axis
        self.conv = nn.Conv3d(
            in_channels,
            1,
            kernel_size=(time_steps, 1, 1),
            padding=(0, 0, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, H, W)
        out = self.conv(x)  # → (B, 1, 1, H, W)
        return out
