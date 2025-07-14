import torch
import torch.nn as nn

class ResidualBlock3D(nn.Module):
    """
    A simple 3D residual block with two convolutional layers and a skip connection.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_p=0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout  = nn.Dropout3d(p=dropout_p)

        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        return self.relu(out)

class Tiny3DUNet(nn.Module):
    """
    A minimal 3D U-Net with two encoder/decoder levels.

    Args:
        in_channels (int): Number of input feature channels
        time_steps (int): Number of frames in the temporal dimension
        base_channels (int): Channels in the first encoder stage
    """
    def __init__(self, in_channels: int=5, time_steps: int=6, base_channels: int=8):
        super().__init__()
        # Encoder
        self.enc1 = ResidualBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.enc2 = ResidualBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Bottleneck
        self.bottleneck = ResidualBlock3D(base_channels * 2, base_channels * 4)

        # Decoder
        self.up2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec2 = ResidualBlock3D(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(
            base_channels * 2, base_channels,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec1 = ResidualBlock3D(base_channels * 2, base_channels)

        # Collapse time dimension
        self.final_conv = nn.Conv3d(
            base_channels, 1,
            kernel_size=(time_steps, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x1 = self.enc1(x)               # (B, base, T, H, W)
        x2 = self.enc2(self.pool1(x1))  # (B, base*2, T, H/2, W/2)
        x3 = self.bottleneck(self.pool2(x2))  # (B, base*4, T, H/4, W/4)

        u2 = self.up2(x3)                # (B, base*2, T, H/2, W/2)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))
        u1 = self.up1(d2)                # (B, base, T, H, W)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))

        out = self.final_conv(d1)       # (B, 1, 1, H, W)
        return out
