import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """
    A 3D residual block with two convolutional layers and a skip connection.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1) 
            if in_channels != out_channels 
            else nn.Identity()
        )

    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection applied
        """
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class SimpleCNNWithResiduals(nn.Module):
    """
    A simple 3D CNN with residual connections for time series prediction.
    
    This model uses a U-Net-like architecture with residual blocks for
    encoding and decoding 3D data (time, height, width dimensions).
    The final output is reduced to a single time step.
    """
    
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock3D(5, 16)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # Downsample H, W
        self.enc2 = ResidualBlock3D(16, 32)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.bottleneck = ResidualBlock3D(32, 64)

        self.up1 = nn.ConvTranspose3d(
            64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec1 = ResidualBlock3D(64, 32)
        self.up2 = nn.ConvTranspose3d(
            32, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec2 = ResidualBlock3D(32, 16)

        self.final_conv = nn.Conv3d(16, 1, kernel_size=3, padding=1)
        self.reduce_time = nn.AdaptiveAvgPool3d((1, 256, 256))  # (T=1, H, W)

    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)
                where B=batch, C=channels, T=time, H=height, W=width
                
        Returns:
            torch.Tensor: Output tensor of shape (B, 1, 1, 256, 256)
        """
        # Encoder
        x1 = self.enc1(x)  # (B, 16, 7, 256, 256)
        x2 = self.enc2(self.pool1(x1))  # (B, 32, 7, 128, 128)
        x3 = self.bottleneck(self.pool2(x2))  # (B, 64, 7, 64, 64)

        # Decoder with skip connections
        x = self.up1(x3)  # (B, 32, 7, 128, 128)
        x = self.dec1(torch.cat([x, x2], dim=1))  # (B, 32, 7, 128, 128)

        x = self.up2(x)  # (B, 16, 7, 256, 256)
        x = self.dec2(torch.cat([x, x1], dim=1))  # (B, 16, 7, 256, 256)

        x = self.final_conv(x)  # (B, 1, 7, 256, 256)
        x = self.reduce_time(x)  # (B, 1, 1, 256, 256)
        return x
