"""Dummy model for testing purposes."""

import logging
from typing import Any, Dict

import torch
from torch import nn


class DummyModel(nn.Module):
    """Dummy model that returns a tensor of the expected output shape.

    This model is used for testing the training pipeline with actual
    learnable parameters while keeping computation minimal.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the dummy model.

        Args:
            config: Configuration dictionary containing model
            and data parameters
        """
        super().__init__()
        self.config = config
        self.time_slice = config["data"]["time_slice"]
        self.logger = logging.getLogger(__name__)

        # Add some learnable parameters
        # Simple 3D convolution to process temporal data
        self.conv3d = nn.Conv3d(
            in_channels=5,  # Assuming 4 input channels from satellite data
            out_channels=4,
            kernel_size=(2, 3, 3),
            padding=(0, 1, 1),
        )

        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Linear layer to process pooled features
        self.classifier = nn.Linear(4, 16)

        # Final output layer
        self.output_layer = nn.Linear(16, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that processes input through learnable layers.

        Args:
            x: Input tensor of shape (B, C, T, H, W) from Amazon dataset

        Returns:
            Output tensor of shape (B, 1, 1, H, W)
        """
        self.logger.debug(
            "Dummy model received tensor with shape: %s", x.shape
        )

        # (B, C, T, H, W)
        batch_size, _, _, height, width = x.shape

        # Process through 3D convolution
        conv_out = self.conv3d(x)  # Shape: (B, 8, T-1, H, W)
        self.logger.debug("After conv3d: %s", conv_out.shape)

        # Apply global pooling to get features
        pooled = self.global_pool(conv_out)  # Shape: (B, 8, 1, 1, 1)
        pooled = pooled.view(batch_size, -1)  # Shape: (B, 8)

        # Process through classifier layers
        features = self.classifier(pooled)  # Shape: (B, 64)
        features = torch.relu(features)
        features = self.dropout(features)

        # Get output probability
        output_prob = torch.sigmoid(
            self.output_layer(features)
        )  # Shape: (B, 1)

        # Expand to match expected output shape (B, 1, 1, H, W)
        # This creates a simple prediction broadcast across spatial dimensions
        output = output_prob.view(batch_size, 1, 1, 1, 1).expand(
            -1, -1, -1, height, width
        )

        self.logger.debug("Dummy model output shape: %s", output.shape)

        return output
