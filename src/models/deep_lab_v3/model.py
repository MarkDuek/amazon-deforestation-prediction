"""DeepLabV3 model implementation for Amazon deforestation prediction."""

import logging
from typing import Any, Dict, List

import segmentation_models_pytorch as smp  # type: ignore[import-untyped]
import torch
from torch import nn


class DeepLabV3(nn.Module):
    """DeepLabV3 model for semantic segmentation with temporal processing.

    This model processes temporal sequences of pre-processed satellite imagery
    to predict deforestation in Amazon rainforest areas.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config["model"]["deep_lab_v3"]
        self.time_slice = config["data"]["time_slice"]
        self.logger = logging.getLogger(__name__)

        super().__init__()
        self.model = smp.DeepLabV3(
            encoder_name=self.config["encoder_name"],
            encoder_weights=self.config["encoder_weights"],
            in_channels=self.config["in_channels"],
            classes=self.config["classes"],
        )

        self.head = nn.Conv3d(
            1, 1, kernel_size=(self.time_slice - 1, 1, 1)
        )  # (B, 1, T, H, W) -> (B, 1, 1, H, W) where T = time_slice-1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DeepLabV3 model.

        Args:
            x: Input tensor of shape (B, C, T, H, W)

        Returns:
            Output tensor of shape (B, 1, 1, H, W)
        """
        if len(x.shape) == 4:
            # If we receive (C, T, H, W), add batch dimension
            x = x.unsqueeze(0)  # Add batch dimension: (1, C, T, H, W)
            self.logger.debug("Added batch dimension, new shape: %s", x.shape)

        self.logger.debug("Model received tensor with shape: %s", x.shape)

        _, _, time_frames, _, _ = x.shape  # (B, C, T, H, W)

        outputs: List[torch.Tensor] = []
        for t in range(time_frames):
            x_t = x[:, :, t, :, :]  # (B, C, H, W)
            x_t = self.model(x_t)
            outputs.append(x_t)

        stacked_output = torch.stack(outputs, dim=2)  # (B, 1, T, H, W)
        final_output = self.head(stacked_output)  # (B, 1, 1, H, W)

        return final_output
