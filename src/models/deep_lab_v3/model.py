import torch
import torch.nn as nn
import einops
import logging
import segmentation_models_pytorch as smp
from typing import Dict, Any

class DeepLabV3(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        self.config = config["model"]["deep_lab_v3"]
        self.time_slice = config["data"]["time_slice"]
        self.logger = logging.getLogger(__name__)

        super(DeepLabV3, self).__init__()
        self.model = smp.DeepLabV3(
            encoder_name=self.config["encoder_name"],
            encoder_weights=self.config["encoder_weights"],
            in_channels=self.config["in_channels"],
            classes=self.config["classes"],
        )

        self.head = nn.Conv3d(1, 1, kernel_size=(self.time_slice, 1, 1))  # (B, 1, T, H, W) -> (B, 1, 1, H, W)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.logger.debug(f"Model received tensor with shape: {x.shape}")
        
        if len(x.shape) == 4:
            # If we receive (C, T, H, W), add batch dimension
            x = x.unsqueeze(0)  # Add batch dimension: (1, C, T, H, W)
            self.logger.debug(f"Added batch dimension, new shape: {x.shape}")
        
        B, C, T, H, W = x.shape # (B, C, T, H, W)
        
        outputs = []
        for t in range(T):
            x_t = x[:, :, t, :, :] # (B, C, H, W)
            x_t = self.model(x_t)
            outputs.append(x_t)
        
        outputs = torch.stack(outputs, dim=2) # (B, 1, T, H, W)
        outputs = self.head(outputs) # (B, 1, 1, H, W)

        return outputs