"""Custom weighted binary cross entropy loss for deforestation prediction."""

from typing import Any, Dict

import torch


class WeightedBCEWithLogitsLoss(torch.nn.Module):
    """
    Weighted Binary Cross Entropy with Logits Loss.

    A custom loss function that applies weighted binary cross entropy with
    logits to handle class imbalance in binary classification tasks. The loss
    is computed using a validity mask based on the target tensor, applying
    different weights for valid and invalid targets.

    Args:
        config: Configuration dictionary containing loss parameters.
            Expected to have a 'loss' key with 'weight' and 'invalid_weight'
            subkeys.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the WeightedBCEWithLogitsLoss module.

        Args:
            config: Configuration dictionary containing loss parameters.
                   Must contain config['loss']['weight'] for the loss weight
                   and config['loss']['invalid_weight'] for the invalid weight.
        """
        super().__init__()
        self.weight = config["loss"]["weight"]
        self.invalid_weight = config["loss"]["invalid_weight"]

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the weighted binary cross entropy with logits loss.

        Args:
            input: Predicted logits tensor (batch_size, ...).
                  Raw, unnormalized predictions from the model.
            target: Ground truth binary labels tensor of shape
                   (batch_size, ...). Should contain values in [0, 1].

       Returns:
            torch.Tensor: Computed weighted binary cross entropy loss scalar.
                         Uses self.weight for valid targets (target > 0) and
                         self.invalid_weight for invalid targets (target <= 0).
        """
        # compute validity mask
        validity_mask = (target > 0.0).float()

        # compute weights
        weights = torch.where(
            validity_mask == 1, self.weight, self.invalid_weight
        )

        # compute loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input, target, weight=weights, pos_weight= torch.tensor(100, device=input.device, dtype=input.dtype)
        )

        return loss
