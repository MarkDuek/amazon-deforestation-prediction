import torch
from typing import Any, Dict
from .softiou import SoftIoULoss

class WeightedBCEWithLogitsAndIoULoss(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.weight = config["loss"]["weight"]
        self.invalid_weight = config["loss"]["invalid_weight"]
        self.pos_weight = config["loss"].get("pos_weight", None)
        self.beta = config["loss"].get("beta", 0.5)

        self.soft_iou = SoftIoULoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        validity_mask = (target > 0.0).float()
        weights = torch.where(validity_mask == 1, self.weight, self.invalid_weight)

        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = torch.tensor(
                self.pos_weight, device=input.device, dtype=input.dtype
            )

        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input, target, weight=weights, pos_weight=pos_weight
        )

        iou_loss = self.soft_iou(input, target)

        return (1 - self.beta) * bce_loss + self.beta * iou_loss
