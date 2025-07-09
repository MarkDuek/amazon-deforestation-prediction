import torch

class SoftIoULoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Sigmoid to convert logits to probabilities
        probs = torch.sigmoid(inputs)

        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Only use valid targets (targets > 0)
        mask = targets > 0.0
        probs = probs[mask]
        targets = targets[mask]

        # Compute intersection and union
        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets) - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        return 1.0 - iou
