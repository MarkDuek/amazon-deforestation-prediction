"""Early stopping implementation for training."""

import logging
from typing import Callable

import torch


class EarlyStopping:
    """Early stopping utility to stop training when validation loss
    stops improving.

    This class monitors validation loss and stops training when the loss
    stops improving for a specified number of epochs (patience).
    """

    def __init__(
        self,
        patience: int = 10,
        verbose: bool = True,
        delta: float = 0.001,
        path: str = "checkpoint.pt",
        trace_func: Callable = logging.info,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            verbose: Whether to print messages
            delta: Minimum change to qualify as improvement
            path: Path to save the model checkpoint
            trace_func: Function to use for logging
        """
        self.logger = logging.getLogger(__name__)
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.path = path
        self.verbose = verbose
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """Check if early stopping criteria are met.

        Args:
            val_loss: Current validation loss
            model: Model to potentially save
        """
        score = -val_loss

        if self.best_loss is None:
            self.best_loss = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_loss + self.delta:
            self.counter += 1
            self.logger.info(
                "EarlyStopping counter: %s out of %s",
                self.counter,
                self.patience,
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model checkpoint when validation loss improves.

        Args:
            val_loss: Current validation loss
            model: Model to save
        """
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.best_loss:.6f} --> "
                f"{val_loss:.6f}).  Saving model..."
            )
        torch.save(model.state_dict(), self.path)
