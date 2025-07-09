"""Early stopping implementation for training."""

import logging
from typing import Any, Callable, Dict, Optional

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
        trace_func: Callable = logging.info,
        config: Optional[Dict[str, Any]] = None,
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
        # Track best (highest) score. Will be set after the first epoch.
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.verbose = verbose
        self.trace_func = trace_func
        # Fallback path if no config provided
        self.path = (
            config["save_paths"]["early_stopping"]
            if config is not None
            else "checkpoints/early_stopping.pt"
        )

    def __call__(self, val_loss: float, model: torch.nn.Module):
        """Check if early stopping criteria are met.

        Args:
            val_loss: Current validation loss
            model: Model to potentially save
        """
        # Convert validation loss to a score (higher is better)
        score = -val_loss

        # If this is the first score or the new score is better by at least
        # `delta`, reset the counter and save a checkpoint.
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            self.logger.info(
                "EarlyStopping counter: %s out of %s",
                self.counter,
                self.patience,
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Save model checkpoint when validation loss improves.

        Args:
            val_loss: Current validation loss
            model: Model to save
        """
        if self.verbose:
            self.trace_func(
                "Validation loss decreased ("
                f"{self.best_score:.6f} --> {val_loss:.6f}). "
                "Saving model...",
            )
        torch.save(model.state_dict(), self.path)
