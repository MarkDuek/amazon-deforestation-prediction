import logging
from typing import Callable

import torch


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        verbose: bool = True,
        delta: float = 0.001,
        path: str = "checkpoint.pt",
        trace_func: Callable = logging.info,
    ):
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
        score = -val_loss

        if self.best_loss is None:
            self.best_loss = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_loss + self.delta:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.best_loss:.6f} --> "
                f"{val_loss:.6f}).  Saving model..."
            )
        torch.save(model.state_dict(), self.path)
