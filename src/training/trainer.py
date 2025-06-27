"""Training module for Amazon deforestation prediction model."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.training.early_stopping import EarlyStopping


class Trainer:
    """Trainer class for model training and evaluation.

    This class handles the training loop, validation, and early stopping
    for the Amazon deforestation prediction model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device,
        config: Dict[str, Any],
    ):
        """Initialize the trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            loss_fn: Loss function
            device: Device to run training on
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(
        self,
    ) -> Tuple[torch.nn.Module, List[float], List[float]]:
        """Train the model with validation and early stopping.

        Returns:
            Tuple of (trained_model, average_train_losses, average_val_losses)
        """
        epochs: int = self.config["training"]["epochs"]
        train_losses: List[float] = []
        val_losses: List[float] = []
        avg_train_loss: List[float] = []
        avg_val_loss: List[float] = []
        early_stopping: EarlyStopping = EarlyStopping(
            patience=self.config["early_stopping"]["patience"],
            delta=self.config["early_stopping"]["delta"],
            path=self.config["early_stopping"]["path"],
            verbose=self.config["early_stopping"]["verbose"],
        )

        for epoch in range(epochs):
            self.logger.info("Epoch %s/%s", epoch + 1, epochs)

            # training phase
            self.model.train()

            for data, target in self.train_loader:
                # move data to device
                data, target = data.to(self.device), target.to(self.device)

                # zero gradients
                self.optimizer.zero_grad()
                # forward pass
                train_output: torch.Tensor = self.model(data)
                # compute loss
                train_loss_tensor: torch.Tensor = self.loss_fn(
                    train_output, target
                )
                # backward pass
                train_loss_tensor.backward()
                # update weights
                self.optimizer.step()
                # update training losses
                train_losses.append(train_loss_tensor.item())

            # validation phase
            self.model.eval()
            for data, target in self.val_loader:
                # move data to device
                data, target = data.to(self.device), target.to(self.device)

                # forward pass
                val_output: torch.Tensor = self.model(data)
                # compute loss
                val_loss_tensor: torch.Tensor = self.loss_fn(
                    val_output, target
                )
                # update validation losses
                val_losses.append(val_loss_tensor.item())

            # compute average losses
            train_loss: float = float(np.average(train_losses))
            val_loss: float = float(np.average(val_losses))
            avg_train_loss.append(train_loss)
            avg_val_loss.append(val_loss)

            # log losses
            self.logger.info(
                "Train loss: %s, Val loss: %s", train_loss, val_loss
            )

            train_losses = []
            val_losses = []

            # early stopping
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                self.logger.info("Early stopping!")
                break

        return self.model, avg_train_loss, avg_val_loss

    def evaluate(self) -> None:
        """Evaluate the model on test data.

        TODO: Implement evaluation logic
        """
        # return avg_loss, accuracy
        return
