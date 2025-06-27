import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.training.early_stopping import EarlyStopping


class Trainer:
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
            self.logger.info(f"Epoch {epoch+1}/{epochs}")

            # training phase
            self.model.train()

            for batch, (data, target) in enumerate(self.train_loader):
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
            self.logger.info(f"Train loss: {train_loss}, Val loss: {val_loss}")

            train_losses = []
            val_losses = []

            # early stopping
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                self.logger.info("Early stopping!")
                break

        return self.model, avg_train_loss, avg_val_loss

    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> None:

        # return avg_loss, accuracy
        return
