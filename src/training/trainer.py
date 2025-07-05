"""Training module for Amazon deforestation prediction model."""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.training.early_stopping import EarlyStopping
from src.utils.utils import (export_memory_snapshot,
                             start_record_memory_history,
                             stop_record_memory_history)


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

    def _format_time(self, seconds: float) -> str:
        """Format time duration in a human-readable format.
        
        Args:
            seconds: Time duration in seconds
            
        Returns:
            Formatted time string (e.g., "1h 23m 45s")
        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {mins}m {secs}s"

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

        # move model to device
        self.model.to(self.device)
        self.logger.info("Model moved to device: %s", self.device)

        # record memory history
        if self.config["memory_record"]["enabled"]:
            start_record_memory_history(
                self.logger, self.config["memory_record"]["num_events"]
            )

        # Training start time
        training_start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.logger.info(
                "Epoch %s/%s started at %s",
                epoch + 1,
                epochs,
                time.strftime("%H:%M:%S", time.localtime(epoch_start_time))
            )

            # training phase
            self.model.train()

            # Progress bar for training batches
            train_pbar = tqdm(
                self.train_loader,
                desc=f"Training Epoch {epoch + 1}/{epochs}",
                unit="batch",
                leave=False
            )

            for data, target in train_pbar:
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
                
                # Update progress bar with current loss
                train_pbar.set_postfix({
                    'loss': f'{train_loss_tensor.item():.4f}'
                })

            # validation phase
            self.model.eval()
            
            # Progress bar for validation batches
            val_pbar = tqdm(
                self.val_loader,
                desc=f"Validation Epoch {epoch + 1}/{epochs}",
                unit="batch",
                leave=False
            )
            
            with torch.no_grad():
                for data, target in val_pbar:
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
                    
                    # Update progress bar with current loss
                    val_pbar.set_postfix({
                        'loss': f'{val_loss_tensor.item():.4f}'
                    })

            # compute average losses
            train_loss: float = float(np.average(train_losses))
            val_loss: float = float(np.average(val_losses))
            avg_train_loss.append(train_loss)
            avg_val_loss.append(val_loss)

            # Calculate epoch duration and estimate remaining time
            epoch_duration = time.time() - epoch_start_time
            elapsed_time = time.time() - training_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            eta = remaining_epochs * avg_epoch_time

            # log losses and timing information
            self.logger.info(
                "Train loss: %s, Val loss: %s | "
                "Epoch time: %.1fs | "
                "Elapsed: %s | "
                "ETA: %s",
                train_loss,
                val_loss,
                epoch_duration,
                self._format_time(elapsed_time),
                self._format_time(eta)
            )

            train_losses = []
            val_losses = []

            # early stopping
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                self.logger.info("Early stopping!")
                break

        if self.config["memory_record"]["enabled"]:
            export_memory_snapshot(self.logger)
            stop_record_memory_history(self.logger)

        return self.model, avg_train_loss, avg_val_loss

    def evaluate(self) -> None:
        """Evaluate the model on test data.

        TODO: Implement evaluation logic
        """
        # return avg_loss, accuracy
        return
