"""Training module for Amazon deforestation prediction model."""

import logging
import time
from typing import Any, Dict, List, Tuple
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex

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
        self.threshold = config["metrics"]["threshold"]

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
            verbose=self.config["early_stopping"]["verbose"],
            config=self.config,
        )

        # move model to device
        self.model.to(self.device)
        self.logger.info("Model moved to device: %s", self.device)

        accuracy = Accuracy(task="binary").to(self.device)
        precision = Precision(task="binary").to(self.device)
        recall = Recall(task="binary").to(self.device)
        f1 = F1Score(task="binary").to(self.device)
        iou = JaccardIndex(task="binary").to(self.device)

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
                time.strftime("%H:%M:%S", time.localtime(epoch_start_time)),
            )

            # training phase
            self.model.train()

            # Progress bar for training batches
            train_pbar = tqdm(
                self.train_loader,
                desc=f"Training Epoch {epoch + 1}/{epochs}",
                unit="batch",
                leave=False,
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

                # compute metrics
                pred = (train_output > self.threshold).int()
                targ = (target > self.threshold).int()

                accuracy(pred, targ)
                precision(pred, targ)
                recall(pred, targ)
                f1(pred, targ)
                iou(pred, targ)

                # Update progress bar with current loss
                train_pbar.set_postfix(
                    {"loss": f"{train_loss_tensor.item():.4f}",
                     "accuracy": f"{accuracy.compute():.4f}",
                     "precision": f"{precision.compute():.4f}",
                     "recall": f"{recall.compute():.4f}",
                     "f1": f"{f1.compute():.4f}",
                     "iou": f"{iou.compute():.4f}"}
                )
            
            accuracy.reset()
            precision.reset()
            recall.reset()
            f1.reset()
            iou.reset()

            # validation phase
            self.model.eval()

            # Progress bar for validation batches
            val_pbar = tqdm(
                self.val_loader,
                desc=f"Validation Epoch {epoch + 1}/{epochs}",
                unit="batch",
                leave=False,
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

                    # compute metrics
                    pred = (val_output > self.threshold).int()
                    targ = (target > self.threshold).int()

                    accuracy(pred, targ)
                    precision(pred, targ)
                    recall(pred, targ)
                    f1(pred, targ)
                    iou(pred, targ)

                    # Update progress bar with current loss
                    val_pbar.set_postfix(
                        {"loss": f"{val_loss_tensor.item():.4f}",
                         "accuracy": f"{accuracy.compute():.4f}",
                         "precision": f"{precision.compute():.4f}",
                         "recall": f"{recall.compute():.4f}",
                         "f1": f"{f1.compute():.4f}",
                         "iou": f"{iou.compute():.4f}"}
                    )

            accuracy.reset()
            precision.reset()
            recall.reset()
            f1.reset()
            iou.reset()

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
                self._format_time(eta),
            )

            train_losses = []
            val_losses = []

            # early stopping
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                self.logger.info("Early stopping!")
                break

            # save model
            torch.save(self.model.state_dict(), self.config["save_paths"]["model"])

        if self.config["memory_record"]["enabled"]:
            export_memory_snapshot(self.logger)
            stop_record_memory_history(self.logger)

        return self.model, avg_train_loss, avg_val_loss

    def evaluate(
        self, 
        trained_model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Evaluate the model on the given data loader.

        Args:
            data_loader: Data loader for evaluation

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Starting model evaluation...")
        self.model = trained_model

        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar for evaluation
        eval_pbar = tqdm(
            data_loader,
            desc="Evaluating",
            unit="batch",
            leave=False,
        )
        
        accuracy = Accuracy(task="binary").to(self.device)
        precision = Precision(task="binary").to(self.device)
        recall = Recall(task="binary").to(self.device)
        f1 = F1Score(task="binary").to(self.device)
        iou = JaccardIndex(task="binary").to(self.device)

        with torch.no_grad():
            for data, target in eval_pbar:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                loss = self.loss_fn(output, target)
                total_loss += loss.item()

                # compute metrics
                pred = (output > self.threshold).int()
                targ = (target > self.threshold).int()
                
                accuracy(pred, targ)
                precision(pred, targ)
                recall(pred, targ)
                f1(pred, targ)
                iou(pred, targ)
                
                # Update progress bar
                eval_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "accuracy": f"{accuracy.compute():.4f}",
                    "precision": f"{precision.compute():.4f}",
                    "recall": f"{recall.compute():.4f}",
                    "f1": f"{f1.compute():.4f}",
                    "iou": f"{iou.compute():.4f}"
                })
                
                num_batches += 1
        
        # Compute average metrics
        avg_metrics = {}
        avg_metrics["accuracy"] = accuracy.compute()
        avg_metrics["precision"] = precision.compute()
        avg_metrics["recall"] = recall.compute()
        avg_metrics["f1"] = f1.compute()
        avg_metrics["iou"] = iou.compute()
        avg_metrics["avg_loss"] = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Log results
        self.logger.info("Evaluation Results:")
        self.logger.info("  Average Loss: %.4f", avg_metrics["avg_loss"])
        self.logger.info("  Accuracy: %.4f", avg_metrics["accuracy"])
        self.logger.info("  Precision: %.4f", avg_metrics["precision"])
        self.logger.info("  Recall: %.4f", avg_metrics["recall"])
        self.logger.info("  F1 Score: %.4f", avg_metrics["f1"])
        self.logger.info("  IoU: %.4f", avg_metrics["iou"])
        
        return avg_metrics
