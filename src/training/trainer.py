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

    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor, 
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute evaluation metrics for binary segmentation.

        Args:
            predictions: Model predictions (logits) of shape (B, 1, 1, H, W)
            targets: Ground truth targets of shape (B, 1, 1, H, W)
            threshold: Threshold for binary classification

        Returns:
            Dictionary containing computed metrics
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(predictions)
        
        # Apply threshold to get binary predictions
        pred_binary = (probs > threshold).float()
        
        # Create validity mask (same logic as loss function)
        validity_mask = (targets > 0.0).float()
        
        # Only compute metrics on valid pixels
        valid_preds = pred_binary * validity_mask
        valid_targets = targets * validity_mask
        
        # Flatten tensors for computation
        valid_preds_flat = valid_preds.view(-1)
        valid_targets_flat = valid_targets.view(-1)
        validity_mask_flat = validity_mask.view(-1)
        
        # Only consider valid pixels
        valid_indices = validity_mask_flat > 0
        if valid_indices.sum() == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "iou": 0.0,
                "dice": 0.0,
                "valid_pixels": 0
            }
        
        pred_valid = valid_preds_flat[valid_indices]
        target_valid = valid_targets_flat[valid_indices]
        
        # Compute confusion matrix elements
        tp = (pred_valid * target_valid).sum().item()
        fp = (pred_valid * (1 - target_valid)).sum().item()
        fn = ((1 - pred_valid) * target_valid).sum().item()
        tn = ((1 - pred_valid) * (1 - target_valid)).sum().item()
        
        # Compute metrics
        total_pixels = tp + fp + fn + tn
        accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precision_recall_sum = precision + recall
        f1 = (2 * precision * recall / precision_recall_sum 
              if precision_recall_sum > 0 else 0.0)
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice_denom = 2 * tp + fp + fn
        dice = 2 * tp / dice_denom if dice_denom > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "dice": dice,
            "valid_pixels": valid_indices.sum().item()
        }

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

                # Update progress bar with current loss
                train_pbar.set_postfix(
                    {"loss": f"{train_loss_tensor.item():.4f}"}
                )

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

                    # Update progress bar with current loss
                    val_pbar.set_postfix(
                        {"loss": f"{val_loss_tensor.item():.4f}"}
                    )

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
        data_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Evaluate the model on the given data loader.

        Args:
            data_loader: Data loader for evaluation
            threshold: Threshold for binary classification

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Starting model evaluation...")

        threshold: float = self.config["metrics"]["threshold"]

        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics storage
        all_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "iou": [],
            "dice": [],
            "valid_pixels": []
        }
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar for evaluation
        eval_pbar = tqdm(
            data_loader,
            desc="Evaluating",
            unit="batch",
            leave=False,
        )
        
        with torch.no_grad():
            for data, target in eval_pbar:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                
                # Compute metrics for this batch
                batch_metrics = self._compute_metrics(
                    output, target, threshold
                )
                
                # Store metrics
                for key, value in batch_metrics.items():
                    all_metrics[key].append(value)
                
                # Update progress bar
                eval_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "iou": f"{batch_metrics['iou']:.4f}",
                    "f1": f"{batch_metrics['f1']:.4f}"
                })
                
                num_batches += 1
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in all_metrics.items():
            if key == "valid_pixels":
                avg_metrics[key] = sum(values)  # Total count
            else:
                # Weighted average based on valid pixels
                if all_metrics["valid_pixels"]:
                    weights = np.array(all_metrics["valid_pixels"])
                    values_array = np.array(values)
                    avg_metrics[key] = np.average(
                        values_array, weights=weights
                    )
                else:
                    avg_metrics[key] = 0.0
        
        avg_metrics["avg_loss"] = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Log results
        self.logger.info("Evaluation Results:")
        self.logger.info("  Average Loss: %.4f", avg_metrics["avg_loss"])
        self.logger.info("  Accuracy: %.4f", avg_metrics["accuracy"])
        self.logger.info("  Precision: %.4f", avg_metrics["precision"])
        self.logger.info("  Recall: %.4f", avg_metrics["recall"])
        self.logger.info("  F1 Score: %.4f", avg_metrics["f1"])
        self.logger.info("  IoU: %.4f", avg_metrics["iou"])
        self.logger.info("  Dice Coefficient: %.4f", avg_metrics["dice"])
        self.logger.info("  Total Valid Pixels: %d", avg_metrics["valid_pixels"])
        
        return avg_metrics
