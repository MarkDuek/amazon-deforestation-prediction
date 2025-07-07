"""Training utilities for Amazon deforestation prediction model."""

import json
import logging
import os
from typing import Any, Dict, List



def plot_training_curves(
    train_losses: List[float], 
    val_losses: List[float], 
    save_path: str
) -> None:
    """Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path where to save the plot
    """
    logger = logging.getLogger(__name__)
    
    try:
        import matplotlib.pyplot as plt
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot loss difference (overfitting indicator)
        plt.subplot(1, 2, 2)
        loss_diff = [t - v for t, v in zip(train_losses, val_losses)]
        plt.plot(loss_diff, label='Train - Val Loss')
        plt.title('Loss Difference (Overfitting Indicator)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info("Training curves saved to: %s", save_path)
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")


def save_training_history(
    train_losses: List[float],
    val_losses: List[float],
    config: Dict[str, Any],
    save_path: str
) -> Dict[str, Any]:
    """Save training history to JSON file.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        config: Configuration dictionary
        save_path: Path where to save the history
        
    Returns:
        Dictionary containing the training history
    """
    logger = logging.getLogger(__name__)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create training history dictionary
    training_history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "num_epochs": len(train_losses),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "config": config
    }
    
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info("Training history saved to: %s", save_path)
    
    return training_history


def save_evaluation_results(
    val_metrics: Dict[str, float],
    train_metrics: Dict[str, float],
    training_history: Dict[str, Any],
    save_path: str
) -> None:
    """Save evaluation results to JSON file.
    
    Args:
        val_metrics: Validation metrics dictionary
        train_metrics: Training metrics dictionary
        training_history: Training history dictionary
        save_path: Path where to save the results
    """
    logger = logging.getLogger(__name__)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create evaluation results dictionary
    evaluation_results = {
        "validation_metrics": val_metrics,
        "training_metrics": train_metrics,
        "training_history": training_history
    }
    
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info("Evaluation results saved to: %s", save_path)


def create_directories(config: Dict[str, Any]) -> None:
    """Create all necessary directories for saving outputs.
    
    Args:
        config: Configuration dictionary containing save paths
    """
    if "save_paths" not in config:
        return
        
    for path_key, path_value in config["save_paths"].items():
        if path_value:
            os.makedirs(os.path.dirname(path_value), exist_ok=True)


def log_training_summary(
    train_losses: List[float],
    val_losses: List[float],
    val_metrics: Dict[str, float],
    train_metrics: Dict[str, float]
) -> None:
    """Log a comprehensive training summary.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_metrics: Validation metrics dictionary
        train_metrics: Training metrics dictionary
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    # Training progress
    logger.info("Training Progress:")
    logger.info("  Total Epochs: %d", len(train_losses))
    if train_losses:
        logger.info("  Initial Train Loss: %.4f", train_losses[0])
        logger.info("  Final Train Loss: %.4f", train_losses[-1])
        logger.info("  Train Loss Reduction: %.4f", 
                    train_losses[0] - train_losses[-1])
    
    if val_losses:
        logger.info("  Initial Val Loss: %.4f", val_losses[0])
        logger.info("  Final Val Loss: %.4f", val_losses[-1])
        logger.info("  Val Loss Reduction: %.4f", 
                    val_losses[0] - val_losses[-1])
    
    # Final performance
    logger.info("\nFinal Model Performance:")
    logger.info("Validation Set:")
    logger.info("  Loss: %.4f | IoU: %.4f | F1: %.4f | Accuracy: %.4f", 
                val_metrics["avg_loss"], val_metrics["iou"], 
                val_metrics["f1"], val_metrics["accuracy"])
    
    logger.info("Training Set:")
    logger.info("  Loss: %.4f | IoU: %.4f | F1: %.4f | Accuracy: %.4f", 
                train_metrics["avg_loss"], train_metrics["iou"], 
                train_metrics["f1"], train_metrics["accuracy"])
    
    # Overfitting analysis
    logger.info("\nOverfitting Analysis:")
    metrics_to_check = ['accuracy', 'f1', 'iou', 'dice']
    for metric in metrics_to_check:
        if metric in train_metrics and metric in val_metrics:
            diff = train_metrics[metric] - val_metrics[metric]
            if diff > 0.1:
                status = "⚠️  High overfitting"
            elif diff < 0.05:
                status = "✅ Low overfitting"
            else:
                status = "⚠️  Moderate overfitting"
            logger.info("  %s difference: %+.4f (%s)", 
                        metric.capitalize(), diff, status)
    
    logger.info("="*60) 