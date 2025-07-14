"""Main module for Amazon deforestation prediction training pipeline.

This module provides the main entry point for training deep learning models
on the Amazon deforestation dataset.
"""

import json
import logging

import torch
from torch.utils.data import DataLoader, random_split
from torchinfo import summary

from src.data.amazon_dataset import AmazonDataset
# from src.models.deep_lab_v3.model import DeepLabV3
from src.loss.weightedBCE import WeightedBCEWithLogitsLoss
from src.loss.bce_plus_iou import WeightedBCEWithLogitsAndIoULoss
from src.models.simple_cnn.model import Tiny3DUNet
from src.models.simple_cnn.baseline import SimpleBaselineB
# from src.models.dummy.model import DummyModel
from src.training.trainer import Trainer
from src.utils.data_utils import inspect_h5_file
from src.utils.utils import get_device, load_config, parse_args
from src.utils.training_utils import plot_training_curves


def valid_pixels(loader: DataLoader) -> float:
    """Compute the number of valid pixels in the target tensor.

    Args:
        target: Target tensor of shape (batch_size, ...).

    Returns:
        Number of valid pixels in the target tensor.
    """
    valid_pixels = 0
    for batch in loader:
        target = batch[1]
        valid_pixels += (target > 0.0).float().sum().item()
    
    return valid_pixels / len(loader)


def main():
    """Main function to run the Amazon deforestation
    prediction training pipeline."""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()
    config = load_config(args.config)
    logger.info("Loaded config:\n%s", json.dumps(config, indent=2))

    # get config variables
    device = get_device(config["device"])
    val_ratio = config["data"]["val_ratio"]

    # inspect h5 files
    inspect_h5_file(config["data"]["h5_paths"]["input"])
    inspect_h5_file(config["data"]["h5_paths"]["target"])

    # define model
    # model = DeepLabV3(config)
    # model = DummyModel(config)
    # model = SimpleBaselineB(5, 6)
    model = Tiny3DUNet()

    logger.info("Model summary:")
    summary(model, input_size=(2, 5, 7, 32, 32))

    # load dataset
    amazon_dataset = AmazonDataset(config)
    logger.info("Dataset input shape: %s", amazon_dataset[0][0].shape)
    logger.info("Dataset target shape: %s", amazon_dataset[0][1].shape)

    # split dataset
    train_data, val_data = random_split(
        amazon_dataset, [(1 - val_ratio), val_ratio]
    )

    # Before creating your DataLoader:
    # labels = []
    # for idx in range(len(train_data)):
    #     _, mask = train_data[idx]
    #     labels.append(int(mask.sum() > 0))  # 1 if any positives, else 0

    # # Compute class weights: rarer class gets higher weight
    # neg_count = labels.count(0)
    # pos_count = labels.count(1)

    # if neg_count == 0 or pos_count == 0:
    #     raise ValueError(
    #         f"Training split contains only one class "
    #         f"(neg_count={neg_count}, pos_count={pos_count}). "
    #         "Try a smaller val_ratio or stratified splitting."
    #     )
    
    # class_weights = [1.0 / neg_count, 1.0 / pos_count]

    # # Make a weight for each sample
    # sample_weights = [class_weights[label] for label in labels]

    # from torch.utils.data import WeightedRandomSampler
    # sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # create dataloaders
    logger.info("Creating Train loader")
    train_loader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,  # Use multiple workers for faster data loading
        pin_memory=True,  # Pin memory for faster GPU transfers
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch batches
    )
    logger.info("Creating Validation loader")
    val_loader = DataLoader(
        val_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,  # Use multiple workers for faster data loading
        pin_memory=True,  # Pin memory for faster GPU transfers
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch batches
    )

    # compute valid pixels
    # avg_valid_pixels = valid_pixels(train_loader)
    # logger.info("Average valid pixels: %s", avg_valid_pixels)


    # define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )
    logger.info("Optimizer summary: %s", optimizer)

    # define loss function
    loss_fn = WeightedBCEWithLogitsAndIoULoss(config)
    logger.info("Loss function summary: %s", loss_fn)

    # define trainer
    trainer = Trainer(
        model, train_loader, val_loader, optimizer, loss_fn, device, config
    )

    # start training
    logger.info("%s Starting training %s", 16 * "=", 16 * "=")
    trained_model, train_losses, val_losses = trainer.train()
    logger.info("%s Training completed %s", 16 * "=", 16 * "=")

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, config["save_paths"]["plot"]
    )

    # # Save training history
    # training_history = {}
    # if "save_paths" in config and "history" in config["save_paths"]:
    #     training_history = save_training_history(
    #         train_losses, val_losses, config, config["save_paths"]["history"]
    #     )

    # evaluate model
    logger.info("%s Starting evaluation %s", 16 * "=", 16 * "=")
    
    # Evaluate on validation set
    _ = trainer.evaluate(trained_model, val_loader)
    logger.info("Validation set evaluation completed.")
    
    # # Optionally evaluate on training set for comparison
    # train_metrics = trainer.evaluate(train_loader)
    # logger.info("Training set evaluation completed.")
    
    # Save evaluation results
    # save_evaluation_results(
    #     val_metrics, train_metrics, training_history, 
    #     config["save_paths"]["results"]
    # )
    
    # Log comprehensive training summary
    # log_training_summary(train_losses, val_losses, val_metrics, train_metrics)
    
    logger.info("%s Evaluation completed %s", 16 * "=", 16 * "=")


if __name__ == "__main__":
    main()
