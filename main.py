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
from src.models.deep_lab_v3.model import DeepLabV3
# from src.models.dummy.model import DummyModel
from src.training.trainer import Trainer
from src.utils.data_utils import inspect_h5_file
from src.utils.utils import get_device, load_config, parse_args


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

    # load dataset
    amazon_dataset = AmazonDataset(config)
    logger.info("Dataset input shape: %s", amazon_dataset[0][0].shape)
    logger.info("Dataset target shape: %s", amazon_dataset[0][1].shape)

    # split dataset
    train_data, val_data = random_split(
        amazon_dataset, [(1 - val_ratio), val_ratio]
    )

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

    # define model
    model = DeepLabV3(config)
    # model = DummyModel(config)
    logger.info("Model summary:")
    summary(model, input_size=(5, 7, 16, 16))

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    logger.info("Optimizer summary: %s", optimizer)

    # define loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logger.info("Loss function summary: %s", loss_fn)

    # define trainer
    trainer = Trainer(
        model, train_loader, val_loader, optimizer, loss_fn, device, config
    )

    # start training
    logger.info("%s Starting training %s", 16 * "=", 16 * "=")
    trainer.train()
    logger.info("%s Training completed %s", 16 * "=", 16 * "=")


if __name__ == "__main__":
    main()
