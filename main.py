import json
import logging

import torch
from torch.utils.data import DataLoader, random_split

from src.data.amazon_dataset import AmazonDataset
from src.models.deep_lab_v3.model import DeepLabV3
from src.training.trainer import Trainer
from src.utils.utils import get_device, load_config, parse_args


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()
    config = load_config(args.config)
    logger.info(f"Loaded config:\n{json.dumps(config, indent=2)}")

    # get config variables
    device = get_device(config["device"])
    val_ratio = config["data"]["val_ratio"]

    # load dataset
    amazon_dataset = AmazonDataset(config)
    logger.info(f"Dataset sample shape: {amazon_dataset[0][0].shape}")

    # split dataset
    train_data, val_data = random_split(amazon_dataset, [(1 - val_ratio), val_ratio])

    # create dataloaders
    logger.info(f"Creating train loader")
    train_loader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    logger.info(f"Creating val loader")
    val_loader = DataLoader(
        val_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    # define model
    model = DeepLabV3(config)
    model.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    logger.info(f"Optimizer summary: {optimizer}")

    # define loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logger.info(f"Loss function summary: {loss_fn}")

    # define trainer
    trainer = Trainer(
        model, train_loader, val_loader, optimizer, loss_fn, device, config
    )

    # start training
    logger.info(f"Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
