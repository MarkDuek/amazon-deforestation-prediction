import logging
import argparse
import yaml
import torch
from torch.utils.data import random_split, DataLoader

from src.data.amazon_dataset import AmazonDataset
from src.utils.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Project")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # define logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # parse args
    args = parse_args()
    config = load_config(args.config)

    # get config variables
    device = get_device(config["device"])
    val_ratio = config["data"]["val_ratio"]
    paths = config["data"]["paths"]
    time_slice = config["data"]["time_slice"]

    # load dataset
    amazon_dataset = AmazonDataset(paths, time_slice)
    logger.info(f"Dataset sample shape: {amazon_dataset[0].shape}")

    # split dataset
    train_data, val_data = random_split(amazon_dataset, [(1 - val_ratio), val_ratio])

    # create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # TODO: define model

    # model = Model()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # TODO:
    # trainer = Trainer(config)


if __name__ == "__main__":
    main()
