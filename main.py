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

    args = parse_args()
    config = load_config(args.config)

    device = get_device(config["device"])
    val_ratio = config["data"]["val_ratio"]
    paths = config["data"]["paths"]

    print(paths)

    amazon_dataset = AmazonDataset(paths)
    train_data, val_data = random_split(amazon_dataset, [(1 - val_ratio), val_ratio])

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

    # model = Model()

    # optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])


if __name__ == "__main__":
    main()
