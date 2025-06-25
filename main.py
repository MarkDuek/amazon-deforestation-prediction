import logging
import json
import torch
from torch.utils.data import random_split, DataLoader

from src.data.amazon_dataset import AmazonDataset
from src.utils.utils import get_device, parse_args, load_config


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    args = parse_args()
    config = load_config(args.config)
    logger.info(f"Loaded config:\n{json.dumps(config, indent=2)}")

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

    # TODO: define loss function
    # loss_fn = 

    # trainer = Trainer(model, train_loader, val_loader, optimizer, loss_fn, device)
    # trainer.train()

if __name__ == "__main__":
    main()
