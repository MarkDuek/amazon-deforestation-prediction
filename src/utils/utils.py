import argparse

import torch
import yaml


def get_device(device_str: str = "cpu") -> torch.device:
    """
    Returns a torch.device object based on the input string.
    If device_str is 'cuda' and CUDA is not available, falls back to CPU.
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def parse_args(args=None) -> argparse.Namespace:
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep Learning Project")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    return parser.parse_args(args)


def load_config(path: str) -> dict:
    """
    Loads a configuration file from the given path.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
