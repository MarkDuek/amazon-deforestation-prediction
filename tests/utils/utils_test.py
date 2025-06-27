import pytest
import torch

from src.utils.utils import get_device, load_config, parse_args


def test_get_device(device):
    result = get_device(device)

    if device == "cuda" and torch.cuda.is_available():
        assert result == torch.device("cuda")
    else:
        assert result == torch.device("cpu")


def test_get_device_fallback():
    result = get_device("cuda")
    assert result in [torch.device("cuda"), torch.device("cpu")]


def test_get_device_default():
    result = get_device()
    assert result == torch.device("cpu")


def test_parse_args():
    args = parse_args(["--config", "config.yaml"])
    assert args.config == "config.yaml"


def test_load_config():
    config = load_config("config.yaml")
    assert config is not None
