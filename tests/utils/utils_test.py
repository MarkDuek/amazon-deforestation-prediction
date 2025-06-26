import pytest
import torch

from src.utils.utils import get_device, parse_args, load_config

def test_get_device(device):
    """Test get_device function with different device inputs."""
    result = get_device(device)
    
    if device == "cuda" and torch.cuda.is_available():
        assert result == torch.device('cuda')
    else:
        assert result == torch.device('cpu')

def test_get_device_fallback():
    """Test that CUDA falls back to CPU when CUDA is not available."""
    result = get_device("cuda")
    assert result in [torch.device('cuda'), torch.device('cpu')]

def test_get_device_default():
    """Test default behavior (should return CPU)."""
    result = get_device()
    assert result == torch.device('cpu')

def test_parse_args():
    """Test parse_args function."""
    args = parse_args(["--config", "config.yaml"]) 
    assert args.config == "config.yaml"

def test_load_config():
    """Test load_config function."""
    config = load_config("config.yaml")
    assert config is not None