import pytest
import torch

from src.models.deep_lab_v3.model import DeepLabV3


def test_forward(config, input_tensor):
    model = DeepLabV3(config)
    output = model.forward(input_tensor)
    assert output.shape == (
        input_tensor.shape[0],
        config["model"]["deep_lab_v3"]["classes"],
        1,
        input_tensor.shape[3],
        input_tensor.shape[4],
    )  # (B, 1, 1, H, W)
