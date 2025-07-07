"""Amazon deforestation dataset implementation for PyTorch."""

import logging
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import Callable


class AmazonDataset(Dataset):
    """PyTorch Dataset for Amazon deforestation prediction.

    This dataset loads and processes temporal pre-processed satellite imagery
    data for Amazon rainforest deforestation prediction tasks.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        transform: Optional[Callable] = None,
    ):
        self.config = config["data"]
        self.data_paths = self.config["paths"]
        self.time_slice = self.config["time_slice"]

        self.logger = logging.getLogger(__name__)
        paths_str = "\n".join([f"  - {path}" for path in self.data_paths])
        self.logger.info(
            "Initializing AmazonDataset with data paths:\n%s", paths_str
        )

        self.input_h5 = h5py.File(self.config["h5_paths"]["input"], "r")
        self.target_h5 = h5py.File(self.config["h5_paths"]["target"], "r")

        self.time_indices = self.input_h5["time_indices"][:]
        self.num_patches_per_time = np.sum(self.time_indices == 0)
        expected_len = self.num_patches_per_time * (
            self.time_indices[-1] - self.time_slice + 1
        )

        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.num_patches_per_time * (
            self.time_indices[-1] - self.time_slice + 1
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: Index for the sample

        Returns:
            Tuple of (input_data, target) tensors
        """

        # Compute sequence indices
        seq_indices = [
            idx + i * self.num_patches_per_time for i in range(self.time_slice)
        ]

        # Check bounds against actual HDF5 dataset size
        h5_dataset_size = self.input_h5["patches"].shape[0]  # type: ignore
        if seq_indices[-1] >= h5_dataset_size:
            raise IndexError(
                f"Index {idx} with time_slice={self.time_slice} "
                f"exceeds HDF5 dataset size {h5_dataset_size}"
            )

        # Load input sequence patches
        sequence = np.stack(
            [self.input_h5["patches"][i] for i in seq_indices]
        )  # type: ignore  # shape: (time_slice, C, h, w)

        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        # Shape: (time_slice-1, C, H, W)
        input_tensor = sequence_tensor[:-1, :, :, :]
        # Transpose to (C, time_slice-1, H, W) to match model expectation
        input_tensor = input_tensor.transpose(0, 1)  # (C, time_slice-1, H, W)

        target_tensor = sequence_tensor[-1, 1, :, :].unsqueeze(0).unsqueeze(0)

        if self.transform:
            # (C, time_slice-1, H, W)
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor
