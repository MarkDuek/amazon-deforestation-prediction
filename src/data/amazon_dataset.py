"""Amazon deforestation dataset implementation for PyTorch."""

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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

        self.logger = logging.getLogger(__name__)
        paths_str = "\n".join([f"  - {path}" for path in self.data_paths])
        self.logger.info(
            "Initializing AmazonDataset with data paths:\n%s", paths_str
        )

        self.data = self.load_npz_files(self.data_paths)

        self.time_slice = self.config["time_slice"]
        self.total_time_steps = len(self.data[0].keys())
        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.total_time_steps - self.time_slice + 1

    def __getitem__(self, time_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            time_idx: Time index for the sample

        Returns:
            Tuple of (input_data, target) tensors
        """
        self.logger.debug("Getting item at index: %s", time_idx)
        time_slice_data = self.get_time_slice(time_idx, self.time_slice)
        concatenated_data = self.concatenate_sparse_matrix(time_slice_data)

        # (C, T-1, H, W) - first T-1 time slices
        input_data = torch.tensor(concatenated_data[:, :-1, :, :])
        # (1, 1, H, W) - last time step of second channel
        target = (
            torch.tensor(concatenated_data[1, -1, :, :])
            .unsqueeze(0)
            .unsqueeze(0)
        )

        input_data = self.pad_to_multiple(
            input_data, self.config["padding_multiple"]
        )
        target = self.pad_to_multiple(target, self.config["padding_multiple"])

        if self.transform is not None:
            input_data = self.transform(input_data)

        return input_data, target

    def get_time_slice(
        self,
        time_idx: int,
        time_slice: int,
    ) -> List[Mapping[str, Any]]:
        """Extract a time slice from the dataset.

        Args:
            time_idx: Starting time index
            time_slice: Length of the time slice

        Returns:
            List of mappings containing the time slice data
        """
        file_channels: List[Mapping[str, Any]] = []

        for file_data in self.data:
            subset_data: Dict[str, Any] = {}
            for i, time_step in enumerate(
                range(time_idx, time_idx + time_slice)
            ):
                key = f"arr_{time_step}"
                if key in file_data:
                    subset_data[f"arr_{i}"] = file_data[key]
                else:
                    self.logger.warning("Key '%s' not found in file data", key)

            file_channels.append(subset_data)

        return file_channels

    def load_npz_files(
        self,
        paths: List[str],
    ) -> List[np.lib.npyio.NpzFile]:
        """Load NPZ files from the given paths.

        Args:
            paths: List of file paths to load

        Returns:
            List of loaded NPZ files
        """
        return [np.load(path, allow_pickle=True) for path in paths]

    def stack_sparse_matrix(
        self,
        data: Mapping[str, Any],
        d_type: Any = np.float32,
    ) -> np.ndarray:
        """Stack sparse matrices into a 3D array.

        Args:
            data: Mapping containing sparse matrix data
            d_type: Data type for the output array

        Returns:
            3D numpy array of stacked sparse matrices
        """
        n_frames = len(data)
        frames: List[np.ndarray] = []
        for i in range(n_frames):
            obj = data[f"arr_{i}"]
            sparse_matrix = obj.item()
            arr: np.ndarray = sparse_matrix.toarray()
            frames.append(arr)

        array_3d: np.ndarray = np.stack(frames).astype(d_type)

        return array_3d

    def concatenate_sparse_matrix(
        self,
        data_list: List[Mapping[str, Any]],
    ) -> np.ndarray:
        """Concatenate sparse matrices from multiple data sources.

        Args:
            data_list: List of data mappings to concatenate

        Returns:
            Concatenated numpy array
        """
        arrays: List[np.ndarray] = [
            self.stack_sparse_matrix(data, d_type=np.float32)
            for data in data_list
        ]

        return np.stack(arrays)

    def pad_to_multiple(
        self, tensor: torch.Tensor, multiple: int
    ) -> torch.Tensor:
        """Pad tensor dimensions to be multiples of a given value.

        Args:
            tensor: Input tensor to pad
            multiple: Value to pad dimensions to multiples of

        Returns:
            Padded tensor
        """
        self.logger.debug(
            "Padding tensor with shape %s to multiple %s",
            tensor.shape,
            multiple,
        )

        *_, h, w = tensor.shape
        target_h = ((h + multiple - 1) // multiple) * multiple
        target_w = ((w + multiple - 1) // multiple) * multiple
        self.logger.debug("Target shape: %sx%s", target_h, target_w)

        pad_h, pad_w = target_h - h, target_w - w
        padding = (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        )
        self.logger.debug("Padding: %s", padding)

        return F.pad(tensor, padding, mode="constant", value=0.0)
