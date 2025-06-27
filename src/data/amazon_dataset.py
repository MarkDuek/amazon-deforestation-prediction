import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing_extensions import Callable


class AmazonDataset(Dataset):

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
            f"Initializing AmazonDataset with data paths:\n{paths_str}"
        )

        self.data = self.load_npz_files(self.data_paths)

        self.time_slice = self.config["time_slice"]
        self.total_time_steps = len(self.data[0].keys())
        self.transform = transform

    def __len__(self) -> int:
        return self.total_time_steps - self.time_slice + 1

    def __getitem__(self, time_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.logger.debug(f"Getting item at index: {time_idx}")
        time_slice_data = self.get_time_slice(time_idx, self.time_slice)
        concatenated_data = self.concatenate_sparse_matrix(time_slice_data)

        input_data = torch.tensor(concatenated_data)  # (C, T, H, W)
        target = torch.tensor(
            concatenated_data[0, -1, :, :]
        )  # (H, W) - last time step of first channel
        input_data = self.pad_to_multiple(
            input_data, self.config["padding_multiple"]
        )
        target = self.pad_to_multiple(target, self.config["padding_multiple"])

        # TODO: Add transform

        return input_data, target

    def get_time_slice(
        self,
        time_idx: int,
        time_slice: int,
    ) -> List[Mapping[str, Any]]:

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
                    self.logger.warning(f"Key '{key}' not found in file data")

            file_channels.append(subset_data)

        return file_channels

    def load_npz_files(
        self,
        paths: List[str],
    ) -> List[np.lib.npyio.NpzFile]:

        return [np.load(path, allow_pickle=True) for path in paths]

    def stack_sparse_matrix(
        self,
        data: Mapping[str, Any],
        d_type: Any = np.float32,
    ) -> np.ndarray:

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

        arrays: List[np.ndarray] = [
            self.stack_sparse_matrix(data, d_type=np.float32)
            for data in data_list
        ]

        return np.stack(arrays)

    def pad_to_multiple(
        self, tensor: torch.Tensor, multiple: int
    ) -> torch.Tensor:
        self.logger.debug(
            f"Padding tensor with shape {tensor.shape} to multiple {multiple}"
        )

        *batch_dims, h, w = tensor.shape
        target_h = ((h + multiple - 1) // multiple) * multiple
        target_w = ((w + multiple - 1) // multiple) * multiple
        self.logger.debug(f"Target shape: {target_h}x{target_w}")

        pad_h, pad_w = target_h - h, target_w - w
        padding = (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        )
        self.logger.debug(f"Padding: {padding}")

        return F.pad(tensor, padding, mode="constant", value=0.0)
