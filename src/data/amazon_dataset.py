import logging
import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Any, Mapping, List, Optional, Tuple
from typing_extensions import Callable


class AmazonDataset(Dataset):

    def __init__(
        self,
        data_paths: List[str],
        time_slice: int,
        transform: Optional[Callable] = None
    ):
        self.logger = logging.getLogger(__name__)
        paths_str = '\n'.join([f"  - {path}" for path in data_paths])
        self.logger.info(f"Initializing AmazonDataset with data paths:\n{paths_str}")
        self.data_paths = data_paths
        self.transform = transform
        self.data = self.load_npz_files(data_paths)
        self.time_slice = time_slice
        self.total_time_steps = len(self.data[0].keys())

    def __len__(self) -> int:
        return self.total_time_steps - self.time_slice + 1

    def __getitem__(
        self,
        time_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.logger.debug(f"Getting item at index: {time_idx}")
        time_slice_data = self.get_time_slice(time_idx, self.time_slice)
        time_slice_data = self.concatenate_sparse_matrix(time_slice_data)
        
        input_data = torch.tensor(time_slice_data)  # (C, T, H, W)
        target = torch.tensor(time_slice_data[0, -1, :, :])  # (H, W) - last time step of first channel
        
        return input_data, target
    
    def get_time_slice(
        self,
        time_idx: int,
        time_slice: int,
    ) -> List[Mapping[str, Any]]:

        file_channels: List[Mapping[str, Any]] = []
        
        for file_data in self.data:
            subset_data: Mapping[str, Any] = {}
            for i, time_step in enumerate(range(time_idx, time_idx + time_slice)):
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
        frames: list[np.ndarray] = []
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

        arrays: list[np.ndarray] = [
            self.stack_sparse_matrix(data, d_type=np.float32) for data in data_list
        ]

        return np.stack(arrays)
