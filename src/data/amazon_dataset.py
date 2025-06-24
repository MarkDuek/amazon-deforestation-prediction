import l"""  """ogging
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Mapping, List


class AmazonDataset(Dataset):

    def __init__(self, data_paths, time_slice, transform=None):
        self.logger = logging.getLogger(__name__)
        paths_str = '\n'.join([f"  - {path}" for path in data_paths])
        self.logger.info(f"Initializing AmazonDataset with data paths:\n{paths_str}")
        self.data_paths = data_paths
        self.transform = transform
        self.data = self.load_npz_files(data_paths)
        self.time_slice = time_slice
        self.total_time_steps = len(self.data[0].keys())

    def __len__(self):
        return self.total_time_steps - self.time_slice + 1

    def __getitem__(self, idx: int):
        self.logger.info(f"Getting item at index: {idx}")
        time_slice_data = self.get_time_slice(idx, self.time_slice)
        time_slice_data = self.concatenate_sparse_matrix(time_slice_data)
        
        return torch.tensor(time_slice_data)
    
    def get_time_slice(self, start_idx: int, time_slice: int):
        file_channels = []
        
        for file_data in self.data:
            subset_data = {}
            for i, time_step in enumerate(range(start_idx, start_idx + time_slice)):
                subset_data[f"arr_{i}"] = file_data[f"arr_{time_step}"]
            
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
