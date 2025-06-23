import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Mapping, List

class AmazonDataset(Dataset):

    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform
        self.data = self.load_data(data_paths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return torch.tensor()

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
            obj = data[f'arr_{i}']
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
            self.stack_sparse_matrix(data, d_type=np.float32)
        for data in data_list
        ]

        return np.concatenate(arrays)

    def load_data(
            self,
            data_paths: List[str],
    ) -> np.ndarray:

        data_list = self.load_npz_files(data_paths)
        data = self.concatenate_sparse_matrix(data_list)

        return data