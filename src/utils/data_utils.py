"""Data utility functions"""

import logging
from typing import Any, Dict, List, Mapping
from numpy.lib.npyio import NpzFile
import h5py

import numpy as np
import torch
import torch.nn.functional as F


# data methods
def load_npz_files(
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
    data_list: List[Mapping[str, Any]],
) -> np.ndarray:
    """Concatenate sparse matrices from multiple data sources.

    Args:
        data_list: List of data mappings to concatenate

    Returns:
        Concatenated numpy array
    """
    arrays: List[np.ndarray] = [
        stack_sparse_matrix(data, d_type=np.float32) for data in data_list
    ]

    return np.stack(arrays)


def get_time_slice(
    data: List[NpzFile[Any]],
    time_idx: int,
    time_slice: int = 1,
) -> List[Mapping[str, Any]]:
    """Extract a time slice from the dataset.

    Args:
        time_idx: Starting time index
        time_slice: Length of the time slice

    Returns:
        List of mappings containing the time slice data
    """
    file_channels: List[Mapping[str, Any]] = []

    for file_data in data:
        subset_data: Dict[str, Any] = {}
        for i, time_step in enumerate(range(time_idx, time_idx + time_slice)):
            key = f"arr_{time_step}"
            if key in file_data:
                subset_data[f"arr_{i}"] = file_data[key]
            # else:
            # self.logger.warning("Key '%s' not found in file data", key)

        file_channels.append(subset_data)

    return file_channels


# padding methods
def pad_to_multiple(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad tensor dimensions to be multiples of a given value.

    Args:
        tensor: Input tensor to pad
        multiple: Value to pad dimensions to multiples of

    Returns:
        Padded tensor
    """
    # self.logger.debug(
    #     "Padding tensor with shape %s to multiple %s",
    #     tensor.shape,
    #     multiple,
    # )

    *_, h, w = tensor.shape
    target_h = ((h + multiple - 1) // multiple) * multiple
    target_w = ((w + multiple - 1) // multiple) * multiple
    # self.logger.debug("Target shape: %sx%s", target_h, target_w)

    pad_h, pad_w = target_h - h, target_w - w
    padding = (
        pad_w // 2,
        pad_w - pad_w // 2,
        pad_h // 2,
        pad_h - pad_h // 2,
    )
    # self.logger.debug("Padding: %s", padding)

    return F.pad(tensor, padding, mode="constant", value=0.0)


# patch methods
def get_patch(data: torch.Tensor, patch_size: int) -> List[Dict[str, Any]]:
    """Extract patches from a tensor.

    Args:
        data: Input tensor with shape (C, T, H, W)
        patch_size: Size of each patch to extract

    Returns:
        List of dictionaries containing patch data with keys:
        - patch: The extracted patch tensor
        - x: X coordinate of the patch
        - y: Y coordinate of the patch
    """
    num_x_patches = data.shape[1] // patch_size
    num_y_patches = data.shape[2] // patch_size

    patches: List[Dict[str, Any]] = []
    for x in range(num_x_patches):
        for y in range(num_y_patches):
            patch = data[
                :,
                x * patch_size: (x + 1) * patch_size,
                y * patch_size: (y + 1) * patch_size,
            ]
            patches.append(
                {
                    "tensor": patch,
                    "position": (x, y),
                }
            )

    return patches


def inspect_h5_file(file_path: str) -> None:
    """Inspect and print information about datasets in an HDF5 file.

    Args:
        file_path: Path to the HDF5 file to inspect

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info("Inspecting %s:", file_path)
    with h5py.File(file_path, "r") as f:
        for name, dataset in f.items():
            logger.info(
                "  Dataset '%s': shape = %s, dtype = %s",
                name, dataset.shape, dataset.dtype
            )
