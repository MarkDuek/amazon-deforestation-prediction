"""Script to patch the data"""

import logging
import os
import sys
import h5py
import numpy as np

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_utils import (concatenate_sparse_matrix, get_patch,
                                  get_time_slice, load_npz_files,
                                  pad_to_multiple, inspect_h5_file)
from src.utils.utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

config = load_config("config.yaml")

# load data
data_paths = config["data"]["paths"]
patch_size = config["data"]["patch_size"]
padding_multiple = config["data"]["padding_multiple"]
data = load_npz_files(data_paths)

total_time_steps = len(data[0].keys())

# Output files
input_h5 = h5py.File("src/data/patches/input_patches.h5", "w")
target_h5 = h5py.File("src/data/patches/target_patches.h5", "w")

# Assume you know C, h, w — or extract from first patch later
# We'll defer this to inside the loop if needed

# Start empty and allow resizing
input_datasets = {}
target_datasets = {}
total_patches = 0

C = len(data_paths)
h, w = patch_size, patch_size
logger.info("C: %d, h: %d, w: %d", C, h, w)

logger.info("Initializing datasets")

for i in range(total_time_steps):

    logger.info("Processing time step %d/%d", i + 1, total_time_steps)

    time_slice_data = get_time_slice(data, i, 1)

    concatenated_data = concatenate_sparse_matrix(time_slice_data)

    # (C, T-1, H, W) - first T-1 time slices
    input_data = torch.tensor(concatenated_data[:, 0, :, :])
    # (1, 1, H, W) - last time step of second channel
    target = (
        torch.tensor(concatenated_data[1, -1, :, :]).unsqueeze(0)
    )

    input_data = pad_to_multiple(input_data, padding_multiple)
    target = pad_to_multiple(target, padding_multiple)

    input_patches = get_patch(input_data, patch_size)
    target_patches = get_patch(target, patch_size)

    input_patches_list = input_patches
    target_patches_list = target_patches

    if total_patches == 0:
        C, h, w = input_patches_list[0]["tensor"].shape

        input_datasets["patches"] = input_h5.create_dataset(
            "patches", shape=(0, C, h, w), maxshape=(None, C, h, w),
            chunks=True, dtype="float32"
        )
        input_datasets["positions"] = input_h5.create_dataset(
            "positions", shape=(0, 2), maxshape=(None, 2),
            chunks=True, dtype="int32"
        )
        input_datasets["time_indices"] = input_h5.create_dataset(
            "time_indices", shape=(0,), maxshape=(None,),
            chunks=True, dtype="int32"
        )

        target_datasets["patches"] = target_h5.create_dataset(
            "patches", shape=(0, 1, h, w), maxshape=(None, 1, h, w),
            chunks=True, dtype="float32"
        )
        target_datasets["positions"] = target_h5.create_dataset(
            "positions", shape=(0, 2), maxshape=(None, 2),
            chunks=True, dtype="int32"
        )
        target_datasets["time_indices"] = target_h5.create_dataset(
            "time_indices", shape=(0,), maxshape=(None,),
            chunks=True, dtype="int32"
        )

    # Extract tensors, positions, and time index for input
    input_tensors = np.stack([p["tensor"].numpy() for p in input_patches_list])
    input_positions = np.array(
        [p["position"] for p in input_patches_list], dtype="int32"
    )
    input_time = np.full((len(input_patches_list),), i, dtype="int32")

    # Same for targets
    target_tensors = np.stack(
        [p["tensor"].numpy() for p in target_patches_list]
    )
    target_positions = np.array(
        [p["position"] for p in target_patches_list], dtype="int32"
    )
    target_time = np.full((len(target_patches_list),), i, dtype="int32")

    num_new = len(input_patches_list)

    # Resize input datasets
    for key, array in zip(
        ["patches", "positions", "time_indices"],
        [input_tensors, input_positions, input_time]
    ):
        input_datasets[key].resize(total_patches + num_new, axis=0)
        input_datasets[key][total_patches:total_patches + num_new] = array

    # Resize target datasets
    for key, array in zip(
        ["patches", "positions", "time_indices"],
        [target_tensors, target_positions, target_time]
    ):
        target_datasets[key].resize(total_patches + num_new, axis=0)
        target_datasets[key][total_patches:total_patches + num_new] = array

    total_patches += num_new

input_h5.close()
target_h5.close()

inspect_h5_file("src/data/patches/input_patches.h5")
inspect_h5_file("src/data/patches/target_patches.h5")

logger.info("Data patching completed successfully")
