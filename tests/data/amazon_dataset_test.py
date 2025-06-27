import numpy as np
import pytest

from src.data.amazon_dataset import AmazonDataset


def test_load_npz_file(config, npz_file):
    file_path, npz_data = npz_file

    config["data"]["paths"] = file_path
    dataset = AmazonDataset(config)

    for loaded_npz in dataset.data:
        assert set(loaded_npz.keys()) == set(npz_data.keys())


def test_get_time_slice(config, npz_file, time_idx, time_slice):
    file_path, npz_data = npz_file

    config["data"]["paths"] = file_path
    dataset = AmazonDataset(config)
    time_slice_data = dataset.get_time_slice(time_idx, time_slice)

    assert len(time_slice_data) == len(dataset.data_paths)

    file_channel_data = time_slice_data[0]

    expected_keys = {f"arr_{i}" for i in range(time_slice)}
    assert set(file_channel_data.keys()) == expected_keys


def test_stack_sparse_matrix(config, npz_file, time_idx, time_slice):
    file_path, npz_data = npz_file

    config["data"]["paths"] = file_path
    dataset = AmazonDataset(config)
    time_slice_data = dataset.get_time_slice(time_idx, time_slice)

    stacked_data = dataset.stack_sparse_matrix(time_slice_data[0])

    sample_array = npz_data["arr_0"].toarray()
    height, width = sample_array.shape

    assert stacked_data.shape == (time_slice, height, width)


def test_concatenate_sparse_matrix(config, npz_file, time_idx, time_slice):
    file_path, npz_data = npz_file

    config["data"]["paths"] = file_path
    dataset = AmazonDataset(config)
    time_slice_data = dataset.get_time_slice(time_idx, time_slice)

    concatenated_data = dataset.concatenate_sparse_matrix(time_slice_data)

    sample_array = npz_data["arr_0"].toarray()
    height, width = sample_array.shape
    channels = len(dataset.data_paths)

    assert concatenated_data.shape == (channels, time_slice, height, width)


def test_get_item(config, npz_file, time_idx, time_slice):
    file_path, npz_data = npz_file

    config["data"]["paths"] = file_path
    dataset = AmazonDataset(config)
    input_data, target_data = dataset[time_idx]

    sample_array = npz_data["arr_0"].toarray()
    height, width = sample_array.shape

    height = ((height // 32) + 1) * 32
    width = ((width // 32) + 1) * 32

    channels = len(dataset.data_paths)

    expected_keys = {f"arr_{i}" for i in range(time_slice)}

    assert input_data.shape == (channels, time_slice, height, width)
    assert target_data.shape == (height, width)


def test_len(config, npz_file, time_slice):
    file_path, npz_data = npz_file

    config["data"]["paths"] = file_path
    dataset = AmazonDataset(config)

    assert len(dataset) == dataset.total_time_steps - time_slice + 1
