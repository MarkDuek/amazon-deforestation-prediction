import pytest
import numpy as np

from src.data.amazon_dataset import AmazonDataset

def test_load_npz_file(npz_file, time_slice):
    file_path, npz_data = npz_file
    
    dataset = AmazonDataset(file_path, time_slice=time_slice)
    
    for loaded_npz in dataset.data:
        assert set(loaded_npz.keys()) == set(npz_data.keys())

def test_get_time_slice(npz_file, time_idx, time_slice):
    file_path, npz_data = npz_file
    
    dataset = AmazonDataset(file_path, time_slice=time_slice)
    time_slice_data = dataset.get_time_slice(time_idx, time_slice)
    
    assert len(time_slice_data) == len(dataset.data_paths)
    
    file_channel_data = time_slice_data[0]

    expected_keys = {f"arr_{i}" for i in range(time_slice)}
    assert set(file_channel_data.keys()) == expected_keys

def test_stack_sparse_matrix(npz_file, time_idx, time_slice):
    file_path, npz_data = npz_file
    
    dataset = AmazonDataset(file_path, time_slice=time_slice)
    time_slice_data = dataset.get_time_slice(time_idx, time_slice)
    
    stacked_data = dataset.stack_sparse_matrix(time_slice_data[0])
    
    sample_array = npz_data["arr_0"].toarray()
    height, width = sample_array.shape
    
    assert stacked_data.shape == (time_slice, height, width)

def test_concatenate_sparse_matrix(npz_file, time_idx, time_slice):
    file_path, npz_data = npz_file
    
    dataset = AmazonDataset(file_path, time_slice=time_slice)
    time_slice_data = dataset.get_time_slice(time_idx, time_slice)
    
    concatenated_data = dataset.concatenate_sparse_matrix(time_slice_data)
    
    sample_array = npz_data["arr_0"].toarray()
    height, width = sample_array.shape
    channels = len(dataset.data_paths)
    
    assert concatenated_data.shape == (channels, time_slice, height, width)

def test_get_item(npz_file, time_idx, time_slice):
    file_path, npz_data = npz_file
    
    dataset = AmazonDataset(file_path, time_slice=time_slice)
    item = dataset[time_idx]
    
    sample_array = npz_data["arr_0"].toarray()
    height, width = sample_array.shape
    channels = len(dataset.data_paths)
    
    expected_keys = {f"arr_{i}" for i in range(time_slice)}

    assert item.shape == (channels, time_slice, height, width)

def test_len(npz_file, time_slice):
    file_path, npz_data = npz_file
    
    dataset = AmazonDataset(file_path, time_slice=time_slice)
    
    assert len(dataset) == dataset.total_time_steps - time_slice + 1