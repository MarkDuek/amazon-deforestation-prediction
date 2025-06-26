import pytest
import numpy as np
from pathlib import Path
import sys
from scipy import sparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    return request.param

@pytest.fixture(params=[
    {f"arr_{i}": sparse.csr_matrix(np.array([[i*4+1, i*4+2], [i*4+3, i*4+4]])) for i in range(175)},
    {f"arr_{i}": sparse.csr_matrix(np.array([[i*0.4+0.1, i*0.4+0.2], [i*0.4+0.3, i*0.4+0.4]])) for i in range(175)},
    {f"arr_{i}": sparse.csr_matrix(np.array([[i*4.4+1.1, i*4.4+2.2], [i*4.4+3.3, i*4.4+4.4]])) for i in range(175)},
])
def npz_data(request):
    return request.param

@pytest.fixture(params=[5])
def npz_file(tmp_path: Path, npz_data: dict, request):
    file_paths: List[str] = []
    for i in range(request.param):
        file_path = tmp_path / f"test{i}.npz"
        np.savez(file_path, **npz_data)
        file_paths.append(str(file_path))
    return file_paths, npz_data

@pytest.fixture(params=[7])
def time_slice(request):
    return request.param

@pytest.fixture(params=[0, 1, 2])
def time_idx(request):
    return request.param