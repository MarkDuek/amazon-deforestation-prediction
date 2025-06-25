import pytest

@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    return request.param

