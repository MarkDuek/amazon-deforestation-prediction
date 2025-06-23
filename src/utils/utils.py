import torch


def get_device(device_str: str = "cpu") -> torch.device:
    """
    Returns a torch.device object based on the input string.
    If device_str is 'cuda' and CUDA is not available, falls back to CPU.
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
