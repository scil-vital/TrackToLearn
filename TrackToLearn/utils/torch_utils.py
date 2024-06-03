import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def assert_accelerator():
    assert torch.cuda.is_available() or torch.backends.mps.is_available()

def get_device_str():
    return str(get_device())
