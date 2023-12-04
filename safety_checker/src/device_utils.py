import torch

# Reference:
# https://github.com/woctezuma/feature-extractor/blob/minimal/src/device_utils.py


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
