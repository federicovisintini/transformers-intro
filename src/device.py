# Get cpu, gpu or mps device for training.
import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DEVICE = torch.device("cpu")

print(f"Using {DEVICE} device")
