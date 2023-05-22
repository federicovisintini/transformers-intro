import torch
from torch import nn

# Get cpu, gpu or mps device for training.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(512, 32)
        self.fn1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 4)
        self.fn2 = nn.Softmax()

    def forward(self, x):
        out1 = self.fn1(self.layer1(x))
        return self.fn2(self.layer2(out1))
