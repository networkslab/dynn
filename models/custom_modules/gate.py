from torch import nn
from torch import Tensor

class Gate(nn.Module):
    """Abstract class for gating"""
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, input: Tensor, previous_mask: Tensor) -> (Tensor, Tensor):
        pass
