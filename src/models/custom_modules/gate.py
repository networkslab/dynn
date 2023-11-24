from enum import Enum
from typing import Tuple
from torch import nn
from torch import Tensor


class GateType(Enum):
    UNCERTAINTY = 'unc'
    IDENTITY = 'identity'


class Gate(nn.Module):
    """Abstract class for gating"""

    def __init__(self):
        super(Gate, self).__init__()
        

    def forward(self, input: Tensor) -> Tensor:
        pass
