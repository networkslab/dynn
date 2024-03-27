from typing import Tuple
import torch
from torch import Tensor
from models.custom_modules.gate import Gate, GateType

# This gate does not do anything, we use it for simplifcation of FLOPS calculation of the baselines.
class IdentityGate(Gate):
    def __init__(self):
        super(Gate, self).__init__()
        self.gate_type = 'IDENTITY'

    def forward(self, logits: Tensor) -> (Tensor):
        return logits

    def get_flops(self, num_classes):
        # This gate has no cost, it is only here to help flops calculation of the baselines.
        return 0
