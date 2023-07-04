from typing import Tuple
import torch
from torch import Tensor
from models.custom_modules.gate import Gate, GateType
from uncertainty_metrics import compute_detached_uncertainty_metrics

class LearnableCodeGate(Gate):
    def __init__(self, num_proj=4, proj_dim=16):
        super(Gate, self).__init__()
        self.gate_type = GateType.CODE
        self.proj_dim =proj_dim
        self.proj_dim =proj_dim
        self.linear = torch.nn.Linear(proj_dim*num_proj, 1)

    def forward(self, codes: Tensor) -> (Tensor):
        lower_dim_codes = self.random_projection(codes)
        return self.linear(lower_dim_codes)

    def inference_forward(self, input: Tensor, previous_mask: Tensor) -> Tuple[Tensor, Tensor]:
        pass