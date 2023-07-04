from typing import Tuple
import torch
from torch import Tensor
from models.custom_modules.gate import Gate, GateType

import numpy as np
from sklearn import random_projection

class LearnableCodeGate(Gate):
    def __init__(self, input_dim, num_proj=4, proj_dim=16.):
        super(Gate, self).__init__()
        self.gate_type = GateType.CODE
        self.num_proj =num_proj
        self.proj_dim =proj_dim
        self.input_dim = input_dim
        random_projs = []
        for _ in range(self.num_proj):
            random_projs.append(torch.normal(0., 1/proj_dim, out=(self.input_dim, self.proj_dim)))

        self.random_projs = torch.cat(random_projs)
        self.linear = torch.nn.Linear(proj_dim*num_proj, 1)

    def forward(self, codes: Tensor) -> (Tensor):
        lower_dim_codes = self.random_projs * codes
        return self.linear(lower_dim_codes)

    def inference_forward(self, input: Tensor, previous_mask: Tensor) -> Tuple[Tensor, Tensor]:
        pass