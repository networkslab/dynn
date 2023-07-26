from typing import Tuple
import torch
from torch import Tensor
from models.custom_modules.gate import Gate, GateType

import numpy as np
from sklearn import random_projection

class LearnableCodeGate(Gate):
    def __init__(self, device, input_dim, num_proj, proj_dim):
        super(Gate, self).__init__()
        self.gate_type = GateType.CODE
        self.num_proj =num_proj
        self.proj_dim =proj_dim
        self.input_dim = int(input_dim) # HARDCODED
        self.device = device # this is terrible
        random_projs = []
        for _ in range(self.num_proj):
            random_projs.append(torch.normal(0., 1/proj_dim, (self.input_dim, self.proj_dim)))

        self.random_projs = torch.cat(random_projs, dim=1).to(self.device)
        self.linear1 = torch.nn.Linear(proj_dim*num_proj, proj_dim)
        self.linear2 = torch.nn.Linear(proj_dim, 1)

    def forward(self, codes: Tensor) -> (Tensor):
        flat_code = codes.reshape(-1,self.input_dim)
        lower_dim_codes =  torch.matmul(flat_code.float(), self.random_projs)
        x = self.linear1(lower_dim_codes)
        x = torch.functional.F.relu(x)
        x = self.linear2(x)
        return x

    def inference_forward(self, input: Tensor, previous_mask: Tensor) -> Tuple[Tensor, Tensor]:
        pass