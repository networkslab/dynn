from typing import Tuple
import torch
from torch import Tensor
from models.custom_modules.gate import Gate, GateType


import numpy as np
from sklearn import random_projection
from metrics_utils import compute_detached_uncertainty_metrics
class LearnableComplexGate(Gate):
    """
    Takes both relu code which are projected and the uncertainty metrics to decide whether to exit or not
    """
    def __init__(self, device, input_dim, num_proj=32, proj_dim=32):
        super(Gate, self).__init__()
        self.gate_type = GateType.CODE_AND_UNC
        self.num_proj =num_proj
        self.proj_dim =proj_dim
        self.input_dim = int(input_dim) 
        self.device = device # this is terrible
        random_projs = []
        for _ in range(self.num_proj):
            random_projs.append(torch.normal(0., 1/proj_dim, (self.input_dim, self.proj_dim)))

        self.random_projs = torch.cat(random_projs, dim=1).to(self.device)
        self.linear1 = torch.nn.Linear(proj_dim*num_proj + 4, proj_dim)
        self.linear2 = torch.nn.Linear(proj_dim, 1)

    def forward(self, codes: Tensor, logits: Tensor) -> (Tensor):
        flat_code = codes.reshape(-1, self.input_dim)
        lower_dim_codes = torch.matmul(flat_code.float(), self.random_projs)
        p_maxes, entropies, _, margins, entropy_pows = compute_detached_uncertainty_metrics(logits, None)
        p_maxes = torch.tensor(p_maxes)[:, None]
        entropies = torch.tensor(entropies)[:, None]
        margins = torch.tensor(margins)[:, None]
        entropy_pows = torch.tensor(entropy_pows)[:, None]
        p_maxes = p_maxes.to(device='cuda')
        entropies = entropies.to(device='cuda')
        margins = margins.to(device='cuda')
        entropy_pows = entropy_pows.to(device='cuda')
        # lower_dim_codes = lower_dim_codes[:, None]
        input = torch.cat((lower_dim_codes, p_maxes, entropies, margins, entropy_pows), dim = 1)
        x = self.linear1(input)
        x = torch.functional.F.relu(x)
        x = self.linear2(x)
        return x

    def inference_forward(self, input: Tensor, previous_mask: Tensor) -> Tuple[Tensor, Tensor]:
        pass