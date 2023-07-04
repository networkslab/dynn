from typing import Tuple
import torch
from torch import Tensor
from models.custom_modules.gate import Gate, GateType
from uncertainty_metrics import compute_detached_uncertainty_metrics

class LearnableUncGate(Gate):
    def __init__(self):
        super(Gate, self).__init__()
        self.gate_type = GateType.UNCERTAINTY
        self.dim = 4
        self.linear = torch.nn.Linear(self.dim, 1)

    def forward(self, logits: Tensor) -> (Tensor):
        p_maxes, entropies, _, margins, entropy_pows = compute_detached_uncertainty_metrics(logits, None)
        p_maxes = torch.tensor(p_maxes)[:, None]
        entropies = torch.tensor(entropies)[:, None]
        margins = torch.tensor(margins)[:, None]
        entropy_pows = torch.tensor(entropy_pows)[:, None]
        uncertainty_metrics = torch.cat((p_maxes, entropies, margins, entropy_pows), dim = 1)
        return self.linear(uncertainty_metrics.to(logits.device))

    def inference_forward(self, input: Tensor, previous_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns 2 equal-size tensors, the prediction tensor and a tensor containing the indices of predictions
        :param input: The softmax logits of the classifier
        """
        input = torch.mul(
            torch.logical_not(previous_mask).to('cuda').float()[:, None],
            input
        )
        max_probs = input.max(dim = 1)
        idx_preds_above_threshold = torch.flatten((max_probs.values > self.threshold).nonzero())
        confident_preds = torch.index_select(input, 0, idx_preds_above_threshold)
        mask = torch.zeros(input.shape[0], dtype=torch.bool)
        mask[idx_preds_above_threshold] = True
        return confident_preds, mask # 1 means early exit, 0 means propagate downstream
