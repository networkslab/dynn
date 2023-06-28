import torch
from torch import Tensor
from src.models.custom_modules.gate import Gate

class ConfidenceScoreThresholdGate(Gate):
    """Concrete gate that accepts the """
    def __init__(self, confidence_score_threshold):
        super(Gate, self).__init__()
        self.threshold = confidence_score_threshold

    def inference_forward(self, input: Tensor, previous_mask: Tensor) -> (Tensor, Tensor):
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
