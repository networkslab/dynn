import torch
from torch import Tensor
from models.custom_modules.gate import Gate

class ConfidenceScoreThresholdGate(Gate):
    """Concrete gate that accepts the """
    def __init__(self, confidence_score_threshold):
        super(Gate, self).__init__()
        self.threshold = confidence_score_threshold

    def forward(self, input: Tensor) -> (Tensor, Tensor, Tensor):
        """Returns 2 equal-size tensors, the prediction tensor and a tensor containing the indices of predictions
        :param input: The softmax logits of the classifier
        """
        max_probs = input.max(dim = 1)
        idx_preds_above_threshold = (max_probs.values > self.threshold).nonzero()
        idx_preds_below_threshold = (max_probs.values <= self.threshold).nonzero() # The ones we still need to classify upstream
        confident_preds = max_probs.indices[idx_preds_above_threshold]
        return confident_preds, torch.flatten(idx_preds_above_threshold), torch.flatten(idx_preds_below_threshold)
