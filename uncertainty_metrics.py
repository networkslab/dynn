from theft_calibration import calibration_curve
import torch 
from utils import free 
import scipy
def compute_uncertainty_metrics(logits, targets):
    probs = torch.nn.functional.softmax(logits, dim=1)

    top2prob, _ = torch.topk(probs, 2)

    p_max = top2prob[:,0]
    next_p_max = top2prob[:,1]
    margins = p_max-next_p_max

    p_max = free(p_max)
    margins = free(margins)

    entropy = scipy.stats.entropy(free(probs), axis=1)

    pow_probs = probs**2
    pow_probs = pow_probs / pow_probs.sum(dim=1, keepdim=True)
    entropy_pow = scipy.stats.entropy(free(pow_probs), axis=1)

    ece = -1
    if targets is not None:
        _, predicted = logits.max(1)
        correct = predicted.eq(targets)
        ground_truth = free(correct)
        _, _, ece = calibration_curve(ground_truth, p_max)

    return list(p_max), list(entropy), ece, list(margins), list(entropy_pow)
