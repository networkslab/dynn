from theft_calibration import calibration_curve
import torch 
from utils import free 
import scipy

def check_hamming_vs_acc(intermediate_logits, intermediate_codes, targets):
    inc_inc_H_list = []
    c_c_H_list = []
    c_inc_H_list = []
    for g, code in enumerate(intermediate_codes):
        _, predicted = intermediate_logits[g].max(1)
        correct_id = predicted.eq(targets)
        sorted_indices = torch.sort(correct_id.float())[1]
        code = code.reshape(targets.shape[0], -1).float()
        length_code = code.shape[1]
        code = code[sorted_indices]
        H = torch.pairwise_distance(code[:, None], code, p=1)
        num_correct = correct_id.sum()
        c_c_H = torch.mean(H[-num_correct:, -num_correct:])/length_code
        c_inc_H = torch.mean(H[:-num_correct, -num_correct:])/length_code
        inc_inc_H = torch.mean(H[:-num_correct, :-num_correct])/length_code
        
        inc_inc_H_list.append(inc_inc_H)
        c_c_H_list.append(c_c_H)
        c_inc_H_list.append(c_inc_H)
    return inc_inc_H_list, c_c_H_list, c_inc_H_list

def compute_detached_uncertainty_metrics(logits, targets):
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
