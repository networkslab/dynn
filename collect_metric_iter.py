# Training

from plotting_util import generate_thresholding_plots
import torch
import scipy
import numpy as np
from theft_calibration import calibration_curve


def free(torch_tensor):
    return torch_tensor.cpu().detach().numpy()



def compute_uncertainty_metrics(logits, targets):
    probs = torch.nn.functional.softmax(logits, dim=1)

    top2prob, _ = torch.topk(probs, 2)

    p_max = top2prob[:,0]
    next_p_max = top2prob[:,1]
    margins = p_max-next_p_max

    p_max = free(p_max)
    margins = free(margins)

    entropy = scipy.stats.entropy(free(probs), axis=1)

    _, predicted = logits.max(1)
    correct = predicted.eq(targets)
    ground_truth = free(correct)
    _, _, ece = calibration_curve(ground_truth, p_max)

    return list(p_max), list(entropy), ece, list(margins)


def collect_metrics(outputs_logits, intermediate_outputs, num_gates, targets,
                    total, correct, device, stored_per_x, stored_metrics):

    _, predicted = outputs_logits.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    # uncertainty related stats to be aggregated
    p_max, entropy, cal, margins = compute_uncertainty_metrics(outputs_logits, targets)
    stored_per_x['final_p_max'] += p_max
    stored_per_x['final_entropy'] += entropy
    stored_per_x['final_margins'] += margins
    stored_metrics['ece'] += cal
    
    # different accuracy to be cumulated
    correctly_classified = torch.full(predicted.eq(targets).shape,
                                      False).to(device)
    for g in range(num_gates):
        # normal accuracy
        _, predicted_inter = intermediate_outputs[g].max(1)
        correct_gate = predicted_inter.eq(targets)
        stored_metrics['correct_per_gate'][g] += correct_gate.sum().item()

        # keeping all the corrects we have from previous gates
        correctly_classified += correct_gate
        stored_metrics['correct_cheating_per_gate'][
            g] += correctly_classified.sum().item()

        p_max, entropy, cal, margins = compute_uncertainty_metrics(
            intermediate_outputs[g], targets)
        stored_per_x['list_correct_per_gate'][g] += list(free(correct_gate))
        stored_per_x['margins_per_gate'][g] += margins
        stored_per_x['p_max_per_gate'][g] += p_max
        stored_per_x['entropy_per_gate'][g] += entropy
        stored_metrics['ece_per_gate'][g] += cal

    correctly_classified += predicted.eq(
        targets)  # getting all the corrects we can
    stored_metrics['cheating_correct'] += correctly_classified.sum().item()

    return stored_per_x, stored_metrics, correct, total


def evaluate_with_gating(thresholds, outputs_logits, intermediate_outputs,
                         targets, stored_metrics):
    G = len(thresholds)
    
    points_reminding = list(range(targets.shape[0]))  # index of all points to classify
   
    gated_outputs = torch.full(outputs_logits.shape,-1.0).to(outputs_logits.device) # outputs storage

    num_classifiction_per_gates = []
    for g, thresh in enumerate(thresholds):
        p_max, _, _, _ = compute_uncertainty_metrics(intermediate_outputs[g],targets)
        
        indices_above_threshold = list(np.argwhere(np.array(p_max) > thresh).flatten())
        
        actual_early_exit_ind = []
        for ind in indices_above_threshold:
            if ind in points_reminding:  # if that index hasn't been classified yet by an earlier gates
                actual_early_exit_ind.append(ind)  # we classify it
                points_reminding.remove( ind)  # we remove it

        num_classifiction_per_gates.append(len(actual_early_exit_ind))
        if len(actual_early_exit_ind) > 0:
            # we add the point to be classified by that gate
            gated_outputs[actual_early_exit_ind, :] = intermediate_outputs[g][
                actual_early_exit_ind, :]
    #classify the reminding points with the end layer
    gated_outputs[points_reminding, :] = outputs_logits[points_reminding, :]

    cost_per_gate = [
        num * (g + 1) / G
        for g, num in enumerate(num_classifiction_per_gates)
    ]
    cost_per_gate.append(len(points_reminding))
    _, gated_pred = gated_outputs.max(1)
    gated_correct = gated_pred.eq(targets).sum().item()
    stored_metrics['gated_correct'] += gated_correct
    #stored_metrics['cost_per_gate'] += cost_per_gate
    stored_metrics['total_cost'] += np.sum(cost_per_gate)
    return stored_metrics



def get_empty_storage_metrics(num_gates):
    stored_per_x = {
        "entropy_per_gate": [[] for _ in range(num_gates)],
        "p_max_per_gate": [[] for _ in range(num_gates)],
        'list_correct_per_gate': [[] for _ in range(num_gates)],
        'margins_per_gate' : [[] for _ in range(num_gates)],
        'final_entropy': [],
        'final_p_max': [],
        'final_margins': []
    }
    stored_metrics = {
        'acc': 0,
        'ece': 0,
        'gated_correct': 0,
        'total_cost': 0,
        'cheating_correct': 0,
        'cost_per_gate': [0 for _ in range(num_gates)],
        'ece_per_gate': [0 for _ in range(num_gates)],
        'correct_per_gate': [0 for _ in range(num_gates)],
        'correct_cheating_per_gate': [0 for _ in range(num_gates)]
    }
    return stored_per_x, stored_metrics

