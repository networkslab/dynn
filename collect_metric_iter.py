# Training

from ploting_util import generate_thresholding_plots
import torch
import scipy
import numpy as np
from theft_calibration import calibration_curve


def free(torch_tensor):
    return torch_tensor.cpu().detach().numpy()

# define a threshold such that each layer tries to hit the target accuracy
def compute_optimal_threshold(threhsold_name, all_p_max, list_correct_gate, target_acc=1):
    list_optimal_threshold = []
    min_x = 10  # min x to average the accuracy
    # store things for plots
    all_sorted_p_max = []
    all_cumul_acc = []
    all_correct = []

    for g, p_max_per_gate in enumerate(all_p_max): # for each gates
        correct = list_correct_gate[ g]  # all correclty classified x at the gate
        p_max_ind = np.argsort(p_max_per_gate)[::-1]  # argsort the p_max high to low 

        sorted_correct = np.array(correct)[p_max_ind] # sort the correct matching the p max  => [1, 1, 0.... 1, 0]
        sorted_p_max = np.array(p_max_per_gate)[ p_max_ind]  # sort the correct matching the p max  => [0.96, 0.9, .... 0.4, 0.1]
        
        cumall_correct = np.cumsum(sorted_correct) 
        cumul_acc = [c / (i +1) for i, c in enumerate(cumall_correct)]  # get the accuracy at each threshold [1,0.9,...0.3]
        
        # store things for plots
        all_sorted_p_max.append(list(sorted_p_max))
        all_cumul_acc.append(cumul_acc)
        all_correct.append(list(sorted_correct))

         
        cumul_acc = cumul_acc[min_x:] # cut the first points to avoid variance issue when averaging 
        
        indices_target_acc = np.argwhere(np.array(cumul_acc)>target_acc) # get all threshold with higher acc tahn target:
        """
        target_acc = 0.5
        cumul_acc = [0.8, 0.7,| 0.3, 0.3, 0.4]
        indices_target_acc = [0,1]
        """
        
        if len(indices_target_acc) == 0: # if no one can hit the accuracy, we set the threshold to 1
            threshold_g = 1
            optimal_index = np.argmax(cumul_acc) + min_x
        else:
            optimal_index = int(indices_target_acc[-1]) + min_x # we get the last threshold that has higher acc 
            threshold_g = sorted_p_max[optimal_index]
        list_optimal_threshold.append(threshold_g)

    generate_thresholding_plots(threhsold_name, all_sorted_p_max, all_cumul_acc, all_correct, min_x, target_acc, list_optimal_threshold)
    return list_optimal_threshold


def compute_uncertainty_metrics(logits, targets):
    probs = torch.nn.functional.softmax(logits, dim=1)
    p_max, _ = probs.max(1)
    p_max = free(p_max)
    entropy = scipy.stats.entropy(free(probs), axis=1)

    _, predicted = logits.max(1)
    correct = predicted.eq(targets)
    ground_truth = free(correct)
    _, _, ece = calibration_curve(ground_truth, p_max)

    return list(p_max), list(entropy), ece


def collect_metrics(outputs_logits, intermediate_outputs, num_gates, targets,
                    total, correct, device, stored_per_x, stored_metrics):

    _, predicted = outputs_logits.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    # uncertainty related stats to be aggregated
    p_max, entropy, cal = compute_uncertainty_metrics(outputs_logits, targets)
    stored_per_x['final_p_max'] += p_max
    stored_per_x['final_entropy'] += entropy
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

        p_max, entropy, cal = compute_uncertainty_metrics(
            intermediate_outputs[g], targets)
        stored_per_x['list_correct_per_gate'][g] += list(free(correct_gate))
        stored_per_x['p_max_per_gate'][g] += p_max
        stored_per_x['entropy_per_gate'][g] += entropy
        stored_metrics['ece_per_gate'][g] += cal

    correctly_classified += predicted.eq(
        targets)  # getting all the corrects we can
    stored_metrics['cheating_correct'] += correctly_classified.sum().item()

    return stored_per_x, stored_metrics, correct, total


def evaluate_with_gating(threshold, outputs_logits, intermediate_outputs,
                         targets, stored_metrics):
    G = len(threshold)
    
    points_reminding = list(range(targets.shape[0]))  # index of all points to classify
   
    gated_outputs = torch.full(outputs_logits.shape,-1.0).to(outputs_logits.device) # outputs storage

    num_classifiction_per_gates = []
    for g, thresh in enumerate(threshold):
        p_max, _, _ = compute_uncertainty_metrics(intermediate_outputs[g],targets)
        
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
        'final_entropy': [],
        'final_p_max': []
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

