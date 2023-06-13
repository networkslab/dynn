# Training

from utils import progress_bar
import torch
import mlflow
import scipy
import numpy as np


def free(torch_tensor):
    return torch_tensor.cpu().detach().numpy()


def compute_optimal_threshold(all_p_max, list_correct_gate):
    list_optimal_threshold = []
    for g, p_max_per_gate in enumerate(all_p_max):
        correct = list_correct_gate[
            g]  # all correclty classified x at the gate
        p_max_ind = np.argsort(
            p_max_per_gate)[::-1]  # sort the p_max high to low
        sorted_correct = np.array(correct)[p_max_ind]
        sorted_p_max = np.array(p_max_per_gate)[
            p_max_ind]  #[ 0.8, ... 0.4, ... 0.3]
        cumall_correct = np.cumsum(
            sorted_correct
        )  # cumul the quantity of correctly classified at each threshold
        min_x = 30  # min x to average the accuracy
        cumall_correct = cumall_correct[min_x:]
        inverse_cost = [c / (i + min_x) for i, c in enumerate(cumall_correct)
                        ]  # inverse cost is the accuracy for preset threshold
        optimal_index = np.argmax(inverse_cost) + min_x
        threshold_g = sorted_p_max[optimal_index]
        list_optimal_threshold.append(threshold_g)
    return list_optimal_threshold


def compute_uncertainty_metrics(logits, targets):
    probs = torch.nn.functional.softmax(logits, dim=1)
    p_max, _ = probs.max(1)
    p_max = free(p_max)
    entropy = scipy.stats.entropy(free(probs), axis=1)
    ECE=0
    return list(p_max), list(entropy), ECE


def collect_metrics(outputs_logits, intermediate_outputs, num_gates, targets,
                    total, correct, device, stored_per_x, stored_metrics):

    _, predicted = outputs_logits.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    # uncertainty related stats to be aggregated
    p_max, entropy, cal = compute_uncertainty_metrics(outputs_logits, targets)
    stored_per_x['final_p_max'] += p_max
    stored_per_x['final_entropy'] += entropy
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

        p_max, entropy, cal = compute_uncertainty_metrics(intermediate_outputs[g], targets)
        stored_per_x['list_correct_per_gate'][g] += list(free(correct_gate))
        stored_per_x['p_max_per_gate'][g] += p_max
        stored_per_x['entropy_per_gate'][g] += entropy

    correctly_classified += predicted.eq(
        targets)  # getting all the corrects we can
    stored_metrics['cheating_correct'] += correctly_classified.sum().item()

    return stored_per_x, stored_metrics, correct, total


def evaluate_with_gating(threshold, outputs_logits, intermediate_outputs,
                         targets, stored_metrics):
    num_gates = len(threshold)
    # this will iterate over the gates with thresholding
    x_index = list(range(targets.shape[0]))  # index of all points to classify
    gated_outputs = torch.full(outputs_logits.shape,
                               -1.0).to(outputs_logits.device)
    num_classifiction_per_gates = []
    for g, thresh in enumerate(threshold):
        p_max, _ = compute_uncertainty_metrics(intermediate_outputs[g])
        early_exit_ind = list(np.argwhere(p_max > thresh).flatten())
        actual_early_exit_ind = []
        for ind in early_exit_ind:
            if ind in x_index:  # if that index hasnt been classified yet by an earlier gates
                actual_early_exit_ind.append(ind)  # we classify it
                x_index.remove(
                    ind)  # we remove that index to be classified in the future

        num_classifiction_per_gates.append(len(actual_early_exit_ind))
        if len(actual_early_exit_ind) > 0:
            gated_outputs[actual_early_exit_ind, :] = intermediate_outputs[
                g][actual_early_exit_ind, :]
    #classify the reminding points with the end layer
    gated_outputs[x_index, :] = outputs_logits[x_index, :]

    cost_per_gate = [
        num * (g + 1) / num_gates
        for g, num in enumerate(num_classifiction_per_gates)
    ]
    cost_per_gate.append(len(x_index))
    _, gated_pred = gated_outputs.max(1)
    gated_correct = gated_pred.eq(targets).sum().item()
    stored_metrics['gated_correct'] += gated_correct
    stored_metrics['cost_per_gate'] += cost_per_gate
    stored_metrics['total_cost'] += np.sum(cost_per_gate)
    return stored_metrics


def get_loss(inputs, targets, optimizer, criterion, net):

    optimizer.zero_grad()
    outputs_logits, intermediate_outputs = net(inputs)
    loss = criterion(
        outputs_logits,
        targets)  # the grad_fn of this loss should be None if frozen
    for intermediate_output in intermediate_outputs:
        intermediate_loss = criterion(intermediate_output, targets)
        loss += intermediate_loss
    return loss, outputs_logits, intermediate_outputs


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
        'gated_correct': 0,
        'total_cost': 0,
        'cheating_correct': 0,
        'cost_per_gate': [0 for _ in range(num_gates)],
        'correct_per_gate': [0 for _ in range(num_gates)],
        'correct_cheating_per_gate': [0 for _ in range(num_gates)]
    }
    return stored_per_x, stored_metrics
