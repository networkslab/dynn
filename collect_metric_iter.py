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
        correct = list_correct_gate[g] # all correclty classified x at the gate
        p_max_ind = np.argsort(p_max_per_gate)[::-1] # sort the p_max high to low
        sorted_correct = np.array(correct)[p_max_ind]
        sorted_p_max = np.array(p_max_per_gate)[p_max_ind] #[ 0.8, ... 0.4, ... 0.3]
        cumall_correct = np.cumsum(sorted_correct) # cumul the quantity of correctly classified at each threshold 
        min_x = 30 # min x to average the accuracy
        cumall_correct = cumall_correct[min_x:]
        inverse_cost = [c/(i+min_x) for i, c in enumerate(cumall_correct)] # inverse cost is the accuracy for preset threshold
        optimal_index = np.argmax(inverse_cost) + min_x
        threshold_g = sorted_p_max[optimal_index]
        list_optimal_threshold.append(threshold_g)
    return list_optimal_threshold
def compute_uncertainty_metrics(logits):
    probs = torch.nn.functional.softmax(logits, dim=1)
    p_max, _ = probs.max(1)
    p_max = free(p_max)
    entropy = scipy.stats.entropy(free(probs), axis=1)
    return list(p_max), list(entropy)


def collect_metrics(name,
                    epoch,
                    loader,
                    net,
                    optimizer,
                    criterion,
                    device,
                    use_mlflow=True):
    epoch_loss = 0
    correct = 0
    total = 0

    stored_per_x = {
        "entropy_per_gate": [[] for _ in range(net.num_gates)],
        "p_max_per_gate": [[] for _ in range(net.num_gates)],
        'list_correct_per_gate': [[] for _ in range(net.num_gates)],
        'final_entropy': [],
        'final_p_max': []
    }
    stored_metrics = {
        'acc': 0,
        'cheating_correct': 0,
        'correct_per_gate': [0 for _ in range(net.num_gates)],
        'correct_cheating_per_gate': [0 for _ in range(net.num_gates)]
    }
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs_logits, intermediate_outputs = net(inputs)
        loss = criterion(
            outputs_logits,
            targets)  # the grad_fn of this loss should be None if frozen
        for intermediate_output in intermediate_outputs:
            intermediate_loss = criterion(intermediate_output, targets)
            loss += intermediate_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = outputs_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # uncertainty related stats to be aggregated
        p_max, entropy = compute_uncertainty_metrics(outputs_logits)
        stored_per_x['final_p_max'] += p_max
        stored_per_x['final_entropy'] += entropy
        # different accuracy to be cumulated
        correctly_classified = torch.full(predicted.eq(targets).shape,
                                          False).to(device)
        for g in range(net.num_gates):
            # normal accuracy
            _, predicted_inter = intermediate_outputs[g].max(1)
            correct_gate = predicted_inter.eq(targets)
            stored_metrics['correct_per_gate'][g] += correct_gate.sum().item()

            # keeping all the corrects we have from previous gates
            correctly_classified += correct_gate
            stored_metrics['correct_cheating_per_gate'][
                g] += correctly_classified.sum().item()

            p_max, entropy = compute_uncertainty_metrics(
                intermediate_outputs[g])
            stored_per_x['list_correct_per_gate'][g] += list(free(correct_gate))
            stored_per_x['p_max_per_gate'][g] += p_max
            stored_per_x['entropy_per_gate'][g] += entropy

        correctly_classified += predicted.eq(
            targets)  # getting all the corrects we can
        stored_metrics['cheating_correct'] += correctly_classified.sum().item()

        cheating_acc = 100. * stored_metrics['cheating_correct'] / total
        acc = 100. * correct / total
        entropy = np.mean(stored_per_x['final_entropy'])
        progress_bar(
            batch_idx, len(loader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d) | Cheating: %.3f%%' %
            (epoch_loss / (batch_idx + 1), acc, correct, total, cheating_acc))
        
        if use_mlflow:
            log_dict = {
                name + '/loss': epoch_loss / (batch_idx + 1),
                name + '/acc': acc,
                name + '/cheating_acc': cheating_acc,
                name + '/entropy': entropy,
            }
            for g in range(net.num_gates):
                acc_gate = 100. * stored_metrics['correct_per_gate'][g] / total
                acc_cheating_gate = 100. * stored_metrics[
                    'correct_cheating_per_gate'][g] / total
                entropy_per_gate = np.mean(stored_per_x['entropy_per_gate'][g])
                log_dict[name + '/acc' + str(g)] = acc_gate
                log_dict[name + '/cheating_acc' + str(g)] = acc_cheating_gate
                log_dict[name + '/entropy' + str(g)] = entropy_per_gate
            mlflow.log_metrics(log_dict,
                               step=batch_idx + (epoch * len(loader)))
    threshold = compute_optimal_threshold(stored_per_x['p_max_per_gate'], stored_per_x['list_correct_per_gate'])
    stored_metrics['acc'] = acc
    stored_metrics['optim_threshold'] = threshold
    return stored_metrics
