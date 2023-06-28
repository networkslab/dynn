# Training

from models.t2t_vit import TrainingPhase
from plotting_util import generate_thresholding_plots
import torch

import numpy as np
from threshold_helper import return_ind_thrs
from uncertainty_metrics import compute_detached_uncertainty_metrics
from utils import free


# define a threshold such that each layer tries to hit the target accuracy
def compute_optimal_threshold(threshold_name, all_p_max, list_correct_gate, target_acc=1):
    list_optimal_threshold = []
    min_x = 10  # min x to average the accuracy
    # store things for plots
    all_sorted_p_max = []
    all_cumul_acc = []
    all_correct = []

    for g, p_max_per_gate in enumerate(all_p_max): # for each gates
        correct = list_correct_gate[g]  # all correctly classified x at the gate
        p_max_ind = np.argsort(p_max_per_gate)[::-1]  # argsort the p_max high to low 

        sorted_correct = np.array(correct)[p_max_ind] # sort the correct matching the p max  => [1, 1, 0.... 1, 0]
        sorted_p_max = np.array(p_max_per_gate)[p_max_ind]  # sort the correct matching the p max  => [0.96, 0.9, .... 0.4, 0.1]
        
        cumall_correct = np.cumsum(sorted_correct) 
        cumul_acc = [c / (i +1) for i, c in enumerate(cumall_correct)]  # get the accuracy at each threshold [1,0.9,...0.3]
        
        # store things for plots
        all_sorted_p_max.append(list(sorted_p_max))
        all_cumul_acc.append(cumul_acc)
        all_correct.append(list(sorted_correct))

         
        cumul_acc = cumul_acc[min_x:] # cut the first points to avoid variance issue when averaging 
        
        indices_target_acc = np.argwhere(np.array(cumul_acc)>target_acc) # get all thresholds with higher acc than target:
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

    generate_thresholding_plots(threshold_name, all_sorted_p_max, all_cumul_acc, all_correct, min_x, target_acc, list_optimal_threshold)
    return list_optimal_threshold


def collect_metrics(things_of_interest, G, targets,
                    device, stored_per_x, stored_metrics, training_phase):
    if training_phase == TrainingPhase.CLASSIFIER or training_phase == TrainingPhase.WARMUP:
        intermediate_logits = things_of_interest['intermediate_logits']
        final_y_logits = things_of_interest['final_logits']
        _, pred_final_head = final_y_logits.max(1)
        stored_metrics['final_head_correct_all'] += pred_final_head.eq(targets).sum().item()

        # uncertainty related stats to be aggregated
        p_max, entropy, ece, margins, entropy_pow = compute_detached_uncertainty_metrics(final_y_logits, targets)
        stored_per_x['final_p_max'] += p_max
        stored_per_x['final_entropy'] += entropy
        stored_per_x['final_pow_entropy'] += entropy_pow
        stored_per_x['final_margins'] += margins
        
        stored_metrics['final_ece'] += ece

        # the cheating accuracy
        correctly_classified = torch.full(pred_final_head.eq(targets).shape,
                                        False).to(device)
        for g in range(G):
            # normal accuracy
            _, predicted_inter = intermediate_logits[g].max(1)
            correct_gate = predicted_inter.eq(targets)
            stored_metrics['correct_per_gate'][g] += correct_gate.sum().item()
            
            # keeping all the corrects we have from previous gates
            correctly_classified += correct_gate
            stored_metrics['correct_cheating_per_gate'][
                g] += correctly_classified.sum().item()

            p_max, entropy, cal, margins, entropy_pow = compute_detached_uncertainty_metrics(
                intermediate_logits[g], targets)
            stored_per_x['list_correct_per_gate'][g] += list(free(correct_gate))
            stored_per_x['margins_per_gate'][g] += margins
            stored_per_x['p_max_per_gate'][g] += p_max
            stored_per_x['entropy_per_gate'][g] += entropy
            stored_per_x['pow_entropy_per_gate'][g] += entropy_pow
            stored_metrics['ece_per_gate'][g] += cal

        correctly_classified += pred_final_head.eq(
            targets)  # getting all the corrects we can
        stored_metrics['cheating_correct'] += correctly_classified.sum().item()


        if training_phase == TrainingPhase.CLASSIFIER:
            num_exits_per_gate = things_of_interest['num_exits_per_gate']
            gated_y_logits = things_of_interest['gated_y_logits']
            _, predicted = gated_y_logits.max(1)
        
            total_cost = compute_cost(num_exits_per_gate, G)
            stored_metrics['total_cost'] += total_cost
            for g in range(G):
                stored_metrics['num_per_gate'][g] += free(num_exits_per_gate[g])
        
        
        

    return stored_per_x, stored_metrics

def compute_cost(num_exits_per_gate, G):
    cost_per_gate = [
        free(num) * (g + 1) / (G+1)
        for g, num in enumerate(num_exits_per_gate)
    ]
    # the last cost_per gate should be equal to the last num
    return  np.sum(cost_per_gate)
def evaluate_with_fixed_gating(thresholds, outputs_logits, intermediate_outputs,
                         targets, stored_metrics, thresh_type):
    G = len(thresholds)
    
    points_reminding = list(range(targets.shape[0]))  # index of all points to classify
   
    gated_outputs = torch.full(outputs_logits.shape,-1.0).to(outputs_logits.device) # outputs storage

    num_classifiction_per_gates = []
    for g, thresh in enumerate(thresholds):
        
        indices_passing_threshold = return_ind_thrs(intermediate_outputs[g], thresh, thresh_type=thresh_type)
            
        
        actual_early_exit_ind = []
        for ind in indices_passing_threshold:
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
    num_classifiction_per_gates.append(points_reminding)
    total_cost = compute_cost(num_classifiction_per_gates, G)
    _, gated_pred = gated_outputs.max(1)
    gated_correct = gated_pred.eq(targets).sum().item()
    stored_metrics['gated_correct'] += gated_correct
    #stored_metrics['cost_per_gate'] += cost_per_gate
    stored_metrics['total_cost'] += total_cost
    return stored_metrics



def get_empty_storage_metrics(num_gates):
    stored_per_x = {
        "entropy_per_gate": [[] for _ in range(num_gates)],
        "pow_entropy_per_gate": [[] for _ in range(num_gates)],
        "p_max_per_gate": [[] for _ in range(num_gates)],
        'list_correct_per_gate': [[] for _ in range(num_gates)],
        'margins_per_gate' : [[] for _ in range(num_gates)],
        'final_entropy': [],
        'final_pow_entropy': [],
        'final_p_max': [],
        'final_margins': []
    }
    stored_metrics = {
        'gated_acc': 0, # using early exiting (computed based on where a point exits)
        'final_ece': 0,
        'gated_correct': 0,
        'final_head_correct_all': 0, # computed on all points including the ones that early exited before
        'total_cost': 0,
        'cheating_correct': 0,
        'num_per_gate': [0 for _ in range(num_gates)],
        'cost_per_gate': [0 for _ in range(num_gates)],
        'ece_per_gate': [0 for _ in range(num_gates)],
        'correct_per_gate': [0 for _ in range(num_gates)],
        'correct_cheating_per_gate': [0 for _ in range(num_gates)]
    }
    return stored_per_x, stored_metrics

