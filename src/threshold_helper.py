from enum import Enum
import numpy as np
from learning_helper import LearningHelper
from plotting_util import generate_thresholding_plots
from metrics_utils import compute_detached_uncertainty_metrics
import torch
import math
import mlflow

def aggregate_logits_targets(loader, helper: LearningHelper, device):
    helper.net.eval()
    all_targets = []
    all_preds = []
    for _, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        _, things_of_interest, _ = helper.get_warmup_loss(inputs, targets)
        logits_per_gates = things_of_interest['intermediate_logits']+ [things_of_interest['final_logits']]
        all_preds.append(torch.stack(logits_per_gates))
        all_targets.append(targets)
    return torch.concatenate(all_preds, dim=1), torch.concatenate(all_targets)


def fixed_threshold_test(args, helper: LearningHelper, device, test_loader, val_loader):
    val_pred, val_target = aggregate_logits_targets(val_loader, helper, device)
    test_pred, test_target = aggregate_logits_targets(test_loader, helper, device)

    costs_at_exit = helper.net.module.normalized_cost_per_exit

    costs = []
    accs = []
    # for p in range(1, 100):
    for p in range(1, 40):
        #print("*********************")
        _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
        
        probs = torch.exp(torch.log(_p) * torch.range(1, 7))
        probs /= probs.sum()
        acc_val, _, T = dynamic_eval_find_threshold(val_pred, val_target, probs, costs_at_exit)
        acc_test, exp_cost, metrics_dict = dynamic_eval_with_threshold(test_pred, test_target, costs_at_exit, T)
        mlflow_dict = get_ml_flow_dict(metrics_dict)
        #print('valid acc: {:.3f}, test acc: {:.3f}, test cost: {:.2f}%'.format(acc_val, acc_test, exp_cost))
        mlflow.log_metrics(mlflow_dict, step=p)
        costs.append(exp_cost)
        accs.append(acc_test)
    return costs, accs

def get_ml_flow_dict(dict):
    mlflow_dict = {}
    for k, v in dict.items():
        if isinstance(v, list):
            for i in range(len(v)):
                mlflow_dict[f'{k}_{i}'] = v[i]
        else:
            mlflow_dict[k] = v
    return mlflow_dict

def dynamic_eval_with_threshold(logits, targets, flops, thresholds):
        metrics_dict = {}
        n_stage, n_sample, _ = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take the max logits as confidence

        acc_rec, exit_count = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        correct_all = [0 for _ in range(n_stage)]
        acc_gated = []
        for i in range(n_sample):
            has_exited = False
            gold_label = targets[i]
            for k in range(n_stage):
                # compute acc over all samples, regardless of exit
                classifier_pred = int(argmax_preds[k][i].item())
                target = int(gold_label.item())
                if classifier_pred == target:
                    correct_all[k] += 1
                if max_preds[k][i].item() >= thresholds[k] and not has_exited: # exit at k
                    if target == classifier_pred:
                        acc += 1
                        acc_rec[k] += 1
                    exit_count[k] += 1 # keeps track of number of exits per gate
                    has_exited = True # keep on looping but only for computing correct_all
        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exit_count[k] * 1.0 / n_sample
            sample_all += exit_count[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]
        exit_rate = []
        for i in range(n_stage):
            acc_gated.append((acc_rec[i] / exit_count[i] * 100).item())
            correct_all[i] = correct_all[i] / n_sample * 100
            exit_rate.append(exit_count[i].item() / n_sample * 100)
        metrics_dict['GATED_ACC_PER_GATE'] = acc_gated
        metrics_dict['ALL_ACC_PER_GATE'] = correct_all
        metrics_dict['EXIT_RATE_PER_GATE'] = exit_rate
        acc = acc * 100.0 / n_sample
        metrics_dict['ACC'] = acc
        metrics_dict['EXPECTED_FLOPS'] = expected_flops.item()

        return acc * 100.0 / n_sample, expected_flops, metrics_dict


def dynamic_eval_find_threshold(logits, targets, p, flops):
        """
            logits: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.size()

        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)
        T = torch.Tensor(n_stage).fill_(1e8)

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage -1] = -1e8 # accept all of the samples at the last stage

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T

class THRESH(Enum):
    PMAX = 1
    MARGINS = 2
    ENTROPY = 3
    POWENTROPY = 4


def compute_all_threshold_strategy(data_name, stored_per_x, stored_metrics,
                                   target_acc):
    threhsold_name = 'p_max_' + data_name
    pmax_threshold = compute_optimal_threshold(
        threhsold_name,
        stored_per_x['p_max_per_gate'],
        stored_per_x['list_correct_per_gate'],
        target_acc=target_acc / 100.)
    stored_metrics['optim_threshold_pmax'] = pmax_threshold

    threhsold_name = 'entropy_' + data_name
    H_threshold = compute_optimal_threshold(
        threhsold_name,
        stored_per_x['entropy_per_gate'],
        stored_per_x['list_correct_per_gate'],
        target_acc=target_acc / 100.,
        high_is_conf=False)
    stored_metrics['optim_threshold_entropy'] = H_threshold

    threhsold_name = 'margin_' + data_name
    margins_threshold = compute_optimal_threshold(
        threhsold_name,
        stored_per_x['margins_per_gate'],
        stored_per_x['list_correct_per_gate'],
        target_acc=target_acc / 100.)
    stored_metrics['optim_threshold_margins'] = margins_threshold

    threhsold_name = 'entropy_pow_' + data_name
    entropy_pow_threshold = compute_optimal_threshold(
        threhsold_name,
        stored_per_x['pow_entropy_per_gate'],
        stored_per_x['list_correct_per_gate'],
        target_acc=target_acc / 100.,
        high_is_conf=False)
    stored_metrics['optim_threshold_entropy_pow'] = entropy_pow_threshold


def return_ind_thrs(outputs, thresh, thresh_type):
    p_max, entropy, _, margins, entropy_pow = compute_detached_uncertainty_metrics(
        outputs, None)
    if thresh_type == THRESH.PMAX:
        indices_pass_threshold = list(
            np.argwhere(np.array(p_max) >= thresh).flatten())
    elif thresh_type == THRESH.ENTROPY:
        indices_pass_threshold = list(
            np.argwhere(np.array(entropy) <= thresh).flatten())
    elif thresh_type == THRESH.MARGINS:
        indices_pass_threshold = list(
            np.argwhere(np.array(margins) >= thresh).flatten())
    if thresh_type == THRESH.POWENTROPY:
        indices_pass_threshold = list(
            np.argwhere(np.array(entropy_pow) <= thresh).flatten())
    return indices_pass_threshold


# define a threshold on the confidence values such that each layer tries to hit the target accuracy.
# if high_is_conf, the max value is 1, else the min value is 0.
def compute_optimal_threshold(threhsold_name,
                              confidence,
                              list_correct_gate,
                              target_acc=1,
                              high_is_conf=True):
    list_optimal_threshold = []
    min_x = 10  # min x to average the accuracy
    # store things for plots
    all_sorted_conf = []
    all_cumul_acc = []
    all_correct = []

    for g, confidence_per_gate in enumerate(confidence):  # for each gates
        correct = list_correct_gate[
            g]  # all correclty classified x at the gate
        if high_is_conf:
            unc_values_ind = np.argsort(
                confidence_per_gate)[::
                                     -1]  # argsort the unc_values high to low
            assert max(
                confidence_per_gate
            ) <= 1  # later we assume that the max possible value is 1
        else:
            unc_values_ind = np.argsort(
                confidence_per_gate)  # argsort the unc_values low to high
            assert min(
                confidence_per_gate
            ) >= 0  # later we assume that the min possible value is 0
        sorted_correct = np.array(
            correct
        )[unc_values_ind]  # sort the correct matching the confidence  => [1, 1, 0.... 1, 0]
        sorted_conf_values = np.array(
            confidence_per_gate
        )[unc_values_ind]  # sort the correct matching the conf => [0.96, 0.9, .... 0.4, 0.1]

        cumall_correct = np.cumsum(sorted_correct)
        cumul_acc = [c / (i + 1) for i, c in enumerate(cumall_correct)
                     ]  # get the accuracy at each threshold [1,0.9,...0.3]

        # store things for plots
        all_sorted_conf.append(list(sorted_conf_values))
        all_cumul_acc.append(cumul_acc)
        all_correct.append(list(sorted_correct))

        cumul_acc = cumul_acc[
            min_x:]  # cut the first points to avoid variance issue when averaging

        indices_target_acc = np.argwhere(
            np.array(cumul_acc)
            > target_acc)  # get all threshold with higher acc than target:
        """
        target_acc = 0.5
        cumul_acc = [0.8, 0.7,| 0.3, 0.3, 0.4]
        indices_target_acc = [0,1]
        """

        if len(
                indices_target_acc
        ) == 0:  # if no one can hit the accuracy, we set the threshold to the max conf value
            if high_is_conf:
                threshold_g = 1  # 1 is the most confident value (p_max, margins)
            else:
                threshold_g = 0  # 0 is the most confident value (entropy)
        else:
            optimal_index = int(
                indices_target_acc[-1]
            ) + min_x  # we get the last threshold that has higher acc
            threshold_g = sorted_conf_values[optimal_index]
        list_optimal_threshold.append(threshold_g)

    generate_thresholding_plots(threhsold_name, all_sorted_conf, all_cumul_acc,
                                all_correct, min_x, target_acc,
                                list_optimal_threshold)
    return list_optimal_threshold