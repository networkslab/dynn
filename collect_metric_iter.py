# Training

from utils import progress_bar
import torch
import mlflow
import scipy
import numpy as np
from sklearn import calibration


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
    stored_metrics['ECE'] +=cal
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
        stored_metrics['ece_per_gate'][g] +=cal

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
        p_max, _, cal = compute_uncertainty_metrics(intermediate_outputs[g],
                                                    targets)
        early_exit_ind = list(np.argwhere(p_max > thresh).flatten())
        actual_early_exit_ind = []
        for ind in early_exit_ind:
            if ind in x_index:  # if that index hasnt been classified yet by an earlier gates
                actual_early_exit_ind.append(ind)  # we classify it
                x_index.remove(
                    ind)  # we remove that index to be classified in the future

        num_classifiction_per_gates.append(len(actual_early_exit_ind))
        if len(actual_early_exit_ind) > 0:
            gated_outputs[actual_early_exit_ind, :] = intermediate_outputs[g][
                actual_early_exit_ind, :]
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
    #stored_metrics['cost_per_gate'] += cost_per_gate
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
        'ECE':0,
        'gated_correct': 0,
        'total_cost': 0,
        'cheating_correct': 0,
        'cost_per_gate': [0 for _ in range(num_gates)],
        'ece_per_gate': [0 for _ in range(num_gates)],
        'correct_per_gate': [0 for _ in range(num_gates)],
        'correct_cheating_per_gate': [0 for _ in range(num_gates)]
    }
    return stored_per_x, stored_metrics


from sklearn.calibration import column_or_1d, check_consistent_length, _check_pos_label_consistency


# straigh up stolen from sklearn
def calibration_curve(
    y_true,
    y_prob,
    *,
    pos_label=None,
    normalize="deprecated",
    n_bins=5,
    strategy="uniform",
):
    """Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Calibration curves may also be referred to as reliability diagrams.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

    pos_label : int or str, default=None
        The label of the positive class.

        .. versionadded:: 1.1

    normalize : bool, default="deprecated"
        Whether y_prob needs to be normalized into the [0, 1] interval, i.e.
        is not a proper probability. If True, the smallest value in y_prob
        is linearly mapped onto 0 and the largest one onto 1.

        .. deprecated:: 1.1
            The normalize argument is deprecated in v1.1 and will be removed in v1.3.
            Explicitly normalizing `y_prob` will reproduce this behavior, but it is
            recommended that a proper probability is used (i.e. a classifier's
            `predict_proba` positive class).

    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).

    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.calibration import calibration_curve
    >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
    >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    >>> prob_true
    array([0. , 0.5, 1. ])
    >>> prob_pred
    array([0.2  , 0.525, 0.85 ])
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy "
                         "must be either 'quantile' or 'uniform'.")

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    ece = np.sum(
        np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    return prob_true, prob_pred, ece