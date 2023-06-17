import numpy as np 
from plotting_util import generate_thresholding_plots
def compute_all_threshold_strategy(data_name, stored_per_x, stored_metrics, target_acc):
    threhsold_name = 'p_max_'+ data_name
    pmax_threshold = compute_optimal_threshold(
        threhsold_name,
        stored_per_x['p_max_per_gate'],
        stored_per_x['list_correct_per_gate'],
        target_acc=target_acc/100.)
    stored_metrics['optim_threshold_pmax'] = pmax_threshold

    threhsold_name = 'entropy_'+ data_name
    H_threshold = compute_optimal_threshold(
        threhsold_name,
        stored_per_x['entropy_per_gate'],
        stored_per_x['list_correct_per_gate'],
        target_acc=target_acc / 100., high_is_conf=False)
    stored_metrics['optim_threshold_entropy'] = H_threshold

    threhsold_name = 'margin_'+ data_name
    margins_threshold = compute_optimal_threshold(
        threhsold_name,
        stored_per_x['margins_per_gate'],
        stored_per_x['list_correct_per_gate'],
        target_acc=target_acc / 100.)
    stored_metrics['optim_threshold_margins'] = margins_threshold



# define a threshold on the confidence values such that each layer tries to hit the target accuracy.
# if high_is_conf, the max value is 1, else the min value is 0.
def compute_optimal_threshold(threhsold_name, confidence, list_correct_gate, target_acc=1, high_is_conf=True):
    list_optimal_threshold = []
    min_x = 10  # min x to average the accuracy
    # store things for plots
    all_sorted_conf = []
    all_cumul_acc = []
    all_correct = []

    for g, confidence_per_gate in enumerate(confidence): # for each gates
        correct = list_correct_gate[g]  # all correclty classified x at the gate
        if high_is_conf:
            unc_values_ind = np.argsort(confidence_per_gate)[::-1]  # argsort the unc_values high to low 
            assert max(confidence_per_gate) <=1 # later we assume that the max possible value is 1
        else:
            unc_values_ind = np.argsort(confidence_per_gate)  # argsort the unc_values low to high
            assert min(confidence_per_gate) >= 0 # later we assume that the min possible value is 0
        sorted_correct = np.array(correct)[unc_values_ind] # sort the correct matching the confidence  => [1, 1, 0.... 1, 0]
        sorted_conf_values = np.array(confidence_per_gate)[ unc_values_ind]  # sort the correct matching the conf => [0.96, 0.9, .... 0.4, 0.1]
        
        cumall_correct = np.cumsum(sorted_correct) 
        cumul_acc = [c / (i +1) for i, c in enumerate(cumall_correct)]  # get the accuracy at each threshold [1,0.9,...0.3]
        
        # store things for plots
        all_sorted_conf.append(list(sorted_conf_values))
        all_cumul_acc.append(cumul_acc)
        all_correct.append(list(sorted_correct))

         
        cumul_acc = cumul_acc[min_x:] # cut the first points to avoid variance issue when averaging 
        
        indices_target_acc = np.argwhere(np.array(cumul_acc)>target_acc) # get all threshold with higher acc than target:
        """
        target_acc = 0.5
        cumul_acc = [0.8, 0.7,| 0.3, 0.3, 0.4]
        indices_target_acc = [0,1]
        """
        
        if len(indices_target_acc) == 0: # if no one can hit the accuracy, we set the threshold to the max conf value
            if high_is_conf:
                threshold_g = 1 # 1 is the most confident value (p_max, margins)
            else:
                threshold_g = 0 # 0 is the most confident value (entropy)
        else:
            optimal_index = int(indices_target_acc[-1]) + min_x # we get the last threshold that has higher acc 
            threshold_g = sorted_conf_values[optimal_index]
        list_optimal_threshold.append(threshold_g)

    generate_thresholding_plots(threhsold_name, all_sorted_conf, all_cumul_acc, all_correct, min_x, target_acc, list_optimal_threshold)
    return list_optimal_threshold