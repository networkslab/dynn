import numpy as np
import mlflow
from data_loading.data_loader_helper import get_abs_path

def get_display(key, cum_metric):
    if 'correct' in key:
            return  100*np.mean(cum_metric)
    else:
        return np.mean(cum_metric)
def log_aggregate_metrics_mlflow(prefix_logger, metrics_dict, gates_count):
    log_dict = {}
    for metric_key, val in metrics_dict.items():
        cumul_metric, total = val
        if type(cumul_metric) is list: 
            if len(cumul_metric) == gates_count and 'per_gate' in metric_key:# if the length is the number of gates we want to see all of them
                for g, cumul_metric_per_gate in enumerate(cumul_metric):
                    log_dict[prefix_logger+'/'+metric_key+ str(g)]  = get_display(metric_key, cumul_metric_per_gate)/total
            else:
                log_dict[prefix_logger+'/'+metric_key]  = get_display(metric_key, np.mean(cumul_metric))/total
        else:
            log_dict[prefix_logger+'/'+metric_key] = get_display(metric_key, cumul_metric)/total
    return log_dict


# def log_metrics_mlflow(prefix_logger, gated_acc, loss, G, stored_per_x, stored_metrics, total_classifier, batch):
    
    
    cheating_acc = 100. * stored_metrics['cheating_correct'] / total_classifier
    cost = stored_metrics['total_cost'] / total_classifier
    ece = stored_metrics['final_ece'] / total_classifier
    entropy = np.mean(stored_per_x['final_entropy']) 
    log_dict = {
                prefix_logger+'/loss': loss,
                prefix_logger+'/ece': ece,
                prefix_logger+'/cheating_acc': cheating_acc,
                prefix_logger+'/entropy': entropy,
                prefix_logger+'/cost': cost,
                prefix_logger+'/final_head_acc_all': stored_metrics['final_head_correct_all'] / total_classifier
    }
    if gated_acc is not None:
        log_dict[prefix_logger+'/gated_acc'] = gated_acc # using early exiting
    for g in range(G):
        acc_gate = 100. * stored_metrics['correct_per_gate'][g] / total_classifier
        acc_cheating_gate = 100. * stored_metrics[
            'correct_cheating_per_gate'][g] / total_classifier
        entropy_per_gate = np.mean(stored_per_x['entropy_per_gate'][g])
        ece_gate = stored_metrics['ece_per_gate'][g] / total_classifier
        log_dict[prefix_logger+'/acc_all' + str(g)] = acc_gate # including points that were previously exited
        log_dict[prefix_logger+'/cheating_acc' + str(g)] = acc_cheating_gate
        log_dict[prefix_logger+'/entropy' + str(g)] = entropy_per_gate
        log_dict[prefix_logger+'/ece' + str(g)] = ece_gate
        log_dict[prefix_logger+'/percent_exit' + str(g)] = stored_metrics['gated_pred_count_per_gate'][g] / total_classifier * 100
        log_dict[prefix_logger + '/gated_acc' + str(g)] = compute_gated_accuracy(stored_metrics, g)

        
        # incorrect_to_incorrect = batch*stored_metrics['hamming_incinc_per_gate'][g] / total_classifier
        # incorrect_to_correct = batch*stored_metrics['hamming_corinc_per_gate'][g] / total_classifier 
        # correct_to_correct = batch*stored_metrics['hamming_corcor_per_gate'][g] / total_classifier
        # incorrect_to_incorrect_std = batch*stored_metrics['hamming_incinc_per_gate_std'][g] / total_classifier
        # incorrect_to_correct_std = batch*stored_metrics['hamming_corinc_per_gate_std'][g] / total_classifier 
        # correct_to_correct_std = batch*stored_metrics['hamming_corcor_per_gate_std'][g] / total_classifier
        # if not np.isnan(correct_to_correct):
        #     log_dict[prefix_logger + '/hamming_cor' + str(g)] = correct_to_correct
        #     log_dict[prefix_logger + '/hamming_cor_std' + str(g)] = correct_to_correct_std
        # if not np.isnan(incorrect_to_incorrect):
        #     log_dict[prefix_logger + '/hamming_inc' + str(g)] = incorrect_to_incorrect
        #     log_dict[prefix_logger + '/hamming_inc_std' + str(g)] = incorrect_to_incorrect_std
        # if not np.isnan(correct_to_correct) and not np.isnan(incorrect_to_incorrect):
        #     log_dict[prefix_logger + '/hamming_corinc' + str(g)] = incorrect_to_correct
        #     log_dict[prefix_logger + '/hamming_corinc_std' + str(g)] = incorrect_to_correct_std


    return log_dict


def setup_mlflow(run_name: str, cfg, experiment_name):
    print(run_name)
    project = experiment_name
    mlruns_path = get_abs_path(["mlruns"])
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(project)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(cfg)


