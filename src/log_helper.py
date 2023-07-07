import numpy as np
import mlflow
from data_loading.data_loader_helper import get_abs_path

def log_metrics_mlflow(prefix_logger, gated_acc, loss, G, stored_per_x, stored_metrics, total, total_classifier):
    
    
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
    return log_dict

def compute_gated_accuracy(stored_metrics, gate_idx):
    pred_count = stored_metrics['gated_pred_count_per_gate'][gate_idx]
    correct_count = stored_metrics['gated_correct_count_per_gate'][gate_idx]
    if pred_count == 0:
        return 0
    return correct_count / pred_count * 100

def setup_mlflow(run_name: str, cfg):
    print(run_name)
    project = 'DyNN_code'
    mlruns_path = get_abs_path(["mlruns"])
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(project)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(cfg)


