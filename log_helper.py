import numpy as np


def log_metrics_mlflow(prefix_logger, acc, loss, G, stored_per_x,stored_metrics, total):
    cheating_acc = 100. * stored_metrics['cheating_correct'] / total
    ece = stored_metrics['ece'] / total
    entropy = np.mean(stored_per_x['final_entropy']) 
    log_dict = {
                prefix_logger+'/loss': loss,
                prefix_logger+'/acc': acc,
                prefix_logger+'/ece': ece,
                prefix_logger+'/cheating_acc': cheating_acc,
                prefix_logger+'/entropy': entropy
    }
    for g in range(G):
        acc_gate = 100. * stored_metrics['correct_per_gate'][g] / total
        acc_cheating_gate = 100. * stored_metrics[
            'correct_cheating_per_gate'][g] / total
        entropy_per_gate = np.mean(stored_per_x['entropy_per_gate'][g])
        ece_gate = stored_metrics['ece_per_gate'][g] / total
        log_dict[prefix_logger+'/acc' + str(g)] = acc_gate
        log_dict[prefix_logger+'/cheating_acc' + str(g)] = acc_cheating_gate
        log_dict[prefix_logger+'/entropy' + str(g)] = entropy_per_gate
        log_dict[prefix_logger+'/ece' + str(g)] = ece_gate
    return log_dict



