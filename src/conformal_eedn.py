import numpy as np
import torch

from utils import free 



def compute_conf_threshold(mixed_score, scores_per_gate, all_score_per_gates):
    MIN_POINTS_FOR_CONF = 20
    alpha_confs = [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05]    
    alpha_qhat_dict = {'qhats': {}, 'qhat':{}, "qhats_all":{}, "qhats_per_gate":{}}
    for alpha_conf in alpha_confs:
        qhat_general = get_conf_thresh(mixed_score, alpha_conf)
        qhats = []
        qhats_all = []
        qhats_per_gate = []
        for l, scores_in_l in enumerate(scores_per_gate):
            all_scores_in_l = all_score_per_gates[l]
            qhat_all_l = get_conf_thresh(all_scores_in_l, alpha_conf)
            qhat_per_gate = get_conf_thresh(scores_in_l, alpha_conf)
            # for the main one, we use qhat_per_gate if we have enough point else we take the one over all points.
            if len(scores_in_l) > MIN_POINTS_FOR_CONF :
                qhat = qhat_per_gate
            else:
                qhat = qhat_all_l

            qhats_per_gate.append(qhat_per_gate)
            qhats.append(qhat)
            qhats_all.append(qhat_all_l)
        alpha_qhat_dict['qhats'][alpha_conf] = qhats
        alpha_qhat_dict['qhat'][alpha_conf] =  qhat_general
        alpha_qhat_dict['qhats_all'][alpha_conf] = qhats_all
        alpha_qhat_dict['qhats_per_gate'][alpha_conf] =  qhats_per_gate

    return alpha_qhat_dict

def get_conf_thresh(list_scores, alpha_conf):

    n = len(list_scores)
    if n == 0 :
        return 1-alpha_conf # random value

    q_level = np.ceil((n+1)*(1-alpha_conf))/n
    if q_level>1:
        q_level = 1
    return np.quantile(list_scores, q_level, method='higher')
    
def get_pred_sets(logits, qhat):
    score = torch.nn.functional.softmax(logits, dim=1)
    C_set = score >= (1-qhat)
    return C_set

def early_exit_conf_sets(alpha_qhat_dict, sample_exit_level_map,  all_logits, gated_logits):
    sets_gated = {}
    sets_gated_all = {}
    sets_gated_strict = {}
    sets_general = {}
    G = len(all_logits)

    dict_sets = {'sets_'+str(l): {} for l in range(G)}

    
    for alpha in  alpha_qhat_dict['qhats'].keys():
        sets_general[alpha] = get_pred_sets(gated_logits, alpha_qhat_dict['qhat'][alpha])
        
        sets_holder_gated = torch.ones_like(all_logits[0]).bool()
        sets_holder_all = torch.ones_like(all_logits[0]).bool()
        sets_holder_gated_strict = torch.ones_like(all_logits[0]).bool()
        
        for l  in range(G):
            logits_at_l = all_logits[l]
            exited_t_l = sample_exit_level_map == l
            exited_prob_at_l = logits_at_l[exited_t_l]
            sets_holder_gated[exited_t_l] = get_pred_sets(exited_prob_at_l, alpha_qhat_dict['qhats'][alpha][l])
            sets_holder_all[exited_t_l] = get_pred_sets(exited_prob_at_l, alpha_qhat_dict['qhats_all'][alpha][l])
            sets_holder_gated_strict[exited_t_l] = get_pred_sets(exited_prob_at_l, alpha_qhat_dict['qhats_per_gate'][alpha][l])
            all_for_l = get_pred_sets(logits_at_l, alpha_qhat_dict['qhats_all'][alpha][l])
            dict_sets['sets_'+str(l)][alpha] = all_for_l
            
        sets_gated[alpha] = sets_holder_gated
        sets_gated_all[alpha] = sets_holder_all
        sets_gated_strict[alpha] = sets_holder_gated_strict
        
    dict_sets['sets_general'] = sets_general
    dict_sets['sets_gated'] = sets_gated
    dict_sets['sets_gated_all'] = sets_gated_all
    dict_sets['sets_gated_strict'] = sets_gated_strict
            
    return dict_sets

def compute_coverage_and_inef(conf_sets, targets):
    C = np.mean(np.sum(free(conf_sets.float()), axis=1))
    in_gated_conf = conf_sets[np.arange(targets.shape[0]),targets]
    emp_alpha = 1- np.mean(free(in_gated_conf))
    return C, emp_alpha

    