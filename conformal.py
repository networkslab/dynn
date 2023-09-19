def get_all_pred_sets(logits, threshold_dict):
    score = torch.nn.functional.softmax(logits, dim=1)
    pred_sets = {}
    for alpha, qhat in threshold_dict.items():
        C_set = score >= (1-qhat)
        pred_sets[alpha] = C_set
    return pred_sets

def early_exit_conf_sets(alpha_qhat_dict, sample_exit_level_map,  all_logits, gated_logits):
    gated_pred_sets = {}
    pred_sets_per_gate = {}
    pred_sets_per_gate_strict = {}
    general_pred_sets = {}
    for alpha, dict_qhats in alpha_qhat_dict.items():
        general_pred_sets = get_all_pred_sets(gated_logits, dict_qhats['qhat'])
        gated_prediction_sets = torch.ones_like(all_logits[0]).bool()
        prediction_sets_per_gates = [torch.ones_like(all_logits[0]).bool() for _ in range(G)] # by default we set all points to 1
        for l, conf_thresh  in enumerate(dict_qhats['qhats']):
            prob_at_l = torch.nn.functional.softmax(all_logits[l], dim=1)
            exited_prob_at_l = prob_at_l[sample_exit_level_map == l]
            gated_prediction_sets[sample_exit_level_map == l] = exited_prob_at_l >= (1-conf_thresh)

            # we also store each conf interval that is accessible
            accessible = sample_exit_level_map <= l
            prediction_sets_per_gates[l][accessible] = prob_at_l[accessible] >= (1-conf_thresh)
        gated_pred_sets[alpha] = gated_prediction_sets
        pred_sets_per_gate[alpha] = prediction_sets_per_gates