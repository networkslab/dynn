import torch
from metrics_utils import check_hamming_vs_acc
from models.t2t_vit import TrainingPhase, Boosted_T2T_ViT
from torch import nn
import numpy as np

criterion = nn.CrossEntropyLoss()

COMPUTE_HAMMING = False


def get_loss(inputs, targets, optimizer, net):

    optimizer.zero_grad()
    final_logits, intermediate_logits, intermediate_codes = net(inputs)
    loss = criterion(
        final_logits,
        targets)  # the grad_fn of this loss should be None if frozen
    for intermediate_logit in intermediate_logits:
        intermediate_loss = criterion(intermediate_logit, targets)
        loss += intermediate_loss
    things_of_interest = {
        'intermediate_logits': intermediate_logits,
        'final_logits': final_logits}
    if COMPUTE_HAMMING:
        inc_inc_H_list, inc_inc_H_list_std, c_c_H_list, c_c_H_list_std,c_inc_H_list,c_inc_H_list_std = check_hamming_vs_acc(
            intermediate_logits, intermediate_codes, targets)
        things_of_interest = things_of_interest| {
            'inc_inc_H_list': inc_inc_H_list,
            'c_c_H_list': c_c_H_list,
            'c_inc_H_list': c_inc_H_list,
            'inc_inc_H_list_std': inc_inc_H_list_std,
            'c_c_H_list_std': c_c_H_list_std,
            'c_inc_H_list_std': c_inc_H_list_std
        }
    return loss, things_of_interest


def get_dumb_loss(inputs, targets, optimizer, net):
    optimizer.zero_grad()
    y_pred, ic_cost, intermediate_outputs = net.module.forward_brute_force(
        inputs, normalize=True)
    loss_performance = criterion(y_pred, targets)

    loss = loss_performance + net.module.cost_perf_tradeoff * torch.sum(
        ic_cost)
    return loss, y_pred, intermediate_outputs


def get_surrogate_loss(inputs, targets, optimizer, net, training_phase=None):
    if net.training:
        optimizer.zero_grad()
        loss = None
        # TODO Move training phase enum to this file.
        if training_phase == TrainingPhase.CLASSIFIER:
            gated_y_logits, things_of_interest = net.module.surrogate_forward(
                inputs, targets, training_phase=training_phase)
            loss = criterion(gated_y_logits, targets)
        elif training_phase == TrainingPhase.GATE:
            loss, things_of_interest = net.module.surrogate_forward(
                inputs, targets, training_phase=training_phase)

    else:
        gated_y_logits, things_of_interest = net.module.surrogate_forward(
            inputs, targets, training_phase=TrainingPhase.CLASSIFIER)
        classifier_loss = criterion(gated_y_logits, targets)
        things_of_interest['gated_y_logits'] = gated_y_logits
        gate_loss, things_of_interest_gate = net.module.surrogate_forward(
            inputs, targets, training_phase=TrainingPhase.GATE)
        loss = (gate_loss + classifier_loss) / 2
        things_of_interest.update(things_of_interest_gate)
    return loss, things_of_interest


def freeze_backbone(network, excluded_submodules: list[str]):
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    total_num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    # set everything to not trainable.
    for param in network.module.parameters():
        param.requires_grad = False

    for submodule_attr_name in excluded_submodules:  # Unfreeze excluded submodules to be trained.
        for submodule in getattr(network.module, submodule_attr_name):
            for param in submodule.parameters():
                param.requires_grad = True

    trainable_parameters = filter(lambda p: p.requires_grad,
                                  network.parameters())
    num_trainable_params = sum(
        [np.prod(p.size()) for p in trainable_parameters])
    print('Successfully froze network: from {} to {} trainable params.'.format(
        total_num_parameters, num_trainable_params))


def get_boosted_loss(inputs, targets, optimizer, net):
    # assert isinstance(net, Boosted_T2T_ViT), 'Boosted loss only available for boosted t2t vit'
    n_blocks = len(net.blocks)
    optimizer.zero_grad()

    # Ensembling
    preds, pred_ensembles = net.forward_all(inputs, n_blocks - 1)
    loss_all = 0
    for stage in range(n_blocks):
        # train weak learner
        # fix F
        with torch.no_grad():
            if not isinstance(pred_ensembles[stage], torch.Tensor):
                out = torch.unsqueeze(torch.Tensor([pred_ensembles[stage]]), 0)  # 1x1
                out = out.expand(inputs.shape[0], net.num_classes).cuda()
            else:
                out = pred_ensembles[stage]
            out = out.detach()
        loss = criterion(preds[stage] + out, targets)
        loss_all = loss_all + loss
    return loss_all
