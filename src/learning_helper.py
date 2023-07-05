import torch
from models.t2t_vit import TrainingPhase
from torch import nn
import numpy as np
from enum import Enum


criterion = nn.CrossEntropyLoss()

def get_loss(inputs, targets, optimizer, net):

    optimizer.zero_grad()
    final_logits, intermediate_logits = net(inputs)
    loss = criterion(
        final_logits,
        targets)  # the grad_fn of this loss should be None if frozen
    for intermediate_logit in intermediate_logits:
        intermediate_loss = criterion(intermediate_logit, targets)
        loss += intermediate_loss
    things_of_interest = {'intermediate_logits':intermediate_logits,'final_logits':final_logits}
    return loss, things_of_interest


def get_dumb_loss(inputs, targets, optimizer, net):
    optimizer.zero_grad()
    y_pred, ic_cost, intermediate_outputs = net.module.forward_brute_force(
        inputs, normalize=True)
    loss_performance = criterion(y_pred, targets)

    loss = loss_performance + net.module.cost_perf_tradeoff * torch.sum(
        ic_cost)
    return loss, y_pred, intermediate_outputs


def get_surrogate_loss(inputs, targets, optimizer, net,
                       training_phase=None):
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
                inputs, targets, training_phase = TrainingPhase.CLASSIFIER)
        classifier_loss = criterion(gated_y_logits, targets)
        things_of_interest['gated_y_logits'] = gated_y_logits
        gate_loss, things_of_interest_gate = net.module.surrogate_forward(
                inputs, targets, training_phase = TrainingPhase.GATE)
        loss = (gate_loss + classifier_loss) / 2
        things_of_interest.update(things_of_interest_gate)
    return loss, things_of_interest

def freeze_backbone(network, excluded_submodules: list[str]):
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    total_num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    # set everything to not trainable.
    for param in network.module.parameters():
        param.requires_grad = False


    for submodule_attr_name in excluded_submodules: # Unfreeze excluded submodules to be trained.
        for submodule in getattr(network.module, submodule_attr_name):
            for param in submodule.parameters():
                param.requires_grad = True

    trainable_parameters = filter(lambda p: p.requires_grad, network.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in trainable_parameters])
    print('Successfully froze network: from {} to {} trainable params.'.format(total_num_parameters,num_trainable_params))
