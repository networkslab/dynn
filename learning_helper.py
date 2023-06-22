import torch
from models.t2t_vit import TrainingPhase
from torch import nn

criterion = nn.CrossEntropyLoss()


def get_loss(inputs, targets, optimizer, net):

    optimizer.zero_grad()
    outputs_logits, intermediate_outputs = net(inputs)
    loss = criterion(
        outputs_logits,
        targets)  # the grad_fn of this loss should be None if frozen
    for intermediate_output in intermediate_outputs:
        intermediate_loss = criterion(intermediate_output, targets)
        loss += intermediate_loss
    return loss, outputs_logits, intermediate_outputs


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
            y_logits, intermediate_logits, final_logits = net.module.surrogate_forward(
                inputs, targets, training_phase=training_phase)
            loss = criterion(y_logits, targets)
        elif training_phase == TrainingPhase.GATE:
            loss, intermediate_logits, final_logits = net.module.surrogate_forward(
                inputs, targets, training_phase=training_phase)
            y_logits = final_logits
    else:
        y_logits, intermediate_logits, final_logits = net.module.surrogate_forward(
                inputs, targets, training_phase=TrainingPhase.CLASSIFIER)
        classifier_loss = criterion(y_logits, targets)
        gate_loss, intermediate_logits, final_logits = net.module.surrogate_forward(
                inputs, targets, training_phase=TrainingPhase.GATE)
        loss = (gate_loss+classifier_loss)/2
    
    return loss, intermediate_logits, y_logits