import torch 
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



def get_dumb_loss(inputs, targets, optimizer, criterion, net):
    optimizer.zero_grad()
    y_pred, ic_cost, intermediate_outputs = net.module.forward_brute_force(inputs)
    loss_performance = criterion(y_pred, targets)

    loss  = loss_performance + net.module.cost_perf_tradeoff * torch.sum(ic_cost)
    return loss, y_pred, intermediate_outputs