import torch
from metrics_utils import check_hamming_vs_acc
from models.t2t_vit import TrainingPhase
from torch import nn
import numpy as np
from models.classifier_training_helper import LossContributionMode, ClassifierTrainingHelper
from models.gate_training_helper import GateTrainingScheme, GateTrainingHelper

criterion = nn.CrossEntropyLoss()

COMPUTE_HAMMING = False

class LearningHelper:
    def __init__(self, net, optimizer, args) -> None:
        self.net = net
        self.optimizer = optimizer
        self._init_classifier_training_helper(args)
        self._init_gate_training_helper(args)

    def _init_classifier_training_helper(self, args) -> None:
        gate_selection_mode = args.gate_selection_mode
        loss_contribution_mode = LossContributionMode.WEIGHTED if args.weighted else LossContributionMode.SINGLE
        self.classifier_training_helper = ClassifierTrainingHelper(self.net, gate_selection_mode, loss_contribution_mode)
    
    def _init_gate_training_helper(self, args) -> None:
        gate_training_scheme = GateTrainingScheme[args.gate_training_scheme]
        self.gate_training_helper = GateTrainingHelper(self.net, gate_training_scheme, args.gate_objective)
    

    def get_surrogate_loss(self, inputs, targets, training_phase=None):
        if self.net.training:
            self.optimizer.zero_grad()
            if training_phase == TrainingPhase.CLASSIFIER:
                return self.classifier_training_helper.get_loss(inputs, targets)
            elif training_phase == TrainingPhase.GATE:
                return self.gate_training_helper.get_loss(inputs, targets)
        else:
            with torch.no_grad():
                classifier_loss, things_of_interest = self.classifier_training_helper.get_loss(inputs, targets)
                gate_loss, things_of_interest_gate = self.gate_training_helper.get_loss(inputs, targets)
                loss = (gate_loss + classifier_loss) / 2
                things_of_interest.update(things_of_interest_gate)
                return loss, things_of_interest


    def get_warmup_loss(self, inputs, targets):
        criterion = nn.CrossEntropyLoss()
        if self.net.training:
            self.optimizer.zero_grad()
            
            final_logits, intermediate_logits, intermediate_codes = self.net(inputs)
            loss = criterion(
                final_logits,
                targets)  # the grad_fn of this loss should be None if frozen
            i = len(intermediate_logits)+1
            for intermediate_logit in intermediate_logits:
                intermediate_loss = criterion(intermediate_logit, targets)
                loss += i*intermediate_loss
                i-=1
        else:
            with torch.no_grad():
                final_logits, intermediate_logits, intermediate_codes = self.net(inputs)
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
