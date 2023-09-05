import torch
from models.t2t_vit import TrainingPhase
from torch import nn
import numpy as np
from models.classifier_training_helper import LossContributionMode, ClassifierTrainingHelper
from models.gate_training_helper import GateTrainingScheme, GateTrainingHelper

criterion = nn.CrossEntropyLoss()

class LearningHelper:
    def __init__(self, net, optimizer, args, device) -> None:
        self.net = net
        self.optimizer = optimizer
        self._init_classifier_training_helper(args)
        self._init_gate_training_helper(args, device)

    def _init_classifier_training_helper(self, args) -> None:
        gate_selection_mode = args.gate_selection_mode
        self.loss_contribution_mode = args.classifier_loss 
        self.classifier_training_helper = ClassifierTrainingHelper(self.net, gate_selection_mode, self.loss_contribution_mode)
    
    def _init_gate_training_helper(self, args, device) -> None:
        gate_training_scheme = GateTrainingScheme[args.gate_training_scheme]
        self.gate_training_helper = GateTrainingHelper(self.net, gate_training_scheme, args.gate_objective, args.G, device)
    
    

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
        self.optimizer.zero_grad()
        final_logits, intermediate_logits, _ = self.net(inputs)
        loss = criterion(final_logits,targets)  # the grad_fn of this loss should be None if frozen
        
        if self.loss_contribution_mode == LossContributionMode.BOOSTED: # our version of boosting is just training early classifier more
            num_gates = len(intermediate_logits)+1
            for l, intermediate_logit in enumerate(intermediate_logits):
                intermediate_loss = criterion(intermediate_logit, targets)
                loss += (num_gates - l)*intermediate_loss # we scale the gradient by G-l => early gates have bigger gradient
               
        else: # plain optimization of all the intermediate classifiers
            for intermediate_logit in intermediate_logits:
                intermediate_loss = criterion(intermediate_logit, targets)
                loss += intermediate_loss
        things_of_interest = {
            'intermediate_logits': intermediate_logits,
            'final_logits': final_logits}
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
