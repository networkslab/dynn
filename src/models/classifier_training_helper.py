import torch
import torch.nn as nn
from queue import Queue
import random 
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from metrics_utils import check_hamming_vs_acc
from models.custom_modules.gate import GateType
from .token_transformer import Token_transformer
from .token_performer import Token_performer
from .transformer_block import Block, get_sinusoid_encoding
from .custom_modules.custom_GELU import CustomGELU
from .custom_modules.learnable_uncertainty_gate import LearnableUncGate
from .custom_modules.learnable_code_gate import LearnableCodeGate
from .custom_modules.learnable_complex_gate import LearnableComplexGate
from sklearn.metrics import accuracy_score
from enum import Enum

class GateSelectionMode(Enum):
    PROBABILISTIC = 'prob'
    DETERMINISTIC = 'det'

class LossContributionMode(Enum):
    SINGLE = 'single' # a sample contributes to the loss at a single classifier
    WEIGHTED = 'weighted'

class InvalidLossContributionModeException(Exception):
    pass

class ClassifierTrainingHelper:
    def __init__(self, net: nn.Module, gate_selection_mode: GateSelectionMode, loss_contribution_mode: LossContributionMode) -> None:
        self.net = net
        self.gate_selection_mode = gate_selection_mode
        self.loss_contribution_mode = loss_contribution_mode
        if self.loss_contribution_mode == LossContributionMode.WEIGHTED:
            self.classifier_criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.classifier_criterion = nn.CrossEntropyLoss()

    def get_loss(self, inputs: torch.Tensor, targets: torch.tensor, compute_hamming = False):
        intermediate_logits = [] # logits from the intermediate classifiers
        num_exits_per_gate = []
        final_head, intermediate_outs, intermediate_codes = self.net.forward_features(inputs)
        final_logits = self.net.head(final_head)
        prob_gates = torch.zeros((inputs.shape[0], 1)).to(inputs.device)
        gated_y_logits = torch.zeros_like(final_logits) # holds the accumulated predictions in a single tensor
        sample_exit_level_map = torch.zeros_like(targets) # holds the exit level of each prediction
        past_exits = torch.zeros((inputs.shape[0], 1)).to(inputs.device)

        # lists for weighted mode
        p_exit_at_gate_list = []
        loss_per_gate_list = []
        G = torch.zeros((targets.shape[0], 1)).to(inputs.device) # holds the g's, the sigmoided gate outputs
        for l, intermediate_head in enumerate(self.net.intermediate_heads):
            current_logits = intermediate_head(intermediate_outs[l])
            intermediate_logits.append(current_logits)
            # TODO: Freezing the gate can be done in learning helper when we switch phase.
            with torch.no_grad(): # Prevent backpropagation to gates.
                    exit_gate_logit = self.net.get_gate_prediction(l, current_logits, intermediate_codes)
            g = torch.nn.functional.sigmoid(exit_gate_logit) # g
        
            no_exit_previous_gates_prob = torch.prod(1 - prob_gates, axis=1)[:,None] # prod (1-g)
            if self.loss_contribution_mode == LossContributionMode.SINGLE:
                current_gate_activation_prob = torch.clip(g/no_exit_previous_gates_prob, min=0, max=1)
            elif self.loss_contribution_mode == LossContributionMode.WEIGHTED:
                sum_previous_gs = torch.sum(G, dim=1)[:, None]
                p_exit_at_gate = torch.max(torch.zeros((targets.shape[0], 1)).to(inputs.device), torch.min(g, 1 - sum_previous_gs))
                p_exit_at_gate_list.append(p_exit_at_gate)
                current_gate_activation_prob = torch.clip(p_exit_at_gate/no_exit_previous_gates_prob, min=0, max=1)
                G = torch.cat((G, g), dim=1)
                loss_at_gate = self.classifier_criterion(current_logits, targets)
                loss_per_gate_list.append(loss_at_gate[:, None])

            prob_gates = torch.cat((prob_gates, current_gate_activation_prob), dim=1) # gate exits are independent so they won't sum to 1 over all cols
            if self.gate_selection_mode == GateSelectionMode.PROBABILISTIC:
                do_exit = torch.bernoulli(current_gate_activation_prob)
            elif self.gate_selection_mode == GateSelectionMode.DETERMINISTIC:
                do_exit = current_gate_activation_prob >= 0.5
            current_exit = torch.logical_and(do_exit, torch.logical_not(past_exits))
            current_exit_index = current_exit.flatten().nonzero()
            sample_exit_level_map[current_exit_index] = l
            num_exits_per_gate.append(torch.sum(current_exit))
            # Update past_exists to include the currently exited ones for next iteration
            past_exits = torch.logical_or(current_exit, past_exits)
            # Update early_exit_logits which include all predictions across all layers
            gated_y_logits = gated_y_logits + torch.mul(current_exit, current_logits)
        final_gate_exit = torch.logical_not(past_exits)
        sample_exit_level_map[final_gate_exit.flatten().nonzero()] = len(self.net.intermediate_heads)
        num_exits_per_gate.append(torch.sum(final_gate_exit))
        gated_y_logits = gated_y_logits + torch.mul(final_gate_exit, final_logits) # last gate
        things_of_interest = {
            'intermediate_logits':intermediate_logits,
            'final_logits':final_logits,
            'num_exits_per_gate':num_exits_per_gate,
            'gated_y_logits': gated_y_logits,
            'sample_exit_level_map': sample_exit_level_map,
            'gated_y_logits': gated_y_logits}
        loss = 0
        if self.loss_contribution_mode == LossContributionMode.SINGLE:
            loss = self._compute_single_loss(gated_y_logits, targets)
        elif self.loss_contribution_mode == LossContributionMode.WEIGHTED:
            loss = self._compute_weighted_loss(p_exit_at_gate_list, loss_per_gate_list)
        else:
            raise InvalidLossContributionModeException('Ca marche pas ton affaire')
 
        if compute_hamming:
            things_of_interest = things_of_interest | self._get_hamming_metrics_dict(intermediate_logits, intermediate_codes, targets)
        return loss, things_of_interest
    
    def _compute_weighted_loss(self, p_exit_at_gate_list, loss_per_gate_list):
        P = torch.cat(p_exit_at_gate_list, dim = 1)
        L = torch.cat(loss_per_gate_list, dim = 1)
        loss_per_point = torch.sum(L, dim=1) # we want to maintain this
        weighted_loss = P * L
        ratio = (loss_per_point/torch.sum(weighted_loss, dim=1))[:,None]
        loss = torch.mean(weighted_loss * ratio)
        return loss
    
    def _compute_single_loss(self, gated_y_logits, targets):
        return self.classifier_criterion(gated_y_logits, targets)
    
    def _get_hamming_metrics_dict(self, intermediate_logits, intermediate_codes, targets):
        if self.net.training:
            return {}
        inc_inc_H_list, inc_inc_H_list_std, c_c_H_list, c_c_H_list_std,c_inc_H_list,c_inc_H_list_std = check_hamming_vs_acc(intermediate_logits, intermediate_codes, targets)
        return {'inc_inc_H_list': inc_inc_H_list,
            'c_c_H_list': c_c_H_list,
            'c_inc_H_list': c_inc_H_list,
            'inc_inc_H_list_std': inc_inc_H_list_std,
            'c_c_H_list_std': c_c_H_list_std,
            'c_inc_H_list_std': c_inc_H_list_std}