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

class GateTrainingScheme(Enum):
    """
    The training scheme when training gates.
    Default means training the optimal gate to exit while all others are forced not to.
    Ignore subsequent means training the optimal gate to exit, the previous gates to not exit and we ignore later (deeper) gates
    Exit subsequent means training the optimal gate to exit, all subsequent gates to exit as well while earlier gates are trained not to exit.
    """
    DEFAULT = 1
    IGNORE_SUBSEQUENT = 2
    EXIT_SUBSEQUENT = 3

class GateObjective(Enum):
    CrossEntropy = 1
    ZeroOne = 2
    Prob = 3

class InvalidLossContributionModeException(Exception):
    pass

class GateTrainingHelper:
    def __init__(self, net: nn.Module, gate_training_scheme: GateTrainingScheme, gate_objective: GateObjective) -> None:
        self.net = net
        self.gate_training_scheme = gate_training_scheme
        self.gate_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.gate_objective = gate_objective
        if self.gate_objective == GateObjective.CrossEntropy:
            self.predictor_criterion = nn.CrossEntropyLoss(reduction='none') # Measures Cross entropy loss
        elif self.gate_objective == GateObjective.ZeroOne:
            self.predictor_criterion = self.zeroOneLoss # Measures the accuracy
        elif self.gate_objective == GateObjective.Prob:
            self.predictor_criterion = self.prob_when_correct # Measures prob of the correct class if accurate, else returns 0
    
    def zeroOneLoss(self, logits, targets):
        _, predicted = logits.max(1)
        correct = predicted.eq(targets)
        return correct

    def prob_when_correct(self, logits, targets):
        probs = torch.nn.functional.softmax(logits, dim=1) # get the probs
        p_max, _ = torch.topk(probs, 1) # get p max


        _, predicted = logits.max(1) # get the prediction
        correct = predicted.eq(targets)[:,None]

        prob_when_correct = correct * p_max # hadamard product, p when the prediciton is correct, else 0
        
        return prob_when_correct

    def get_loss(self, inputs: torch.Tensor, targets: torch.tensor):
        final_head, intermediate_zs, intermediate_codes = self.net.module.forward_features(inputs)
        final_logits = self.net.module.head(final_head)
        intermediate_losses = []
        gate_logits = []
        intermediate_logits = []
        
        optimal_exit_count_per_gate = dict.fromkeys(range(len(self.net.module.intermediate_heads)), 0) # counts number of times a gate was selected as the optimal gate for exiting
        
        for l, intermediate_head in enumerate(self.net.module.intermediate_heads):
            current_logits = intermediate_head(intermediate_zs[l])
            intermediate_logits.append(current_logits)
            current_gate_logits = self.net.module.get_gate_prediction(l, current_logits, intermediate_codes)      
            gate_logits.append(current_gate_logits)
            pred_loss = self.predictor_criterion(current_logits, targets)
            ic_loss = (l + 1) / (len(intermediate_zs) + 1)
            level_loss = pred_loss + self.net.module.CE_IC_tradeoff * ic_loss
            level_loss = level_loss[:, None]
            intermediate_losses.append(level_loss)
        # add the final head as an exit to optimize for
        final_pred_loss = self.predictor_criterion(final_logits, targets)
        final_ic_loss = 1
        final_level_loss = final_pred_loss + self.net.module.CE_IC_tradeoff * final_ic_loss
        final_level_loss = final_level_loss[:, None]
        intermediate_losses.append(final_level_loss)
        
        gate_target = torch.argmin(torch.cat(intermediate_losses, dim = 1), dim = 1) # For each sample in batch, which gate should exit
        for gate_level in optimal_exit_count_per_gate.keys():
            count_exit_at_level = torch.sum(gate_target == gate_level).item()
            optimal_exit_count_per_gate[gate_level] += count_exit_at_level
        things_of_interest = {'exit_count_optimal_gate': optimal_exit_count_per_gate}
        gate_target_one_hot = torch.nn.functional.one_hot(gate_target, len(self.net.module.intermediate_heads) + 1)
        gate_logits = torch.cat(gate_logits, dim=1)
        if self.gate_training_scheme == GateTrainingScheme.IGNORE_SUBSEQUENT:
            loss, correct_exit_count = self._get_ignore_subsequent_loss(gate_target_one_hot, gate_target, gate_logits)
        elif self.gate_training_scheme == GateTrainingScheme.EXIT_SUBSEQUENT:
            loss, correct_exit_count = self._get_exit_subsequent_loss(gate_target_one_hot, gate_logits)
        things_of_interest = things_of_interest | {'intermediate_logits': intermediate_logits, 'final_logits':final_logits, 'correct_exit_count': correct_exit_count}
        return loss, things_of_interest
    
    def _get_exit_subsequent_loss(self, gate_target_one_hot, gate_logits):
        correct_exit_count = 0
        gate_target_one_hot = gate_target_one_hot[:,:-1] # remove exit since there is not associated gate
        hot_encode_subsequent = gate_target_one_hot.cumsum(dim=1)
        gate_loss = self.gate_criterion(gate_logits.flatten(), hot_encode_subsequent.double().flatten())
        # addressing the class imbalance avec classe
        num_ones = torch.sum(hot_encode_subsequent)
        # add a check up to make sure we have at least 1 ones. If not, add one at random to avoid nan issue
        if num_ones < 1:
            print('Warning, this batch is pushing everything to the last gate')
            hot_encode_subsequent[random.choice(range(hot_encode_subsequent.shape[0])),-1] = 1 # place it at the second last gate for some random point
            num_ones = torch.sum(hot_encode_subsequent)
            assert num_ones == 1
        num_zeros = (torch.prod(torch.Tensor(list(hot_encode_subsequent.shape)))) - num_ones
        zero_to_one_ratio = num_zeros / num_ones
        ones_loss_multiplier = hot_encode_subsequent.double().flatten() * zero_to_one_ratio # balances ones
        zeros_loss_multiplier = torch.logical_not(hot_encode_subsequent).double().flatten()
        multiplier = ones_loss_multiplier + zeros_loss_multiplier
        # compute gate accuracies
        actual_exits_binary = torch.nn.functional.sigmoid(gate_logits) >= 0.5
        correct_exit_count += accuracy_score(actual_exits_binary.flatten().cpu(), hot_encode_subsequent.double().flatten().cpu(), normalize=False)
        gate_loss = torch.mean(gate_loss * multiplier)
        # things_of_interest = things_of_interest | {'intermediate_logits': intermediate_logits, 'final_logits':final_logits, 'correct_exit_count': correct_exit_count}
        return gate_loss, correct_exit_count
    
    def _get_ignore_subsequent_loss(self, gate_target_one_hot, gate_target, gate_logits):
        losses = []
        exit_counts = []
        correct_exit_count = 0
        for gate_idx in range(0, len(self.gate_positions)):
            samples_idx_should_exit_at_gate = (gate_target == gate_idx).nonzero().flatten()
            exit_counts.append(len(samples_idx_should_exit_at_gate))
            if len(samples_idx_should_exit_at_gate) == 0:
                continue
            one_hot_exiting_at_gate = gate_target_one_hot[samples_idx_should_exit_at_gate]
            one_hot_exiting_at_gate = one_hot_exiting_at_gate[:, :gate_idx + 1] # chop all gates after the relevant gate
            gate_logits_at_gate = gate_logits[samples_idx_should_exit_at_gate, :(gate_idx + 1)]
            actual_exit_binary = torch.nn.functional.sigmoid(gate_logits_at_gate) >= 0.5
            correct_exit_count += accuracy_score(actual_exit_binary.flatten().cpu(), one_hot_exiting_at_gate.double().flatten().cpu(), normalize=False)
            losses.append(self.gate_criterion(gate_logits_at_gate, one_hot_exiting_at_gate.double()))
        num_ones = gate_logits.shape[0] # this was len(target) ie the batch size.
        num_zeroes = 0
        for idx, exit_count in enumerate(exit_counts):
            num_zeroes += idx * exit_count
        zero_to_one_ratio = num_zeroes / num_ones
        weighted_losses = []
        for idx, loss in enumerate(losses):
            multiplier = torch.ones_like(loss)
            multiplier[:, -1] = zero_to_one_ratio
            weighted_losses.append((loss * multiplier).flatten())
        gate_loss = torch.mean(torch.cat(weighted_losses))
        return gate_loss, correct_exit_count