# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer
from .transformer_block import Block, get_sinusoid_encoding
from src.models.custom_modules.custom_GELU import CustomGELU
from src.models.custom_modules.learnable_gate import LearnableGate
from enum import Enum
class TrainingPhase(Enum):
    CLASSIFIER = 1
    GATE = 2


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            #self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            #self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with convolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x

class IntermediateOutput:
    def __init__(self, level: int, predictions: torch.Tensor, predictions_idx: torch.Tensor, remaining_idx: torch.Tensor):
        self.level = level
        self.predictions = predictions
        self.predictions_idx = predictions_idx
        self.remaining_idx = remaining_idx

class T2T_ViT(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cost_perf_tradeoff = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=CustomGELU)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()


        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    '''
    sets intermediate classifiers that are hooked after inner transformer blocks
    '''
    def set_intermediate_heads(self, intermediate_head_positions):
        self.intermediate_head_positions = intermediate_head_positions
        self.cost_per_gate = [(i+1)/len(self.intermediate_head_positions) for i in range(len(self.intermediate_head_positions))]

        self.intermediate_heads = nn.ModuleList([
            nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
            for _ in range(len(self.intermediate_head_positions))])
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        intermediate_transformer_outs = []
        for blk_idx, blk in enumerate(self.blocks):
            x, act_code = blk.forward_get_code(x)
            if hasattr(self, 'intermediate_head_positions') and blk_idx in self.intermediate_head_positions:
                intermediate_transformer_outs.append(x)
        intermediate_transformer_outs = list(map(lambda inter_out: self.norm(inter_out)[:, 0], intermediate_transformer_outs))
        x = self.norm(x)
        return x[:, 0], intermediate_transformer_outs

    def forward(self, x):
        x, intermediate_transformer_outs = self._forward_features(x)
        intermediate_outs = []
        if not not intermediate_transformer_outs:
            for head_idx, intermediate_head in enumerate(self.intermediate_heads):
                intermediate_outs.append(intermediate_head(intermediate_transformer_outs[head_idx]))
        x = self.head(x)
        # The intermediate outs are unnormalized
        return x, intermediate_outs


    # this is only to be used for training
    def forward_brute_force(self, inputs, normalize = False):
        x, intermediate_transformer_outs = self._forward_features(inputs)
        intermediate_outs = []
        all_y = []
        total_inference_cost = []
        prob_gates = torch.zeros((inputs.shape[0], 1)).to(inputs.device)
        for l, intermediate_head in enumerate(self.intermediate_heads):
            intermediate_logits = intermediate_head(intermediate_transformer_outs[l])
            
            intermediate_outs.append(intermediate_logits)
            current_gate_prob = torch.nn.functional.sigmoid(self.gates[l](intermediate_transformer_outs[l]))

            prob_gates = torch.cat((prob_gates, current_gate_prob), dim=1) # gate exits are independent so they won't sum to 1 over all cols
            cumul_previous_gates = torch.prod(1 - prob_gates[:,:-1], axis=1) # check this 
            if normalize:
                cumul_previous_gates = torch.pow(cumul_previous_gates, 1/(l + 1))
            cumul_previous_gates = cumul_previous_gates[:, None]


            y_prob_intermediate = torch.nn.functional.softmax(intermediate_logits, dim=1)
        
            gate_coef = cumul_previous_gates * current_gate_prob
            # log_metrics({gate_coef},
            #                    step=batch_idx + (epoch * len(trainloader)))
            weighted_by_gate_prob =  y_prob_intermediate * gate_coef
            
            all_y.append(weighted_by_gate_prob[:,None])
            cost_of_gate = self.cost_per_gate[l] * gate_coef
            total_inference_cost.append(cost_of_gate)
        
        return torch.sum(torch.cat(all_y, axis=1), axis=1),  torch.sum(torch.cat(total_inference_cost, axis=1), axis=1), intermediate_outs


    def surrogate_forward(self, inputs: torch.Tensor, targets: torch.tensor, training_phase: TrainingPhase):
        final_head, intermediate_transformer_outs = self._forward_features(inputs)
        final_logits = self.head(final_head)
        if training_phase == TrainingPhase.CLASSIFIER:
            # Gates are frozen, find first exit gate
            intermediate_logits = []
            num_exits_per_gate = []
            early_exit_logits = torch.zeros_like(final_logits)
            past_exits = torch.zeros((inputs.shape[0], 1)).to(inputs.device)
            for l, intermediate_head in enumerate(self.intermediate_heads):
                current_logits = intermediate_head(intermediate_transformer_outs[l])
                intermediate_logits.append(current_logits)
                current_gate_prob = torch.nn.functional.sigmoid(self.gates[l](current_logits))
                do_exit = torch.bernoulli(current_gate_prob)
                current_exit = torch.logical_and(do_exit, torch.logical_not(past_exits))
                num_exits_per_gate.append(torch.sum(current_exit))
                past_exits = torch.logical_or(current_exit, past_exits) 
                early_exit_logits = early_exit_logits + torch.mul(current_exit, current_logits)
            final_gate_exit = torch.logical_not(past_exits)
            num_exits_per_gate.append(torch.sum(final_gate_exit))
            early_exit_logits = early_exit_logits + torch.mul(final_gate_exit, final_logits) # last gate
            things_of_interest = {'intermediate_logits':intermediate_logits, 'final_logits':final_logits, 'num_exits_per_gate':num_exits_per_gate}
            return early_exit_logits, things_of_interest
        elif training_phase == TrainingPhase.GATE:
            intermediate_losses = []
            gate_logits = []
            intermediate_logits = []
            accuracy_criterion = nn.CrossEntropyLoss(reduction='none') # Measures accuracy of classifiers
       
            gate_criterion = nn.BCEWithLogitsLoss(reduction='none')
            for l, intermediate_head in enumerate(self.intermediate_heads):
                current_logits = intermediate_head(intermediate_transformer_outs[l])
                intermediate_logits.append(current_logits)
                current_gate_logits = self.gates[l](current_logits)
                gate_logits.append(current_gate_logits)
                ce_loss = accuracy_criterion(current_logits, targets)
                ic_loss = (l + 1) / (len(intermediate_transformer_outs) +  1)
                level_loss = ce_loss + self.cost_perf_tradeoff * ic_loss
                level_loss = level_loss[:, None]
                intermediate_losses.append(level_loss)
            gate_target = torch.argmin(torch.cat(intermediate_losses, dim = 1), dim = 1) # For each sample in batch, which gate should exit
            gate_target_one_hot = torch.nn.functional.one_hot(gate_target, len(self.intermediate_heads))
            gate_logits = torch.cat(gate_logits, dim=1)
            gate_loss = gate_criterion(gate_logits.flatten(), gate_target_one_hot.double().flatten())
            # addressing the class imbalance avec audace
            weight_per_sample_per_gate = gate_target_one_hot.double().flatten() * (len(self.gates)-1) +1
            gate_loss = torch.mean(gate_loss* weight_per_sample_per_gate)
            things_of_interest = {'intermediate_logits':intermediate_logits, 'final_logits':final_logits}
            return gate_loss, things_of_interest

    def set_threshold_gates(self, gates):
        assert len(gates) == len(self.intermediate_heads), 'Net should have as many gates as there are intermediate classifiers'
        self.gates = gates

    
    def set_learnable_gates(self, gate_positions):

        self.gate_positions = gate_positions
        self.gates = nn.ModuleList([
            LearnableGate() for _ in range(len(self.gate_positions))])


@register_model
def t2t_vit_7(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_10(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_10']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_12(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_19(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_24(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_t_14(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_t_19(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_t_24(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# rexnext and wide structure
@register_model
def t2t_vit_14_resnext(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=32, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_resnext']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_14_wide(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=768, depth=4, num_heads=12, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_wide']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
