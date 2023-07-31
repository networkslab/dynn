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
from .gate_training_helper import GateTrainingScheme
from .classifier_training_helper import GateSelectionMode
from sklearn.metrics import accuracy_score
from enum import Enum
class TrainingPhase(Enum):
    CLASSIFIER = 1
    GATE = 2
    WARMUP = 3



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
    'T2t_vit_7_boosted': _cfg(),
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

class ClassifierAccuracyTracker:
    def __init__(self, level: int, patience: int = 3):
        self.level = level
        self.patience = patience
        self.test_accs = Queue(maxsize=patience)
        self.frozen = False

    def insert_acc(self, test_acc):
        if self.frozen:
            return
        if self.test_accs.qsize() == self.patience:
            removed_acc = self.test_accs.get()
            print(f"Removing tracked value of {removed_acc} and inserting {test_acc}")
        self.test_accs.put(test_acc)

    def should_freeze(self, most_recent_acc):
        if self.frozen:
            print(f"Classifier {self.level} already frozen")
        if self.test_accs.qsize() < self.patience:
            return False
        should_freeze = True
        while not self.test_accs.empty():
            acc = self.test_accs.get()
            if acc < most_recent_acc:
                should_freeze = False
        self.frozen = should_freeze
        return should_freeze

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
        self.mlp_dim = mlp_ratio*embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
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

    def set_CE_IC_tradeoff(self, CE_IC_tradeoff):
        self.CE_IC_tradeoff = CE_IC_tradeoff

    def set_gate_training_scheme_and_mode(self, gate_training_scheme: GateTrainingScheme, gate_selection_mode: GateSelectionMode):
        self.gate_training_scheme = gate_training_scheme
        self.gate_selection_mode = gate_selection_mode


    '''
    sets intermediate classifiers that are hooked after inner transformer blocks
    '''
    def set_intermediate_heads(self, intermediate_head_positions):
        self.intermediate_head_positions = intermediate_head_positions
        self.cost_per_gate = [(i+1)/len(self.intermediate_head_positions) for i in range(len(self.intermediate_head_positions))]

        self.intermediate_heads = nn.ModuleList([
            nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
            for _ in range(len(self.intermediate_head_positions))])
        self.accuracy_trackers = []
        for i in range(len(self.intermediate_head_positions)):
            self.accuracy_trackers.append(ClassifierAccuracyTracker(i))

    def freeze_intermediate_classifier(self, classifier_idx):
        classifier = self.intermediate_heads[classifier_idx]
        for param in classifier.parameters():
            param.requires_grad = False

    def are_all_classifiers_frozen(self):
        for i in range(len(self.accuracy_trackers)):
            if not self.accuracy_trackers[i].frozen:
                return False
        return True

    def set_threshold_gates(self, gates):
        assert len(gates) == len(self.intermediate_heads), 'Net should have as many gates as there are intermediate classifiers'
        self.gates = gates

    
    def set_learnable_gates(self, device, gate_positions, direct_exit_prob_param=False, gate_type=GateType.UNCERTAINTY, proj_dim=32, num_proj=16):
        self.gate_positions = gate_positions
        self.direct_exit_prob_param = direct_exit_prob_param
        self.gate_type = gate_type
        input_dim_code = self.mlp_dim*(self.tokens_to_token.num_patches+1)
        if gate_type == GateType.UNCERTAINTY:
            self.gates = nn.ModuleList([
                LearnableUncGate() for _ in range(len(self.gate_positions))])
        elif gate_type == GateType.CODE:
            self.gates = nn.ModuleList([
                LearnableCodeGate(device, input_dim=input_dim_code, proj_dim=proj_dim, num_proj=num_proj) for _ in range(len(self.gate_positions))])
        elif gate_type == GateType.CODE_AND_UNC:
            self.gates = nn.ModuleList([
                LearnableComplexGate(device, input_dim=input_dim_code, proj_dim=proj_dim, num_proj=num_proj) for _ in range(len(self.gate_positions))])
        

    def get_gate_prediction(self, l, current_logits, intermediate_codes):
        if self.gate_type == GateType.UNCERTAINTY:
            return self.gates[l](current_logits)
        elif self.gate_type == GateType.CODE:
            return self.gates[l](intermediate_codes[l])
        elif self.gate_type == GateType.CODE_AND_UNC:
            return self.gates[l](intermediate_codes[l], current_logits)

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

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        intermediate_z = [] # the embedding fed into the agumenting classifiers
        intermediate_codes = []
        for blk_idx, blk in enumerate(self.blocks):
            x, act_code = blk.forward_get_code(x)
            if hasattr(self, 'intermediate_head_positions') and blk_idx in self.intermediate_head_positions:
                intermediate_z.append(x)
                intermediate_codes.append(act_code)
        intermediate_z = list(map(lambda inter_out: self.norm(inter_out)[:, 0], intermediate_z))
        x = self.norm(x)
        return x[:, 0], intermediate_z, intermediate_codes

    # Similar to forward_features but passes the intermediate z's through the augmenting classifiers
    def forward(self, x):
        x, intermediate_outs, intermediate_codes = self.forward_features(x)
        intermediate_logits = []
        if intermediate_outs: # what is this?
            for head_idx, intermediate_head in enumerate(self.intermediate_heads):
                intermediate_logits.append(intermediate_head(intermediate_outs[head_idx]))
        x = self.head(x)
        # The intermediate outs are unnormalized
        return x, intermediate_logits, intermediate_codes

# BOOSTED MODEL
class Boosted_T2T_ViT(T2T_ViT):

    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64, ensemble_reweight = [0.5]):
        super().__init__(img_size, tokens_type, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                         qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, token_dim)
        self.ensemble_reweight = ensemble_reweight

    def set_intermediate_heads(self, intermediate_head_positions):
        super().set_intermediate_heads(intermediate_head_positions)
        # Deal with ensemble reweighing
        n_blocks = len(self.blocks)
        assert len(self.ensemble_reweight) in [1, 2, n_blocks]
        if len(self.ensemble_reweight) == 1:
            self.ensemble_reweight = self.ensemble_reweight * n_blocks
        elif len(self.ensemble_reweight) == 2:
            self.ensemble_reweight = list(np.linspace(self.ensemble_reweight[0], self.ensemble_reweight[1], n_blocks))
   

    def boosted_forward(self, x): # Equivalent of forward in msdnet with gradient rescaling.
        res = []
        nBlocks = len(self.blocks)
        B = x.shape[0]
        x = self.tokens_to_token(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk_idx, blk in enumerate(self.blocks):
            x, _ = blk.forward_get_code(x)
            x[-1] = gradient_rescale(x[-1], 1.0 / (nBlocks - blk_idx))
            normalized = self.norm(x)
            if blk_idx < len(self.intermediate_heads): # intermediate block
                pred = self.intermediate_heads[blk_idx](normalized[:, 0])
            else:
                pred = self.head(normalized[:, 0]) # last block
            x[-1] = gradient_rescale(x[-1], (nBlocks - blk_idx - 1))
            res.append(pred)
        return res
    
    def forward_all(self, x, stage=None): # from forward_all in dynamic net which itself calls forward (boosted_forward in our case)
        """Forward the model until block `stage` and get a list of ensemble predictions
        """
        nBlocks = len(self.blocks)
        assert 0 <= stage < nBlocks
        outs = self.boosted_forward(x)
        preds = [0]
        for i in range(len(outs)):
            pred = (outs[i] + preds[-1]) * self.ensemble_reweight[i]
            preds.append(pred)
            if i == stage:
                break
        return outs, preds

class GradientRescaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        ctx.gd_scale_weight = weight
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.gd_scale_weight * grad_output
        return grad_input, grad_weight


gradient_rescale = GradientRescaleFunction.apply
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
def t2t_vit_7_boosted(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = Boosted_T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7_boosted']
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
