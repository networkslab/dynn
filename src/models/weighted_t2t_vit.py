import torch
import torch.nn as nn
import numpy as np
from models.t2t_vit import T2T_ViT

# Weighted model
class WeightedT2tVit(T2T_ViT):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__(img_size, tokens_type, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                         qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, token_dim)

    # Adds intermediate classifiers to take the output of the attention blocks
    def set_intermediate_heads(self, intermediate_head_positions):
        super().set_intermediate_heads(intermediate_head_positions)

    def forward(self, x):
        x, intermediate_outs, _ = self.forward_features(x)
        intermediate_logits = []
        if intermediate_outs: # what is this?
            for head_idx, intermediate_head in enumerate(self.intermediate_heads):
                intermediate_logits.append(intermediate_head(intermediate_outs[head_idx]))
        x = self.head(x)
        # The intermediate logits are unnormalized
        # append x and intermediate logits in a single list
        outs = intermediate_logits
        outs.append(x)
        return outs
