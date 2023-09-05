import torch
import torch.nn as nn
import numpy as np
from models.gradient_rescale import GradientRescaleFunction
from models.t2t_vit import T2T_ViT

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

    def forward(self, x, stage=None):
        outs = self.boosted_forward(x)
        preds = [0]
        for i in range(len(outs)):
            pred = outs[i] + preds[-1] * self.ensemble_reweight[i]
            preds.append(pred)
        preds = preds[1:]
        return preds




gradient_rescale = GradientRescaleFunction.apply
