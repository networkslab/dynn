import torch
from torch import nn


def compute_layer_weight_norm(module: nn.Module):
    return torch.linalg.vector_norm(module.weight)

def get_network_weight_norm_dict(net: nn.Module, only_modules = ['intermediate_heads', 'gates']):
    weight_dict = {}
    with torch.no_grad():
        params_to_collect = net.named_parameters() if len(only_modules) == 0 else list(
            filter(lambda m: m[0].split('.')[0] in only_modules, list(net.named_parameters()))
        )
        weights_only = list(filter(lambda m: 'weight' in m[0], params_to_collect))
        for name, param in weights_only:
            weight_dict[name] = torch.linalg.vector_norm(param).item()
    return weight_dict