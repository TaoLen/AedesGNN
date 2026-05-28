import torch.nn as nn


def mpnn_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
    elif isinstance(module, nn.Sequential):
        for submodule in module:
            mpnn_weights(submodule)
    elif hasattr(module, "reset_parameters"):
        module.reset_parameters()
