import torch.nn as nn


def create_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU(inplace=True)
    elif name == "prelu":
        return nn.PReLU(inplace=True)
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU(inplace=True)
    else:
        raise NotImplementedError(f"{name} is not implemented.")