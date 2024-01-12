import torch
import torch.nn.functional as F

@torch.jit.script
def loss_func(outs, targets):
    return - (targets * F.log_softmax(outs, dim=-1)).mean() * 3