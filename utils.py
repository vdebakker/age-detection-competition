import torch


def cycle(dl):
    while True:
        for batch in dl:
            yield batch
        
        
@torch.jit.script
def ema_avg(averaged_model_parameter: torch.Tensor, 
            model_parameter: torch.Tensor, num_averaged: torch.Tensor):
    return .999 * averaged_model_parameter + (1 - .999) * model_parameter