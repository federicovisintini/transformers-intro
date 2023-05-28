import torch


def count_parameters(model):
    return sum(torch.numel(p) for p in model.parameters() if p.requires_grad)
