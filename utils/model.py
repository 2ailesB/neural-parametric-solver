import torch

""" Model utilities """

def count_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params

def count_trainable_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params
