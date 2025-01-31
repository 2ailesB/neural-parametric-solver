import torch
import torch.nn as nn

from components.activation.BeyondPeriodicity import GaussianActivation, QuadraticActivation, MultiQuadraticActivation, LaplacianActivation, SuperGaussianActivation, ExpSinActivation
from components.activation.PINNsformer import SinCosActivation

activations = {
    "relu": torch.nn.ReLU, 
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "gelu": torch.nn.GELU, 
    "swish": torch.nn.SiLU, 
    "silu": torch.nn.SiLU,
    "gaussian": GaussianActivation,
    "quadratic": QuadraticActivation,
    "multiquadratic": MultiQuadraticActivation,
    "laplacian": LaplacianActivation,
    "supergaussian": SuperGaussianActivation,
    "expsin": ExpSinActivation,
    'sincos': SinCosActivation
    }

def init_activation(activation: str) -> nn.Module:
    """Initializes an activation function from the dictionary.
    
    Args:
        activation (str): Name of the activation function.
    
    Returns:
        nn.Module: An instance of the requested activation function.
    
    Raises:
        ValueError: If the activation function is unknown.
    """
    try:
        return activations[activation]()
    except KeyError:
        raise ValueError(f'Unknwon Activation function {activation}')
