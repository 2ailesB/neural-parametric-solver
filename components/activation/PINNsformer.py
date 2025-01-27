""""
Implementation of PINNsformer Activation function :https://github.com/AdityaLab/pinnsformer/blob/main/model/pinnsformer.py
"""

import torch
import torch.nn as nn


class SinCosActivation(nn.Module):
    """Applies a combined sine and cosine activation function.
    
    f(x) = w1 * sin(x) + w2 * cos(x)
    
    Args:
        None (weights w1 and w2 are trainable parameters).
    """
    def __init__(self):
        super(SinCosActivation, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x)+ self.w2 * torch.cos(x)