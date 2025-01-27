""""
Activation function taken from https://github.com/kwea123/Coordinate-MLPs/tree/master
"""

import torch.nn as nn
import torch

class SinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        """Applies a sine activation function.
    
        f(x) = sin(a * x)
        
        Args:
            a (float, optional): Amplitude of the sine function. Default is 1.
            trainable (bool, optional): Whether 'a' is trainable. Default is True.
        """
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.sin(self.a*x)

class GaussianActivation(nn.Module):
    """Applies a Gaussian activation function.
    
    f(x) = exp(-x^2 / (2 * a^2))
    
    Args:
        a (float, optional): Scaling parameter. Default is 1.
        trainable (bool, optional): Whether 'a' is trainable. Default is True.
    """
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))
    
    def forward(self, x):
        return torch.exp(-x**2 / (2 * self.a**2))

class QuadraticActivation(nn.Module):
    """Applies a quadratic activation function.
    
    f(x) = 1 / (1 + (a * x)^2)
    
    Args:
        a (float, optional): Scaling parameter. Default is 1.
        trainable (bool, optional): Whether 'a' is trainable. Default is True.
    """
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))
    
    def forward(self, x):
        return 1 / (1 + (self.a * x) ** 2)

class MultiQuadraticActivation(nn.Module):
    """Applies a multi-quadratic activation function.
    
    f(x) = 1 / sqrt(1 + (a * x)^2)
    
    Args:
        a (float, optional): Scaling parameter. Default is 1.
        trainable (bool, optional): Whether 'a' is trainable. Default is True.
    """
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))
    
    def forward(self, x):
        return 1 / (1 + (self.a * x) ** 2) ** 0.5

class LaplacianActivation(nn.Module):
    """Applies a Laplacian activation function.
    
    f(x) = exp(-|x| / a)
    
    Args:
        a (float, optional): Scaling parameter. Default is 1.
        trainable (bool, optional): Whether 'a' is trainable. Default is True.
    """
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))
    
    def forward(self, x):
        return torch.exp(-torch.abs(x) / self.a)

class SuperGaussianActivation(nn.Module):
    """Applies a super-Gaussian activation function.
    
    f(x) = exp(-x^2 / (2 * a^2))^b
    
    Args:
        a (float, optional): Scaling parameter. Default is 1.
        b (float, optional): Exponent parameter. Default is 1.
        trainable (bool, optional): Whether 'a' and 'b' are trainable. Default is True.
    """
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b * torch.ones(1), trainable))
    
    def forward(self, x):
        return torch.exp(-x**2 / (2 * self.a**2)) ** self.b

class ExpSinActivation(nn.Module):
    """Applies an exponential sine activation function.
    
    f(x) = exp(-sin(a * x))
    
    Args:
        a (float, optional): Scaling parameter. Default is 1.
        trainable (bool, optional): Whether 'a' is trainable. Default is True.
    """
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))
    
    def forward(self, x):
        return torch.exp(-torch.sin(self.a * x))
