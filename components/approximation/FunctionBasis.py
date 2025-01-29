import torch
import einops

from abc import ABC, abstractmethod

class FunctionBasis(ABC):
    """ 
    Abstract class for function basis
    """
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device('cpu')
        self.bias = cfg.add_bias
        self.N = cfg.N
        self.Ns4tr = (cfg.N, ) 
        self.dim = 1
        self.channels = 1 

        self.norms = 0

    def to(self, device):
        self.device=device

    def get_theta_size(self):
        return self.N + self.bias

    @abstractmethod
    def compute_u(self, x, theta): 
        basis = self.get_basis(x)
        return torch.bmm(basis, theta)

    @abstractmethod
    def compute_uderivativex(self, x, theta):
        basis=self.get_basis_derivativex(x)
        return torch.bmm(basis, theta)

    @abstractmethod
    def compute_uderivativex2(self, x, theta):
        basis=self.get_basis_derivativex2(x)
        return torch.bmm(basis, theta)

    @abstractmethod
    def get_basis(self, x):
        pass

    @abstractmethod
    def get_basis_derivativex(self, x):
        pass

    @abstractmethod
    def get_basis_derivativex2(self, x):
        pass

    def compute_L2_norm(self, xlim=(0, 1)):
        """x.shape: torch.Size([B, XYZ, D])
        theta.shape: torch.Size([B, N1N2N3, C])"""
        
        x = torch.linspace(xlim[0], xlim[1], 64, device=self.device).unsqueeze(0)
        base = self.get_basis(x)
        
        return torch.norm(base, dim=1)
    
    def projection(self, f, x):
        """x.shape: torch.Size([B, XYZ, D])
        theta.shape: torch.Size([B, N1N2N3, C])
        f.shape: torch.Size([3, XYZ, D])"""

        base = self.get_basis(x)
        f=f.transpose(1, 2)
        dot_product = torch.bmm(f, base)
        norm_base_squared = torch.norm(base, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1) ** 2
        scalar_proj = dot_product / norm_base_squared
        
        return scalar_proj.transpose(1, 2)

