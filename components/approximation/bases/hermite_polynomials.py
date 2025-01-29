import torch

from components.approximation.FunctionBasis import FunctionBasis
from utils.gradients import gradients

class Hermite_Poly(FunctionBasis):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        
        self.cfg = cfg
        self.N = cfg.N

        self.dim=1
        self.channels=1

    def compute_u(self, x, theta):
        basis = torch.cat([torch.special.hermite_polynomial_h(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1) # B X N
        return torch.bmm(basis, theta)
    
    def compute_uderivativex(self, x, theta):
        nsm1 = torch.arange(0, self.N - 1, 1, device=self.device)

        basis = torch.cat([torch.zeros(x.shape, device=self.device).unsqueeze(-1)] + [torch.special.hermite_polynomial_h(x, i).unsqueeze(-1) for i in nsm1], dim=-1) # B X N
        basis = torch.einsum('BXN, N -> BXN', basis, torch.arange(0, self.N, 1, device=self.device))

        return torch.bmm(basis, theta)

    def compute_uderivativex2(self, x, theta):
        ns = torch.arange(0, self.N , 1, device=self.device)
        nsm1 = torch.arange(-1, self.N - 1, 1, device=self.device)
        nsm2 = torch.arange(0, self.N - 2, 1, device=self.device)

        basis = torch.cat([torch.zeros(x.shape, device=self.device).unsqueeze(-1)] + [torch.zeros(x.shape, device=self.device).unsqueeze(-1)] + [torch.special.hermite_polynomial_h(x, i).unsqueeze(-1) for i in nsm2], dim=-1) # B X N
        basis = torch.einsum('BXN, N -> BXN', basis, ns * nsm1)

        return torch.bmm(basis, theta)

    def compute_uderivativetheta(self, x, theta):
        return torch.cat([torch.special.hermite_polynomial_h(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1)

    def get_basis(self, x): 
        return torch.cat([torch.special.hermite_polynomial_h(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1) # B X N

    def get_basis_derivativex(self, x):
        nsm1 = torch.arange(0, self.N - 1, 1, device=self.device)

        basis = torch.cat([torch.zeros(x.shape, device=self.device).unsqueeze(-1)] + [torch.special.hermite_polynomial_h(x, i).unsqueeze(-1) for i in nsm1], dim=-1) # B X N
        return torch.einsum('BXN, N -> BXN', basis, torch.arange(0, self.N, 1, device=self.device))

    def get_basis_derivativex2(self, x):
        ns = torch.arange(0, self.N , 1, device=self.device)
        nsm1 = torch.arange(-1, self.N - 1, 1, device=self.device)
        nsm2 = torch.arange(0, self.N - 2, 1, device=self.device)

        basis = torch.cat([torch.zeros(x.shape, device=self.device).unsqueeze(-1)] + [torch.zeros(x.shape, device=self.device).unsqueeze(-1)] + [torch.special.hermite_polynomial_h(x, i).unsqueeze(-1) for i in nsm2], dim=-1) # B X N
        return torch.einsum('BXN, N -> BXN', basis, ns * nsm1)
