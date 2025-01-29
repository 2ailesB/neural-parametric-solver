import torch

from components.approximation.FunctionBasis import FunctionBasis
from utils.gradients import gradients

class Legendre_Poly(FunctionBasis):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        
        self.cfg = cfg
        self.N = cfg.N

        self.dim=1
        self.channels=1

    def compute_u(self, x, theta):
        basis = torch.cat([torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1) # B X N

        return torch.bmm(basis, theta)
    
    def compute_uderivativex(self, x, theta):
        ns1 = torch.arange(1, self.N+1, 1, device=self.device)

        basis = torch.cat([torch.zeros(x.shape, device=self.device).unsqueeze(-1)] + [torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in range(self.N-1)], dim=-1) # B X N
        basis1 = torch.cat([torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in ns1], dim=-1) # B X N
        # XPn(x) - Pn+1(x)
        basis = torch.einsum('BXN, BX -> BXN', basis, x) - basis1
        # ./X^2-1
        basis = torch.einsum('BXN, BX -> BXN', basis, 1 / (x**2 - 1))
        # (n+1) * .
        basis = torch.einsum('BXN, N -> BXN', basis, -ns1)

        return torch.bmm(basis, theta)

    def compute_uderivativex2(self, x, theta):
        ns = torch.arange(0, self.N, 1, device=self.device)
        ns1 = torch.arange(1, self.N+1, 1, device=self.device)

        basis = torch.cat([torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1) # B X N
        basis1 = torch.cat([torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in ns1], dim=-1) # B X N
        ply = torch.einsum('N, BX -> BXN', ns, x**2 - 1) + 2*x.unsqueeze(-1).repeat((1, 1, self.N)) ** 2
        # (n(x^2-1)+2x^2)Pn(x) - 2xPn+1(x)
        basis = ply * basis - torch.einsum('BXN, BX -> BXN', basis1, -2*x)
        # ./X^2-1
        basis = torch.einsum('BXN, BX -> BXN', basis, 1 / (x**2 - 1))
        # (n+1) * .
        basis = torch.einsum('BXN, N -> BXN', basis, ns1)

        return torch.bmm(basis, theta)

    def compute_uderivativetheta(self, x, theta):
        return torch.cat([torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1)

    def get_basis(self, x): 
        return torch.cat([torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1) # B X N
    
    def get_basis_derivativex(self, x):
        ns1 = torch.arange(1, self.N+1, 1, device=self.device)

        basis = torch.cat([torch.zeros(x.shape, device=self.device).unsqueeze(-1)] + [torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in range(self.N-1)], dim=-1) # B X N
        basis1 = torch.cat([torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in ns1], dim=-1) # B X N
        # XPn(x) - Pn+1(x)
        basis = torch.einsum('BXN, BX -> BXN', basis, x) - basis1
        # ./X^2-1
        basis = torch.einsum('BXN, BX -> BXN', basis, 1 / (x**2 - 1))
        # (n+1) * .
        basis = torch.einsum('BXN, N -> BXN', basis, -ns1)
         return basis

    def get_basis_derivativex2(self, x):
        ns = torch.arange(0, self.N, 1, device=self.device)
        ns1 = torch.arange(1, self.N+1, 1, device=self.device)

        basis = torch.cat([torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1) # B X N
        basis1 = torch.cat([torch.special.legendre_polynomial_p(x, i).unsqueeze(-1) for i in ns1], dim=-1) # B X N
        ply = torch.einsum('N, BX -> BXN', ns, x**2 - 1) + 2*x.unsqueeze(-1).repeat((1, 1, self.N)) ** 2
        # (n(x^2-1)+2x^2)Pn(x) - 2xPn+1(x)
        basis = ply * basis - torch.einsum('BXN, BX -> BXN', basis1, -2*x)
        # ./X^2-1
        basis = torch.einsum('BXN, BX -> BXN', basis, 1 / (x**2 - 1))
        # (n+1) * .
        basis = torch.einsum('BXN, N -> BXN', basis, ns1)

        return basis
