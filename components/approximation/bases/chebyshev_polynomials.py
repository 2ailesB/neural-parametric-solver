import torch

from components.approximation.FunctionBasis import FunctionBasis
from utils.gradients import gradients

class Chebyshev_Poly(FunctionBasis):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        
        self.cfg = cfg
        self.N = cfg.N

        self.dim=1
        self.channels=1

    def compute_u(self, x, theta):
        basis = torch.cat([torch.special.chebyshev_polynomial_t(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1) # B X N

        return torch.bmm(basis, theta)
    
    def get_theta_size(self):
        return self.N + self.bias
    
    def compute_uderivativex(self, x, theta):
        ns = torch.arange(0, self.N-1, 1, device=self.device)

        basis = torch.cat([torch.zeros(x.shape, device=self.device).unsqueeze(-1)] + [ns[i] * torch.special.chebyshev_polynomial_u(x, i).unsqueeze(-1) for i in range(self.N-1)], dim=-1) # B X N
        
        return torch.bmm(basis, theta)

    def compute_uderivativex2(self, x, theta):
        ns = torch.arange(0, self.N, 1, device=self.device)
        ns1 = torch.arange(1, self.N+1, 1, device=self.device)

        basist = torch.cat([torch.special.chebyshev_polynomial_t(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1)
        basisu = torch.cat([torch.special.chebyshev_polynomial_u(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1)

        basis = (torch.einsum('N, BXN -> BXN', ns1, basist) - basisu) 
        basis = torch.einsum('N, BXN -> BXN', ns, basis)

        basis = torch.einsum('BXN, BX -> BXN', basis, 1 / (x**2 - 1)) # TODO : à checker que c'est pas nimpp

        return torch.bmm(basis, theta)

    def compute_uderivativetheta(self, x, theta):
        return torch.cat([torch.special.chebyshev_polynomial_t(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1)

    def get_basis(self, x): 
        return torch.cat([torch.special.chebyshev_polynomial_t(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1) # B X N

    
    def get_basis_derivativex(self, x):
        ns = torch.arange(0, self.N-1, 1, device=self.device)

        return torch.cat([torch.zeros(x.shape, device=self.device).unsqueeze(-1)] + [ns[i] * torch.special.chebyshev_polynomial_u(x, i).unsqueeze(-1) for i in range(self.N-1)], dim=-1) # B X N
        

    def get_basis_derivativex2(self, x):
        ns = torch.arange(0, self.N, 1, device=self.device)
        ns1 = torch.arange(1, self.N+1, 1, device=self.device)

        basist = torch.cat([torch.special.chebyshev_polynomial_t(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1)
        basisu = torch.cat([torch.special.chebyshev_polynomial_u(x, i).unsqueeze(-1) for i in range(self.N)], dim=-1)

        basis = (torch.einsum('N, BXN -> BXN', ns1, basist) - basisu) 
        basis = torch.einsum('N, BXN -> BXN', ns, basis)

        return basis = torch.einsum('BXN, BX -> BXN', basis, 1 / (x**2 - 1))