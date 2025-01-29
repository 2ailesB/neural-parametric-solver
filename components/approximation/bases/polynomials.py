import torch

from components.approximation.FunctionBasis import FunctionBasis

class Polynomials (FunctionBasis):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.N = cfg.N
        self.roots = cfg.roots

        self.dim=1
        self.channels=1

    def get_theta_size(self):
        return self.N + self.roots * self.N
    
    def compute_u(self, x, theta):
        coeffs = theta[:, :self.N]
        if self.roots:
            roots = theta[:, self.N:] 

        idxs = torch.arange(0, self.N, 1, device=self.device)

        if self.roots: 
            ins = (x.unsqueeze(1).repeat(1, roots.shape[1], 1) - roots.repeat(1, 1, x.shape[1])) ** idxs.unsqueeze(0).unsqueeze(-1) # X = B, X and idxs : N x ** idxs  should be B, X, N 
            xs = torch.einsum('BNX, BNC -> BXC', ins, coeffs)
        else:
            ins = (x.unsqueeze(-1)) ** idxs # X = B, X and idxs : N x ** idxs  should be B, X, N 
            xs = torch.einsum('BXN, BNC -> BXC', ins, coeffs)

        return xs
    
    def compute_uderivativex(self, x, theta):
        coeffs = theta[:, :self.N]
        if self.roots:
            roots = theta[:, self.N+1:] 
        bsize, xsize = x.shape

        idxs0 = torch.arange(0, self.N, 1, device=self.device)
        idxs1 = torch.cat(torch.zeros(1, device=self.device), torch.arange(0, self.N - 1, 1, device=self.device), dim=0)

        if self.roots: 
            ins = idxs0 * (x.unsqueeze(1).repeat(1, roots.shape[1], 1) - roots.repeat(1, 1, x.shape[1])) ** idxs1.unsqueeze(0).unsqueeze(-1) # X = B, X and idxs : N x ** idxs  should be B, X, N 
            xs = torch.einsum('BNX, BNC -> BXC', ins, coeffs)
        else:
            ins = idxs0 * (x.unsqueeze(-1)) ** idxs1 # X = B, X and idxs : N x ** idxs  should be B, X, N 
            xs = torch.einsum('BXN, BNC -> BXC', ins, coeffs)

        return xs
    
    def compute_uderivativex2(self, x, theta):
        coeffs = theta[:, :self.N]
        if self.roots:
            roots = theta[:, self.N+2:] 
        bsize, xsize = x.shape

        idxs0 = torch.arange(0, self.N, 1, device=self.device)
        idxs1 = torch.cat(torch.zeros(1, device=self.device), torch.arange(0, self.N - 1, 1, device=self.device), dim=0)
        idxs2 = torch.cat(torch.zeros(2, device=self.device), torch.arange(0, self.N - 2, 1, device=self.device), dim=0)

        if self.roots: 
            ins = idxs0 * idxs1 * (x.unsqueeze(1).repeat(1, roots.shape[1], 1) - roots.repeat(1, 1, x.shape[1])) ** idxs2.unsqueeze(0).unsqueeze(-1) # X = B, X and idxs : N x ** idxs  should be B, X, N 
            xs = torch.einsum('BNX, BNC -> BXC', ins, coeffs)
        else:
            ins = idxs0 * idxs1 * (x.unsqueeze(-1)) ** idxs2 # X = B, X and idxs : N x ** idxs  should be B, X, N 
            xs = torch.einsum('BXN, BNC -> BXC', ins, coeffs)

        return xs
    
    def get_basis(self, x): 
        coeffs = theta[:, :self.N]
        if self.roots:
            roots = theta[:, self.N:] 

        idxs = torch.arange(0, self.N, 1, device=self.device)

        if self.roots: 
            ins = (x.unsqueeze(1).repeat(1, roots.shape[1], 1) - roots.repeat(1, 1, x.shape[1])) ** idxs.unsqueeze(0).unsqueeze(-1) # X = B, X and idxs : N x ** idxs  should be B, X, N 
        else:
            ins = (x.unsqueeze(-1)) ** idxs # X = B, X and idxs : N x ** idxs  should be B, X, N 

        return ins
    
    def get_basis_derivativex(self, x):
        idxs0 = torch.arange(0, self.N, 1, device=self.device)
        idxs1 = torch.cat(torch.zeros(1, device=self.device), torch.arange(0, self.N - 1, 1, device=self.device), dim=0)

        if self.roots: 
            ins = idxs0 * (x.unsqueeze(1).repeat(1, roots.shape[1], 1) - roots.repeat(1, 1, x.shape[1])) ** idxs1.unsqueeze(0).unsqueeze(-1) # X = B, X and idxs : N x ** idxs  should be B, X, N 
        else:
            ins = idxs0 * (x.unsqueeze(-1)) ** idxs1 # X = B, X and idxs : N x ** idxs  should be B, X, N 
        return ins

    def get_basis_derivativex2(self, x):
        coeffs = theta[:, :self.N]
        if self.roots:
            roots = theta[:, self.N+2:] 
        bsize, xsize = x.shape

        idxs0 = torch.arange(0, self.N, 1, device=self.device)
        idxs1 = torch.cat(torch.zeros(1, device=self.device), torch.arange(0, self.N - 1, 1, device=self.device), dim=0)
        idxs2 = torch.cat(torch.zeros(2, device=self.device), torch.arange(0, self.N - 2, 1, device=self.device), dim=0)

        if self.roots: 
            ins = idxs0 * idxs1 * (x.unsqueeze(1).repeat(1, roots.shape[1], 1) - roots.repeat(1, 1, x.shape[1])) ** idxs2.unsqueeze(0).unsqueeze(-1) # X = B, X and idxs : N x ** idxs  should be B, X, N 
        else:
            ins = idxs0 * idxs1 * (x.unsqueeze(-1)) ** idxs2 # X = B, X and idxs : N x ** idxs  should be B, X, N 

        return ins
