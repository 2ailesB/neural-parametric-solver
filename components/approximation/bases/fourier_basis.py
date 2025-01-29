import torch
import torch.nn as nn

from components.approximation.FunctionBasis import FunctionBasis

class Fourier_basis(FunctionBasis):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        self.N = cfg.N
        self.Ns4tr = (self.get_theta_size(),)
        self.ff = cfg.ff

        self.dim=1
        self.channels=1

        self.device = torch.device('cpu')
        self.ks = torch.arange(1, self.N+1, 1, device=self.device)
    
    def to(self, device):
        self.device=device

    def get_theta_size(self):
        return 2*self.N+1

    def compute_u(self, x, theta):
        idxs = torch.arange(1, self.N+1, 1, device=self.device)
        # k * x
        ins = torch.einsum('Nj, k -> Njk', x, idxs)
        # sum thetak * sin/cos(k * x)
        cs = torch.einsum('Njk, Nkc -> Njc', torch.cos(self.ff * ins), theta[:, 1:self.N+1]) # j = x, n = channels, k = N chosen
        sn = torch.einsum('Njk, Nkc -> Njc', torch.sin(self.ff * ins), theta[:, self.N+1:])
        return theta[:, 0].repeat((1, x.shape[1])).unsqueeze(-1) + cs + sn
    
    def compute_uderivativex(self, x, theta):
        theta1N = theta[:, 1:self.N+1]
        thetaN2N = theta[:, self.N+1:]
        idxs = torch.arange(1, self.N+1, 1, device=self.device)
        # k * x
        ins = torch.einsum('Nj, k -> Njk', x, idxs)
        # thetakf
        kfs1 = self.ff * idxs.unsqueeze(-1) * theta1N
        kfs2 = self.ff * idxs.unsqueeze(-1) * thetaN2N
        # sum thetakf * sin/cos(k * x)
        cs = torch.einsum('Njk, Nkc -> Njc', torch.sin(self.ff * ins), -kfs1) # j = x, n = channels, k = N chosen
        sn = torch.einsum('Njk, Nkc -> Njc', torch.cos(self.ff * ins), kfs2)
        return cs + sn

    def compute_uderivativex2(self, x, theta):
        theta1N = theta[:, 1:self.N+1]
        thetaN2N = theta[:, self.N+1:]
        idxs = torch.arange(1, self.N+1, 1, device=self.device)
        # k * x
        ins = torch.einsum('Nj, k -> Njk', x, idxs)
        # thetakf
        kfs1 = self.ff**2 * idxs.unsqueeze(-1)**2 * theta1N
        kfs2 = self.ff**2 * idxs.unsqueeze(-1)**2 * thetaN2N
        # sum thetakf * sin/cos(k * x)
        cs = torch.einsum('Njk, Nkc -> Njc', torch.cos(self.ff * ins), -kfs1) # j = x, n = channels, k = N chosen
        sn = torch.einsum('Njk, Nkc -> Njc', torch.sin(self.ff * ins), -kfs2)
        return cs + sn
    
    def compute_uderivativetheta(self, x, theta):
        idxs = torch.arange(1, self.N+1, 1, device=self.device)
        # k * x
        ins = torch.einsum('Nj, k -> Njk', x, idxs)
        coss = torch.cos(ins * self.ff)
        sins = torch.sin(ins * self.ff)
        out = torch.cat((torch.ones(x.shape[0], 1), coss, sins), dim = 1) # 100, 2N+1
        return out

    def get_basis(self, x): 
        idxs = torch.arange(1, self.N+1, 1, device=self.device)
        # k * x
        ins = torch.einsum('Nj, k -> Njk', x, idxs)
        # sum thetak * sin/cos(k * x)
        cs = torch.cos(self.ff * ins)
        sn = torch.sin(self.ff * ins) # B, X, N

        return torch.cat((torch.ones(cs.shape[:2], device=self.device).unsqueeze(-1), cs, sn), dim=-1) # theta[:, 0].repeat((1, x.shape[1])).unsqueeze(-1) + cs + sn
        # return B, X, N
    
    def get_basis_derivativex(self, x):
        idxs = torch.arange(1, self.N+1, 1, device=self.device)
        # k * x
        ins = torch.einsum('Nj, k -> Njk', x, idxs)
        # thetakf
        kfs1 = self.ff**2 * idxs.unsqueeze(-1)**2
        kfs2 = self.ff**2 * idxs.unsqueeze(-1)**2
        # sum thetakf * sin/cos(k * x)
        cs = torch.einsum('Njk, Nkc -> Njc', torch.cos(self.ff * ins), -kfs1) # j = x, n = channels, k = N chosen
        sn = torch.einsum('Njk, Nkc -> Njc', torch.sin(self.ff * ins), -kfs2)
        return torch.cat((torch.zeros(cs.shape[:2], device=self.device).unsqueeze(-1), cs, sn), dim=-1) # theta[:, 0].repeat((1, x.shape[1])).unsqueeze(-1) + cs + sn

    def get_basis_derivativex2(self, x):
        idxs = torch.arange(1, self.N+1, 1, device=self.device)
        # k * x
        ins = torch.einsum('Nj, k -> Njk', x, idxs)
        # thetakf
        kfs1 = self.ff**2 * idxs.unsqueeze(-1)**2 
        kfs2 = self.ff**2 * idxs.unsqueeze(-1)**2 
        # sum thetakf * sin/cos(k * x)
        cs = torch.einsum('Njk, Nkc -> Njc', torch.cos(self.ff * ins), -kfs1) # j = x, n = channels, k = N chosen
        sn = torch.einsum('Njk, Nkc -> Njc', torch.sin(self.ff * ins), -kfs2)
        return torch.cat((torch.zeros(cs.shape[:2], device=self.device).unsqueeze(-1), cs, sn), dim=-1) # theta[:, 0].repeat((1, x.shape[1])).unsqueeze(-1) + cs + sn
