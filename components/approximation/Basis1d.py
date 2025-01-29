import torch
import einops

from components.approximation.FunctionBasis import FunctionBasis
from components.approximation.init import init_1dbasis
from utils.gradients import gradients, finite_diff

class Basis1d(FunctionBasis):
    """
    Basis1d class
    """
    def __init__(self, cfg) -> None:

        super().__init__(cfg)

        self.dim = cfg.dim
        assert self.dim == 1, 'dim other than 2 not implemented' 
        self.order = cfg.order

        self.Ns = self.cfg.N
        self.bases = []
        
        self.Ns4tr = (self.cfg.N, )
        self.base = init_1dbasis(self.cfg.name, self.cfg)
        self.Ns4tr = self.base.Ns4tr

        assert len(self.Ns4tr)==self.dim, f'Need to specify as many bases ({len(self.Ns)}) as dims ({self.dim}).'
        
        self.cfg.N = self.Ns

    def to(self, device):
        self.device=device
        self.base.to(self.device)

    def get_theta_size(self):
        return self.base.get_theta_size()

    def compute_u(self, x, theta):
        """
        x = BXD, theta = BNC
        return u = BXC
        """
        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        if self.cfg.name=='hnet' or self.cfg.autograd=False:
            return self.base.compute_u(x, theta)

        basis = []
        # compute bases
        basis = self.base.get_basis(x[:, :, 0]) 

        bias = torch.zeros(1).to(self.device)
        if self.bias:
            bias = theta[:, -1]
            theta = theta[:, :-1]

        out = torch.bmm(basis, theta) # 'BXN, BNC -> BXC'

        return out + self.bias * bias.unsqueeze(-1)

    def compute_uderivativex(self, x, theta, requires_grad_after=False):
        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        if self.cfg.name=='hnet' or self.cfg.autograd=False:
            return self.base.compute_uderivativex(x, theta)

        if not requires_grad_after:
            x = x.detach().clone()
        x.requires_grad_(True)
        dx = gradients(self.compute_u(x, theta), x)# B, XT, 1
        if not requires_grad_after:
            x.requires_grad_(False)

        return dx # d[..., 0], d[..., 1] # separate dx, dt

    def compute_uderivativex2(self, x, theta, requires_grad_after=False):

        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        if self.cfg.name=='hnet' or self.cfg.autograd=False:
            return self.base.compute_uderivativex2(x, theta)

        if not requires_grad_after:
            x = x.detach().clone()
        x.requires_grad_(True)
        dx2 = gradients(gradients(self.compute_u(x, theta), x), x) # B, XT, 2
        if not requires_grad_after:
            x.requires_grad_(False)
        return dx2
    
    def get_basis(self, x):
        return self.base.get_basis(x[..., 0])
    
    def get_basis_derivativex(self, x):
       return self.base.get_basis_derivativex(x)
    
    def get_basis_derivativex2(self, x):
       return self.base.get_basis_derivativex2(x)
    

