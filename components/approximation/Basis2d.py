import torch
import einops

from components.approximation.FunctionBasis import FunctionBasis
from components.approximation.init import init_1dbasis
from utils.gradients import gradients, finite_diff

class Basis2d(FunctionBasis):
    def __init__(self, cfg) -> None:

        super().__init__(cfg)

        self.dim = cfg.dim
        assert self.dim == 2, 'dim other than 2 not implemented' 
        self.order = cfg.order

        self.Ns = self.cfg.N
        self.bases = []
        for N in self.Ns:
            self.cfg.N = N
            self.bases += [init_1dbasis(self.cfg.name, self.cfg)]

        self.Ns4tr = tuple(i+1  for i in self.Ns)
        assert len(self.Ns)==self.dim, f'Need to specify as many bases ({len(self.Ns)}) as dims ({self.dim}).'
        
        self.cfg.N = self.Ns

    def to(self, device):
        self.device=device
        [b.to(self.device) for b in self.bases]

    def get_theta_size(self):
        n=[]
        for b in self.bases:
            n+= [b.get_theta_size()]

        n = torch.tensor(n)

        return (n.sum() + n.prod()).item() # 2 * Ns if term **1, **2 + prod for crossed terms # TODO: marche pas avec les biasi

    def compute_u(self, x, theta):
        """
        x = BXD, theta = BNC
        return u = BXC
        """

        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        basis = []
        # compute bases
        for i, b in enumerate(self.bases):
            basis += [b.get_basis(x[:, :, i])] # BXN : add term of order 1 and 2

        # add crossed terms
        cb = torch.einsum('BXNT, BXTM -> BXNM', basis[0].unsqueeze(-1), basis[1].unsqueeze(-2))
        cb = einops.rearrange(cb, 'B X ... -> B X (...)') # "squeeze" dims

        basis = torch.cat((basis[0], basis[1], cb), dim=2) # basis[0]**2, basis[1]**2, # BXN

        bias = torch.zeros(1).to(self.device)
        if self.bias:
            bias = theta[:, -1]
            theta = theta[:, :-1]

        out = torch.bmm(basis, theta) # 'BXN, BNC -> BXC'

        return out + self.bias * bias.unsqueeze(-1)

    def compute_uderivativex(self, x, theta, requires_grad_after=False):
        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        # In dim > 1, we systematiccaly use autograd to compute the derivatives
        if not requires_grad_after:
            x = x.detach().clone()
        x.requires_grad_(True)
        dx = gradients(self.compute_u(x, theta), x)[:, :, 0] # B, XT, 2
        dt = gradients(self.compute_u(x, theta), x)[:, :, 1] # B, XT, 2
        if not requires_grad_after:
            x.requires_grad_(False)

        return dx, dt # d[..., 0], d[..., 1] # separate dx, dt

    def compute_uderivativex2(self, x, theta, requires_grad_after=False):

        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        if not requires_grad_after:
            x = x.detach().clone()
        x.requires_grad_(True)

        # In dim > 1, we systematiccaly use autograd to compute the derivatives
        dx2 = gradients(gradients(self.compute_u(x, theta), x)[:, :, 0], x) # B, XT, 2
        dt2 = gradients(gradients(self.compute_u(x, theta), x)[:, :, 1], x) # B, XT, 2
        if not requires_grad_after:
            x.requires_grad_(False)

        return dx2[:, :, 0], dx2[:, :, 1], dt2[:, :, 0], dt2[:, :, 1]
    
    def get_basis(self, x):
        basis = []
        # compute bases
        for i, b in enumerate(self.bases):
            basis += [b.get_basis(x[:, :, i])] # BXN : add term of order 1 and 2

        # add crossed terms
        cb = torch.einsum('BXNT, BXTM -> BXNM', basis[0].unsqueeze(-1), basis[1].unsqueeze(-2))
        cb = einops.rearrange(cb, 'B X ... -> B X (...)') # "squeeze" dims

        basis = torch.cat((basis[0], basis[1], cb), dim=2) # basis[0]**2, basis[1]**2, # BXN

        bias = torch.zeros(1).to(self.device)
        if self.bias:
            bias = theta[:, -1]
            theta = theta[:, :-1]

        return basis
    
    def get_basis_derivativex(self, x):
        raise NotImplementedError('get_basis_derivativex not implemented for Basis2d')
    
    def get_basis_derivativex2(self, x):
        raise NotImplementedError('get_basis_derivativex2 not implemented for Basis2d')
    
