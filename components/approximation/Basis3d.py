import torch
import einops

from components.approximation.FunctionBasis import FunctionBasis
from components.approximation.init import init_1dbasis
from utils.gradients import gradients, finite_diff

class Basis3d(FunctionBasis):
    def __init__(self, cfg) -> None:

        """TODO : implémenté uniquement en 2d à plusieurs endroits"""
        super().__init__(cfg)

        self.dim = cfg.dim
        assert self.dim == 3, 'dim other than 3 not implemented' 
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
        cp = n[0] * n[1] + n[0] * n[2] + n[1] * n[2] # TODO: moche

        return (n.sum() + n.prod() + cp).item() # 2 * Ns if term **1, **2 + prod for crossed terms # TODO: marche pas avec les biasi

    def compute_u(self, x, theta):
        """
        x = BXD, theta = BNC
        return u = BXC
        WARNING ONLY IN 3D # TODO
        """
        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        basis = []
        # compute bases
        for i, b in enumerate(self.bases):
            basis += [b.get_basis(x[:, :, i])] # BXN : add term of order 1 

        # add termes croisés
        cb3 = einops.einsum(basis[0], basis[1], basis[2], 'B X N, B X M, B X O -> B X N M O') # B X N M O
        cb2a = einops.einsum(basis[0], basis[1], 'B X N, B X M -> B X N M')
        cb2b = einops.einsum(basis[0], basis[2], 'B X N, B X M -> B X N M')
        cb2c = einops.einsum(basis[1], basis[2], 'B X N, B X M -> B X N M')
        cb3 = einops.rearrange(cb3, 'B X ... -> B X (...)') # B X N M O
        cb2a = einops.rearrange(cb2a, 'B X ... -> B X (...)')
        cb2b = einops.rearrange(cb2b, 'B X ... -> B X (...)')
        cb2c = einops.rearrange(cb2c, 'B X ... -> B X (...)')
        
        basis = torch.cat((basis[0], basis[1], basis[2], cb3, cb2a, cb2b, cb2c), dim=2) # basis[0]**2, basis[1]**2, # BXN

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
        dy = gradients(self.compute_u(x, theta), x)[:, :, 1] # B, XT, 2
        dz = gradients(self.compute_u(x, theta), x)[:, :, 2] # B, XT, 2
        if not requires_grad_after:
            x.requires_grad_(False)

        return dx, dy, dz # d[..., 0], d[..., 1] # separate dx, dt

    def compute_uderivativex2(self, x, theta, requires_grad_after=False):

        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        # In dim > 1, we systematiccaly use autograd to compute the derivatives
        if not requires_grad_after:
            x = x.detach().clone()
        x.requires_grad_(True)
        dx2 = gradients(gradients(self.compute_u(x, theta), x)[:, :, 0], x) # B, XT, 2
        dy2 = gradients(gradients(self.compute_u(x, theta), x)[:, :, 1], x) # B, XT, 2
        dz2 = gradients(gradients(self.compute_u(x, theta), x)[:, :, 2], x) # B, XT, 2
        if not requires_grad_after:
            x.requires_grad_(False)

        return (dx2[:, :, 0], dx2[:, :, 1], dx2[:, :, 2]), (dy2[:, :, 0], dy2[:, :, 1], dy2[:, :, 2]), (dz2[:, :, 0], dz2[:, :, 1], dz2[:, :, 2])
    
    def get_basis(self, x):
        basis = []
        # compute bases
        for i, b in enumerate(self.bases):
            basis += [b.get_basis(x[:, :, i])] # BXN : add term of order 1 
            # print("b.get_basis(x[..., i]).shape : ", b.get_basis(x[..., i]).shape) 

        # add termes croisés
        cb3 = einops.einsum(basis[0], basis[1], basis[2], 'B X N, B X M, B X O -> B X N M O') # B X N M O
        cb2a = einops.einsum(basis[0], basis[1], 'B X N, B X M -> B X N M')
        cb2b = einops.einsum(basis[0], basis[2], 'B X N, B X M -> B X N M')
        cb2c = einops.einsum(basis[1], basis[2], 'B X N, B X M -> B X N M')
        cb3 = einops.rearrange(cb3, 'B X ... -> B X (...)') # B X N M O
        cb2a = einops.rearrange(cb2a, 'B X ... -> B X (...)')
        cb2b = einops.rearrange(cb2b, 'B X ... -> B X (...)')
        cb2c = einops.rearrange(cb2c, 'B X ... -> B X (...)')
        basis = torch.cat((basis[0], basis[1], basis[2], cb3, cb2a, cb2b, cb2c), dim=2) # basis[0]**2, basis[1]**2, # BXN

        bias = torch.zeros(1).to(self.device)
        if self.bias:
            bias = theta[:, -1]
            theta = theta[:, :-1]
        return basis # B, X, N
    
    def get_basis_derivativex(self, x):
        raise NotImplementedError('get_basis_derivativex not implemented for Basis3d')

    def get_basis_derivativex2(self, x):
        raise NotImplementedError('get_basis_derivativex2 not implemented for Basis3d')
    

if __name__=="__main__":
    from omegaconf import DictConfig
    t = y = x = torch.linspace(0, 1.0, 10)
    xx, yy, tt = torch.meshgrid([x, y, t], indexing='ij') # X, T
    # print("xx.shape, yy.shape, tt.shape : ", xx.shape, yy.shape, tt.shape)
    grid = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1), tt.unsqueeze(-1)), dim=-1)
    print("grid.shape : ", grid.shape)

    cfg = DictConfig({'channels': 1, 'dim': 3, 'order': 2, 'degree': 3, 'N': [10, 11, 12], 'knots_type': 'shifted', 'open_knots': False, 'add_bias': False, 'name': 'bsplines'})
    print("cfg : ", cfg)
    base = Basis3d(cfg)
    print("base : ", base)
    print("base.get_theta_size() : ", base.get_theta_size())
    theta = torch.randn((1, base.get_theta_size(), 1))
    print("theta.shape : ", theta.shape)
    # print("einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D').shape : ", einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D').shape)
    print(base.compute_u(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta).shape) # BXC
    print(base.compute_uderivativex(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0].shape) # BXC
    print(base.compute_uderivativex(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1].shape) # BXC
    print(base.compute_uderivativex(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[2].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][0].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][1].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][2].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][0].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][1].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][2].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[2][0].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[2][1].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[2][2].shape) # BXC