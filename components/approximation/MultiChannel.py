import torch
import einops

from components.approximation.FunctionBasis import FunctionBasis
from components.approximation.init_nd import init_ndbasis

class MultiChannel(FunctionBasis):
    def __init__(self, cfg) -> None:
        """ MultiChannel Basis class
        Currently not used in paper
        Might be useful for future work
        Might contain bugs
        """
        super().__init__(cfg)

        self.channels = cfg.channels
        
        self.bases = [init_ndbasis(self.cfg.name, self.cfg) for _ in range(self.channels)]
        
        self.Ns = self.cfg.N
        self.Ns4tr = tuple(i+1  for i in self.Ns) # bases of the same size for now

    def to(self, device):
        self.device=device
        [b.to(self.device) for b in self.bases]

    def get_theta_size(self):
        n=0

        for b in self.bases:
            n += b.get_theta_size() 

        return n

    def compute_u(self, x, theta):
        """
        x = BXD, theta = BNC
        return u = BXC
        """
        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        u = torch.empty((bsize, sres, 0), device=self.device) 
        ns = 0
        for (i, base) in enumerate(self.bases):
            thetab = theta[:, ns:ns + base.get_theta_size()]
            ns += base.get_theta_size()
            u = torch.cat((u, base.compute_u(x, thetab)), dim=-1) # B, X, 1

        return u # B, X, C

    def compute_uderivativex(self, x, theta, requires_grad_after=False):
        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        u = ()
        ns = 0
        for base in self.bases:
            thetab = theta[:, ns:ns + base.get_theta_size()]
            ns += base.get_theta_size()
            u += (base.compute_uderivativex(x, thetab),)

        return u

    def compute_uderivativex2(self, x, theta, requires_grad_after=False):

        bsize, sres, dim = x.shape
        _, tsize, channels = theta.shape

        u = ()
        ns = 0
        for base in self.bases:
            thetab = theta[:, ns:ns + base.get_theta_size()]
            ns += base.get_theta_size()
            u += (base.compute_uderivativex2(x, thetab),)

        return u
    
    def get_basis(self, x):
        bsize, sres, dim = x.shape

        bases = torch.empty((bsize, sres, 0), device=self.device) # Consider channels are independant
        ns = 0
        for (i, base) in enumerate(self.bases):
            bases = torch.cat((bases, base.get_basis(x)), dim=-1) # B, X, N
        return bases
    
    def get_basis_derivativex(self, x):
        bsize, sres, dim = x.shape

        bases = torch.empty((bsize, sres, 0), device=self.device) # Consider channels are independant
        ns = 0
        for (i, base) in enumerate(self.bases):
            bases = torch.cat((bases, base.get_basis_derivativex(x)), dim=-1) # B, X, N
        return bases

    def get_basis_derivativex2(self, x):
        bsize, sres, dim = x.shape

        bases = torch.empty((bsize, sres, 0), device=self.device) # Consider channels are independant
        ns = 0
        for (i, base) in enumerate(self.bases):
            bases = torch.cat((bases, base.get_basis_derivativex2(x)), dim=-1) # B, X, N
        return bases







if __name__ == "__main__":
    from omegaconf import DictConfig
    t = y = x = torch.linspace(0, 1.0, 10)
    xx, yy, tt = torch.meshgrid([x, y, t], indexing='ij') # X, T
    # print("xx.shape, yy.shape, tt.shape : ", xx.shape, yy.shape, tt.shape)
    grid = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1), tt.unsqueeze(-1)), dim=-1)
    # grid = (grid, grid)
    print("grid.shape : ", grid.shape)

    cfg = DictConfig({'channels': 2, 'dim': 3, 'order': 2, 'degree': 3, 'N': [10, 11, 12], 'knots_type': 'shifted', 'open_knots': False, 'add_bias': False, 'name': 'bsplines'})
    print("cfg : ", cfg)
    base = MultiChannel(cfg)
    print("base : ", base)
    print("base.get_theta_size() : ", base.get_theta_size())
    theta = torch.randn((1, base.get_theta_size(), 1))

    print("theta.shape : ", theta.shape)
    # print("einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D').shape : ", einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D').shape)
    print(base.compute_u(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta).shape) # BXC
    print(len(base.compute_uderivativex(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)))
    print(base.compute_uderivativex(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][0].shape) # BXC
    print(base.compute_uderivativex(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][1].shape) # BXC
    print(base.compute_uderivativex(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][2].shape) # BXC
    print(base.compute_uderivativex(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][0].shape) # BXC
    print(base.compute_uderivativex(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][1].shape) # BXC
    print(base.compute_uderivativex(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][2].shape) # BXC
    print(len(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta))) # BXC
    print(len(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0])) # channel
    print(len(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][0])) # coordinate
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][0][0].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][0][1].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][0][2].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][1][0].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][1][1].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][1][2].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][2][0].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][2][1].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[0][2][2].shape) # BXC
    print(len(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1])) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][0][0].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][0][1].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][0][2].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][1][0].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][1][1].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][1][2].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][2][0].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][2][1].shape) # BXC
    print(base.compute_uderivativex2(einops.rearrange(grid.unsqueeze(0), 'B ... D -> B (...) D'), theta)[1][2][2].shape) # BXC