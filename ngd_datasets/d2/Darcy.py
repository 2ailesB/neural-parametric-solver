import torch
import einops
from torchviz import make_dot

from utils.data import apply_normalization, remove_normalization
from ngd_datasets.abstract import PDE
from utils.gradients import gradients, gradients_unused

class Darcy(PDE):
    def __init__(self, cfg, ntraj, mode='train'):
        """ Darcy flow dataset: see abstract class for detailed docstring
        """
        super().__init__(cfg, ntraj, mode)

        self.mode = mode
        self.sub_from_y = cfg.sub_from_y
        self.sub_from_x = cfg.sub_from_x

        path2ds= f'{cfg.path2data}/FNO/darcy_{mode}_64.pt' # [5000, 32, 32] 1000 in test

        ds = torch.load(path2ds)
        self.datas = ds['x'][0:ntraj, ::self.sub_from_x, ::self.sub_from_y, ...].unsqueeze(-1) # B, X, Y, C
        self.labels = ds['y'][0:ntraj, ::self.sub_from_x, ::self.sub_from_y, ...].unsqueeze(-1) # B, X, Y, C

        self.ntraj= self.labels.shape[0]
        assert self.ntraj == ntraj
        self.spatialres = self.labels.shape[1]

        self.channels = 1
        self.dim = 2

        self.gridx = torch.linspace(0, 1, self.spatialres).unsqueeze(-1) ## ATTENTION AU 1 ???
        self.gridy = torch.linspace(0, 1, self.spatialres).unsqueeze(-1)

        xx, yy = torch.meshgrid([self.gridx.squeeze(-1), self.gridy.squeeze(-1)], indexing='ij') # X, Y 
        self.grid = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1)), dim=-1) # X, Y, 2
        self.gridic = torch.empty((0)) # 0 Not IC in Darcy Flow (static, no time)
        self.gridbc = torch.cat((self.grid[0, :].unsqueeze(1), self.grid[:, 0].unsqueeze(1), self.grid[-1, :].unsqueeze(1), self.grid[:, -1].unsqueeze(1)), dim=1) # X, 4 2
        xx, yy = torch.meshgrid([self.gridx[1:-1].squeeze(-1), self.gridy[1:-1].squeeze(-1)], indexing='ij') # X-2, Y-2
        self.gridin = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1)), dim=-1) # X-2, Y-2, 2

        self.forcings = torch.ones((self.ntraj, self.spatialres, self.spatialres, 1))
        
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        params = self.datas[index, 1:-1, 1:-1, ...]
        ic = torch.empty((0))
        # bc = torch.cat((self.labels[index, 0, ...], self.labels[index, :, 0, ...], self.labels[index, -1, ...], self.labels[index, :, -1, ...]), dim=1).unsqueeze(-1)
        bc = self.labels[index, ...].clone()
        bc[1:-1, 1:-1, ...] = 0
        forcings = self.forcings[index, 1:-1, 1:-1, ...]
        datas = (params, forcings, ic, bc)

        x = (self.grid, self.gridin, self.gridic, self.gridbc)

        u = self.labels[index, ...]

        return (datas, x, u, index)
    
    def pde(self, x, datas, theta, uhat_f):
        params, forcings, ic, bc = datas
        (x, x_in, x_ic, x_bc) = x #  :-1 to remove boundary effects
        
        a = einops.rearrange(params, 'B X Y C -> B (X Y) C')
        forcings = einops.rearrange(forcings, 'B X Y C -> B (X Y) C')
        bsize, Ns, dim = x.shape
        x = x.detach().clone()
        x.requires_grad_(True)
        u_x, u_y = uhat_f.compute_uderivativex(x, theta, True) # B, XT
        dx = gradients(a*u_x.unsqueeze(-1), x)[:, :, 0] # B, XT, 2
        dy = gradients(a*u_y.unsqueeze(-1), x)[:, :, 1] # B, XT, 2
        x.requires_grad_(False)
        Du = dx + dy
        lpde = Du + forcings.squeeze(-1) # B, XT

        # bc = einops.rearrange(bc, 'B ... C -> B (...) C')
        uhat_bc = uhat_f.compute_u(x_bc, theta) # B, X, C
        bc = torch.zeros(uhat_bc.shape, device=uhat_bc.get_device())
        lbc = uhat_bc - bc

        return lpde, lbc
    
    def loss_pinns(self, x_in, x_ic, x_b, uhat_f, theta, params, lbd):
        params, forcings, ic, bc = params

        lpde = self.loss_pde(x_in, uhat_f, theta, params, forcings) # B, 1
        lbc = self.loss_bc(x_b, uhat_f, theta, bc) # B, 1

        return lpde + lbd * lbc, lpde, lbc

    def loss_pde(self, x, uhat_f, theta, params, forcings):
        a = einops.rearrange(params, 'B X Y C -> B (X Y) C')
        forcings = einops.rearrange(forcings, 'B X Y C -> B (X Y) C')
        bsize, Ns, dim = x.shape

        # V1, hand made
        # by dev, auxx + axux + auyy + ayuy
        # u_xx, _, _, u_yy = uhat_f.compute_uderivativex2(x, theta) # B, XT 
        # Du = a*u_xx + ax*u_x + a*u_yy + ay*u_y # TODO: FD for ax, ay 

        # V2 autograd
        # d(aux)/dx d(auy)/dy
        x = x.detach().clone()
        x.requires_grad_(True)
        u_x, u_y = uhat_f.compute_uderivativex(x, theta, True) # B, XT
        dx = gradients(a*u_x.unsqueeze(-1), x)[:, :, 0] # B, XT, 2
        dy = gradients(a*u_y.unsqueeze(-1), x)[:, :, 1] # B, XT, 2
        x.requires_grad_(False)
        Du = dx + dy # + bc we computed + nabla(a*nablau) instead of - nabla(a*nablau)

        F_g = Du + forcings.squeeze(-1) # B, XT # + bc we computed + nabla(a*nablau) instead of - nabla(a*nablau)

        # Riemann approx
        loss = (F_g.unsqueeze(-1) ** 2).sum(dim=1) / Ns # B, 1

        return loss

    def loss_bc(self, x_b, uhat_f, theta, bc):
        bsize, Ns, dim = x_b.shape
        bc = einops.rearrange(bc, 'B ... C -> B (...) C')

        uhat_bc = uhat_f.compute_u(x_b, theta) # B, X, C

        bc = torch.zeros(uhat_bc.shape, device=uhat_bc.get_device())
        loss = ((uhat_bc - bc) ** 2).sum(1) / Ns # TODO essayer avec bc = data

        return loss
    
    def get_sizes(self):
        self.frame_size = self.grid.shape[0] * self.grid.shape[1]
        self.paramssize = self.gridin.shape[0] * self.gridin.shape[1]
        self.forcingssize = self.gridin.shape[0] * self.gridin.shape[1]
        self.icsize = 0
        self.bcsize = 2 * self.grid.shape[1] + 2 * self.grid.shape[0]

        return self.frame_size, self.paramssize, self.forcingssize, self.icsize, self.bcsize

    def get_channels(self):
        self.frame_channels = self.channels
        self.paramschannels = 1
        self.forcingschannels = 1
        self.icchannels = 0
        self.bcchannels = 4 
        return self.frame_channels, self.paramschannels, self.forcingschannels, self.icchannels, self.bcchannels

if __name__=="__main__":
    from omegaconf import DictConfig

    cfg = DictConfig({'name':'reactdiff', 'ntrain': 1000, 'normalization':0, 'nu': 0.5, 'rho': 1.0})
    ds = Darcy(cfg, 1000, 'train')
    print("ds : ", ds)
    params, x, u = ds[0]
    print("params, x, u.shape : ", params, x, u.shape)
    print("params[-1] : ", params[-1])