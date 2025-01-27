import torch
import numpy as np
from ngd_datasets.abstract import PDE
from utils.data import dynamics_different_subsample

class ForcingMSPoisson1d(PDE):
    def __init__(self, cfg, ntraj=1000, mode='train'):
        """ Poisson PDE dataset: see abstract class for detailed docstring
        """
        super().__init__(cfg, ntraj, mode)

        if mode=='train':
            ds = np.load(f'{cfg.path2data}/poisson_fxms_N800_x64_rhssubsgbsTf_{mode}_unif.npz')            
        elif mode == 'test':
            ds = np.load(f'{cfg.path2data}/poisson_fxms_N200_x64_rhssubsgbsTf_{mode}_unif.npz')            
        else:
            raise ValueError(f'Unknown mode {mode}.')
        self.labels = torch.from_numpy(ds['u'][0:ntraj, ...]).unsqueeze(-1)
        self.datas = torch.from_numpy(ds['params'][0:ntraj, ...]).unsqueeze(-1)
        self.forcings = torch.from_numpy(ds['forcings'][0:ntraj, ...]).unsqueeze(-1)
        self.gridx = torch.from_numpy(ds['xs'][0:ntraj, ...]).unsqueeze(-1)
        self.mean = self.labels.mean()
        self.std = self.labels.std()

        # self.labels = (self.labels - self.mean) / self.std
        # self.labels = (self.labels + self.labels.min()) / self.labels.max()

        self.ntraj = self.labels.shape[0]
        assert ntraj == self.ntraj
        self.spatialres = self.labels.shape[1]

        if mode == 'train' and cfg.sub_tr:
            self.labels, self.gridx, permutations = dynamics_different_subsample(self.labels, self.gridx, cfg.sub_tr)
            self.forcings = torch.gather(self.forcings, 1, permutations.unsqueeze(-1))
        elif mode == 'test' and cfg.sub_te :
            self.labels, self.gridx, permutations = dynamics_different_subsample(self.labels, self.gridx, cfg.sub_te)
            self.forcings = torch.gather(self.forcings, 1, permutations.unsqueeze(-1))
        else: 
            self.gridx = torch.linspace(0, 1, self.spatialres).unsqueeze(0).repeat(ntraj, 1) # BX

        self.channels = 1
        self.dim = 1

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        params = self.datas[index].float()
        forcings = self.forcings[index, :-1].float()

        u_b, g_b, ais, T = params[0], params[1], params[2:-1], params[-1] 
        # x = torch.linspace(0, T.item(), self.spatialres)
        x = self.gridx[index].squeeze(-1) # B, X, 1, 1

        u_i = self.labels[index, :-1].float()

        x = (x[:-1].unsqueeze(-1), x[1:-1].unsqueeze(-1), torch.empty((0)), x[[0]].unsqueeze(-1)) # x, xin, xic, xbc

        # params, forcings, ic, bc
        datas = (params[2:].float(), forcings, torch.empty((0)), params[0:2])

        return datas, x, u_i, index
    
    def pde(self, x, datas, theta, uhat_f):
        params, forcings, ic, bc = datas
        (x, x_in, x_ic, x_b) = x #  :-1 to remove boundary effects

        u_b, g_b = bc[:, 0], bc[:, 1]
        ais, T = params[:, 0:-1], params[:, -1]

        u_xx = uhat_f.compute_uderivativex2(x_in, theta)
        lpde = u_xx - forcings[:, 1:] # B, X, C

        uhat_b = uhat_f.compute_u(x_b, theta).squeeze(-1)
        gt_x = uhat_f.compute_uderivativex(x_b, theta).squeeze(-1)
        loss_ext = (uhat_b - u_b) # initial condition on u
        loss_d = (gt_x - g_b)
        loss_bc = loss_ext + loss_d

        return lpde, loss_bc
    
    def loss_pinns(self, x_in, x_ic, x_b, uhat_f, theta, datas, lbd): # regrouper dans des params pour pv généraliser 
        params, forcings, ic, bc = datas
        u_b, g_b = bc[:, 0], bc[:, 1]
        ais, T = params[:, 0:-1], params[:, -1]
        lpde = self.loss_pde(x_in, uhat_f, theta, forcings, T)
        lbc = self.loss_bc(x_b, uhat_f, theta, u_b, g_b)

        return lpde + lbd * lbc, lpde, lbc

    def loss_pde(self, x, uhat_f, theta, forcings, T):
        """x = point of evaluation
        uhat = estimation of u on the points x
        omega = omega from the equation"""

        u_xx = uhat_f.compute_uderivativex2(x, theta)
        F_g = u_xx - forcings[:, 1:] # B, X, C

        # Riemann approx
        loss = (F_g ** 2).sum(dim=1) * (x[:, 1]-x[:, 0]).unsqueeze(-1) / T

        return loss 

    def loss_bc(self, x_b, uhat_f, theta, u_b, g_b):
        """x_b = points on boundary conditions
        uhat_b = estimation of u on the points x_b
        u_b = true boundary condition on u
        g_b = true boundary condition on gradient"""

        # gt_x = gradients(uhat_b, x_b)[0]
        uhat_b = uhat_f.compute_u(x_b, theta).squeeze(-1)
        gt_x = uhat_f.compute_uderivativex(x_b, theta).squeeze(-1)

        loss_ext = ((uhat_b - u_b) ** 2) # initial condition on u
        loss_d = ((gt_x - g_b) ** 2) # initial condition on du/dx

        return (loss_ext + loss_d) / 2
     
    def get_sizes(self):
        self.frame_size = self.spatialres - 1
        self.paramssize = self.datas.shape[1]
        self.forcingssize = self.forcings.shape[1] - 1 
        self.icsize = 0
        self.bcsize = 2
        return self.frame_size, self.paramssize, self.forcingssize, self.icsize, self.bcsize
    
    def get_channels(self):
        self.frame_channels = self.channels
        self.paramschannels = self.datas.shape[2]
        self.forcingschannels = self.forcings.shape[2]
        self.icchannels = 0
        self.bcchannels = 2
        return self.frame_channels, self.paramschannels, self.forcingschannels, self.icchannels, self.bcchannels
    
if __name__=='__main__':
    from omegaconf import DictConfig

    cfg = DictConfig({'path2data':'/data/lise.leboudec/datasets/Custom/output_datasets', 'name':'forcingmspoisson', 
                    'ntrain': 1000, 'normalization':0, 'nu': 0.5, 'rho': 1.0,
                    'sub_from_x':1})
    ds = ForcingMSPoisson1d(cfg, 800, 'train')
    print("ds : ", ds)
    params, x, u, i = ds[0]
    print("params, u, i : ", params, u, i)
    ds = ForcingMSPoisson1d(cfg, 200, 'test')
    print("ds : ", ds)
    params, x, u, i = ds[0]
    print("params, u, i : ", params, u, i)