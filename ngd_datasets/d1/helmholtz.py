import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ngd_datasets.abstract import PDE
from utils.data import apply_normalization, remove_normalization, dynamics_different_subsample, dynamics_different_subsample_diffgrid
from utils.gradients import gradients

class Helmholtz(PDE):
    def __init__(self, cfg, ntraj=1000, mode='train'):
        """ Helmholtz PDE dataset """
        super().__init__(cfg, ntraj, mode)

        if self.cfg.name=='helmholtz-hf':
            omegamin=0.5
            omegamax=50
        elif self.cfg.name=='helmholtz-ood':
            omegamin=-5
            omegamax=55
        elif self.cfg.name=='helmholtz-low':
            omegamin=0.5
            omegamax=3
        else:
            omegamin=0.5
            omegamax=10
            
        if mode =='train':
            ds = np.load(f'{cfg.path2data}/helmholtz_N1024_x256_omegas{omegamin}_{omegamax}ubsgbsT_train_unif.npz')        
        elif mode == 'test': 
            ds = np.load(f'{cfg.path2data}/helmholtz_N1024_x256_omegas{omegamin}_{omegamax}ubsgbsT_test_unif.npz')        
        else:
            raise ValueError(f'Unknown mode {mode}.')

        self.labels = torch.from_numpy(ds['u'][0:ntraj, ::self.cfg.sub_from_x, ...]).unsqueeze(-1)
        self.spatialres = self.labels.shape[1]
        self.datas = torch.from_numpy(ds['params'][0:ntraj, ...]).unsqueeze(-1)
        self.numpydatas = ds['params'][0:ntraj, ...]
        self.gridx = torch.from_numpy(ds['xs'][::self.cfg.sub_from_x]).unsqueeze(-1)
        self.ntraj = self.labels.shape[0]

        if mode == 'train' and cfg.sub_tr:
            self.labels, self.gridx, _ = dynamics_different_subsample(self.labels, self.gridx, cfg.sub_tr)
        elif mode == 'test' and cfg.sub_te :
            self.labels, self.gridx, _ = dynamics_different_subsample(self.labels, self.gridx, cfg.sub_te)
        else: 
            self.gridx = torch.linspace(0, 1, self.spatialres).unsqueeze(0).repeat(ntraj, 1) # BX

        assert ntraj == self.ntraj

        self.channels = 1
        self.dim = 1


    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        params = self.datas[index]

        u_b, g_b, omega, T = params
        x = self.gridx[index].squeeze(-1) # B, X, 1, 1
        u_i = self.labels[index][:-1].float()

        datas = (params[2:].float(), torch.empty((0)), torch.empty((0)), params[0:2].float()) # (params, forcings, ic, bc)
        x = (x[:-1].unsqueeze(-1), x[1:-1].unsqueeze(-1), torch.empty((0)), x[[0]].unsqueeze(-1)) # x, xin, xic, xbc

        return datas, x, u_i, index

    def getgrid(self):
        return self.gridx
    
    def pde(self, x, params, theta, uhat_f):
        # compute pde residuals + bc residuals
        (x, x_in, x_ic, x_b) = x #  :-1 to remove boundary effects

        params, forcings, ic, bc = params
        u_b, g_b = bc[:, 0], bc[:, 1]
        omega, T = params[:, 0], params[:, 1]  

        uhat = uhat_f.compute_u(x_in, theta)
        u_xx = uhat_f.compute_uderivativex2(x_in, theta)
        wuhat = torch.einsum('bn, bxn -> bxn', omega**2, uhat) 
        loss_pde = u_xx + wuhat

        # gt_x = gradients(uhat_b, x_b)[0]
        uhat_b = uhat_f.compute_u(x_b, theta).squeeze(-1)
        gt_x = uhat_f.compute_uderivativex(x_b, theta).squeeze(-1)
        loss_ext = (uhat_b - u_b)  # initial condition on u
        loss_d = (gt_x - g_b)  # initial condition on du/dx
        loss_bc = loss_ext + loss_d
        
        return loss_pde, loss_bc
    
    def loss_pinns(self, x_in, x_ic, x_b, uhat_f, theta, params, lbd):
        params, forcings, ic, bc = params

        u_b, g_b = bc[:, 0], bc[:, 1]
        omega, T = params[:, 0], params[:, 1]  
        
        lpde = self.loss_pde(x_in, uhat_f, theta, omega, T)
        lbc = self.loss_bc(x_b, uhat_f, theta, u_b, g_b)
        
        return lpde + lbd * lbc, lpde, lbc

    def loss_pde(self, x, uhat_f, theta, omega, T):
        """x = point of evaluation
        uhat = estimation of u on the points x
        omega = omega from the equation"""

        x = x.requires_grad_(True)
        uhat = uhat_f.compute_u(x, theta)
        # u_x = gradients(uhat, x)
        # u_xx1 = gradients(u_x, x)
        u_xx = uhat_f.compute_uderivativex2(x, theta)

        wuhat = torch.einsum('bn, bxn -> bxn', omega**2, uhat) 
        F_g = u_xx + wuhat # B, X, C

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
    
    def compute_u_from_data(self, x, u0, v0, omega):
        if u0 == 0:
            alpha, beta = -v0 / omega, np.pi / 2
        else:
            beta = np.arctan(- v0 / (omega * u0))
            alpha =  u0 / np.cos(beta)
        
        return alpha * torch.cos(omega * x + beta).unsqueeze(-1)

    def get_sizes(self):
        self.frame_size = self.spatialres - 1
        self.paramssize = 1
        self.forcingssize = 0
        self.icsize = 0
        self.bcsize = 2

        return self.frame_size, self.paramssize, self.forcingssize, self.icsize, self.bcsize

    def get_channels(self):
        self.frame_channels = self.channels
        self.paramschannels = 1
        self.forcingschannels = 0
        self.icchannels = 0
        self.bcchannels = 2
        return self.frame_channels, self.paramschannels, self.forcingschannels, self.icchannels, self.bcchannels
        

if __name__=='__main__':
    from omegaconf import DictConfig

    cfg = DictConfig({'path2data':'/data/lise.leboudec/datasets/Custom/output_datasets', 'name':'helmholtz-hf', 
                    'ntrain': 1000, 'normalization':0, 'nu': 0.5, 'rho': 1.0,
                    'sub_from_x':4})
    ds = Helmholtz(cfg, 800, 'train')
    print("ds : ", ds)
    params, x, u, i = ds[0]
    print("params, u, i : ", params, u, i)
    ds = Helmholtz(cfg, 800, 'test')
    print("ds : ", ds)
    params, x, u, i = ds[0]
    print("params, u, i : ", params, u, i)