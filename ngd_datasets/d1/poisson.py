import torch
import numpy as np
from ngd_datasets.abstract import PDE

from utils.data import apply_normalization, remove_normalization


class Poisson(PDE):
    def __init__(self, cfg, ntraj=1000, mode='train'):
        """ Poisson PDE dataset : see abstract class for detailed docstring """
        super().__init__(cfg, ntraj, mode)

        if mode=='train':
            ds = np.load(f'{cfg.path2data}/poisson_N800_x64_rhssubsgbsTf_train_unif.npz')            
        elif mode == 'test':
            ds = np.load(f'{cfg.path2data}/poisson_N200_x64_rhssubsgbsTf_test_unif.npz')            
        else:
            raise ValueError(f'Unknown mode {mode}.')

        self.labels = torch.from_numpy(ds['u'][0:ntraj, ...]).unsqueeze(-1)
        self.datas = torch.from_numpy(ds['params'][0:ntraj, ...]).unsqueeze(-1)
        self.gridx = torch.from_numpy(ds['xs'][0:ntraj, ...]).unsqueeze(-1)
        self.mean = self.labels.mean()
        self.std = self.labels.std()

        self.ntraj = self.labels.shape[0]
        assert ntraj == self.ntraj
        self.spatialres = self.labels.shape[1]

        if self.norm:
            self.m, self.s = self.norm(self.datas[:, :-1], 0)
            self.datas[:, :-1] = apply_normalization(self.datas[:, :-1], self.m, self.s)

        self.channels = 1
        self.dim = 1

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        params = self.datas[index]

        u_b, g_b, rhs, T = params
        x = torch.linspace(0, T.item(), self.spatialres)
        # u_i = self.compute_u_from_data(x, u_b, g_b, omega).float()
        u_i = self.labels[index, :-1].float()

        datas = (params[[-1]].float(), params[[2]], torch.empty((0)), params[0:2].float())
        x = (x[:-1].unsqueeze(-1), x[1:-1].unsqueeze(-1), torch.empty((0)), x[[0]].unsqueeze(-1)) # x, xin, xic, xbc

        return datas, x, u_i, index

    def getgrid(self):
        """TODO : return only one grid ? 
        """
        return self.gridx
    
    def pde(self, x, params, theta, uhat_f):
        params, forcings, ic, bc = params
        (x, x_in, x_ic, x_b) = x #  :-1 to remove boundary effects

        u_b, g_b = bc[:, 0], bc[:, 1]
        rhs, T = forcings[:, 0], params[:, 0]  

        u_xx = uhat_f.compute_uderivativex2(x_in, theta)
        lpde = u_xx - rhs.unsqueeze(1).repeat(1, u_xx.shape[1], 1) # B, X, C

        uhat_b = uhat_f.compute_u(x_b, theta).squeeze(-1)
        gt_x = uhat_f.compute_uderivativex(x_b, theta).squeeze(-1)
        loss_ext = (uhat_b - u_b) # initial condition on u
        loss_d = (gt_x - g_b)
        loss_bc = loss_ext + loss_d

        return lpde, loss_bc
    
    def loss_pinns(self, x_in, x_ic, x_b, uhat_f, theta, params, lbd): # regrouper dans des params pour pv généraliser 
        params, forcings, ic, bc = params

        u_b, g_b = bc[:, 0], bc[:, 1]
        rhs, T = forcings[:, 0], params[:, 0]  

        lpde = self.loss_pde(x_in, uhat_f, theta, rhs, T)
        lbc = self.loss_bc(x_b, uhat_f, theta, u_b, g_b)

        return lpde + lbd * lbc, lpde, lbc

    def loss_pde(self, x, uhat_f, theta, rhs, T):
        """x = point of evaluation
        uhat = estimation of u on the points x
        omega = omega from the equation"""

        u_xx = uhat_f.compute_uderivativex2(x, theta)

        F_g = u_xx - rhs.unsqueeze(1).repeat(1, u_xx.shape[1], 1) # B, X, C

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
    
    def compute_u_from_data(self, x, u0, v0, rhs):
        a, b, c = rhs/2, v0, u0
        
        return a * x**2 + b * x + c

    def get_sizes(self):
        self.frame_size = self.spatialres - 1
        self.paramssize = self.datas.shape[1]
        self.forcingssize = 1
        self.icsize = 0
        self.bcsize = 2

        return self.frame_size, self.paramssize, self.forcingssize, self.icsize, self.bcsize
        
    def get_channels(self):
        self.frame_channels = self.channels
        self.paramschannels = self.datas.shape[2]
        self.forcingschannels = 1
        self.icchannels = 0
        self.bcchannels = 2
        return self.frame_channels, self.paramschannels, self.forcingschannels, self.icchannels, self.bcchannels