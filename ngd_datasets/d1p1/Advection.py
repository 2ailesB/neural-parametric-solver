"""
Adapted from https://github.com/pdebench/PDEBench/tree/main
"""

import torch
import h5py
import numpy as np

from utils.data import apply_normalization, remove_normalization
from ngd_datasets.abstract import PDE

class Advection(PDE):
    def __init__(self, cfg, ntraj, mode='train'):
        """ Advection PDE dataset: see abstract class for detailed docstring """

        super().__init__(cfg, ntraj, mode)

        self.beta = cfg.beta # 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 4.0, 7.0
        h5_file = h5py.File(f'{cfg.path2data}/PDEBench/1D/Advection/Train/1D_Advection_Sols_beta{self.beta}.hdf5', "r")
        # (10000, 101, 1024) (1024,) (102,)
        
        reduced_resolution_t = cfg.sub_from_t
        reduced_resolution = cfg.sub_from_x

        # read files 
        _data = np.array(h5_file['tensor'], dtype=np.float32)
        if mode == 'train':
            _data = _data[:self.ntraj,::reduced_resolution_t,::reduced_resolution] # B, T, X
        elif mode == 'test':
            _data = _data[-self.ntraj:,::reduced_resolution_t,::reduced_resolution] # B, T, X
        else:
            raise NotImplemetedError(f'mode {mode} not existing')
        tdim = _data.shape[1]
        self.gridx = np.array(h5_file["x-coordinate"], dtype=np.float32)
        self.gridx = torch.tensor(self.gridx[::reduced_resolution], dtype=torch.float) # X
        self.gridt = np.array(h5_file["t-coordinate"], dtype=np.float32)
        self.gridt = torch.tensor(self.gridt[::reduced_resolution_t][:tdim], dtype=torch.float) # T

        # convert to [x1, ..., xd, t, v] + remove above 1
        _data = np.transpose(_data[:, :, :], (0, 2, 1)) # B, X, T
        self.data = torch.tensor(_data[..., None])  # batch, x, t, c
        self.data = self.data[:, self.gridx < 1, :, :]
        self.data = self.data[:, :, self.gridt < 1, :]
        self.gridx = self.gridx[self.gridx < 1].unsqueeze(-1) # X, 1 + remove 1 for basis
        self.gridt = self.gridt[self.gridt < 1].unsqueeze(-1) # T, 1

        xx, tt = torch.meshgrid([self.gridx.squeeze(-1), self.gridt.squeeze(-1)], indexing='ij') # X, T
        self.grid = torch.cat((xx.unsqueeze(-1), tt.unsqueeze(-1)), dim=-1) # X, T, 2
        xx, tt = torch.meshgrid([self.gridx.squeeze(-1), self.gridt[0].squeeze(-1)], indexing='ij') # X, 1
        self.gridic = torch.cat((xx.unsqueeze(-1), tt.unsqueeze(-1)), dim=-1) # X, 1, 2
        xx, tt = torch.meshgrid([self.gridx[[0, -1]].squeeze(-1), self.gridt.squeeze(-1)], indexing='ij') # 2, T
        self.gridbc = torch.cat((xx.unsqueeze(-1), tt.unsqueeze(-1)), dim=-1) # 2, T, 2
        xx, tt = torch.meshgrid([self.gridx[1:-1].squeeze(-1), self.gridt[1:].squeeze(-1)], indexing='ij') # X-2, T-1
        self.gridin = torch.cat((xx.unsqueeze(-1), tt.unsqueeze(-1)), dim=-1) # X-2, T-1, 2

        del xx, tt

        h5_file.close()

        self.dim = 2
        self.channels = 1
                    

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        # return items as (params, forcings, ic, bc), grid, u # TODO : faire ça sur GPU ? 
        params = torch.tensor([self.beta]).unsqueeze(-1) # 1, 1 (P, C)
        ic = self.data[idx, :, 0, :] # X, 1
        bc = torch.cat((self.data[idx, 0, :, :], self.data[idx, -1, :, :]), dim=1) # T, 2
        forcings = torch.empty((0))
        datas = (params, forcings, ic, bc)

        x = (self.grid, self.gridin, self.gridic, self.gridbc)

        u = self.data[idx, ...] # X, T, C

        return (datas, x, u, idx)

    def pde(self, x, datas, theta, uhat_f):
        params, forcings, ic, bc = datas
        (x, x_in, x_ic, x_bc) = x #  :-1 to remove boundary effects

        u_x, u_t = uhat_f.compute_uderivativex(x_in, theta)
        lpde = u_t + params.squeeze(-1) * u_x

        u0hat = uhat_f.compute_u(x_ic, theta)
        lic = u0hat - ic

        return lpde, lic
    
    def loss_pinns(self, x_in, x_ic, x_b, uhat_f, theta, params, lbd): # regrouper dans des params pour pv généraliser 
        params, forcings, ic, bc = params

        lpde = self.loss_pde(x_in, uhat_f, theta, params) # B, 1
        lbc = self.loss_bc(x_ic, uhat_f, theta, ic) # B, 1

        return lpde + lbd * lbc, lpde, lbc

    def loss_pde(self, x, uhat_f, theta, beta):
        bsize, Ns, dim = x.shape

        u_x, u_t = uhat_f.compute_uderivativex(x, theta) # B, XT

        F_g = u_t + beta.squeeze(-1) * u_x # B, XT

        # Riemann approx
        loss = (F_g.unsqueeze(-1) ** 2).sum(dim=1) / Ns

        return loss 

    def loss_bc(self, x_b, uhat_f, theta, ic):
        bsize, Ns, dim = x_b.shape
        uhat_ic = uhat_f.compute_u(x_b, theta) # B, X, C

        loss = ((uhat_ic - ic) ** 2).sum(1) / Ns

        return loss

    def get_sizes(self):
        self.frame_size = self.grid.shape[0] * self.grid.shape[1]
        self.paramssize = 1
        self.forcingssize = 0
        self.icsize = self.grid.shape[0]
        self.bcsize = self.grid.shape[1] * 2 # BC en x=0 et x=1
        return self.frame_size, self.paramssize, self.forcingssize, self.icsize, self.bcsize

    def get_channels(self):
        self.frame_channels = self.channels
        self.paramschannels = 1
        self.forcingschannels = 1
        self.icchannels = 1
        self.bcchannels = 2
        return self.frame_channels, self.paramschannels, self.forcingschannels, self.icchannels, self.bcchannels
       
if __name__=="__main__":
    from omegaconf import DictConfig

    cfg = DictConfig({'name':'advection', 'ntrain': 1000, 'normalization':0, 'beta': 0.1})
    ds = Advection(cfg, 1000, 'train')
    params, x, u = ds[0]
    print("params[0].shape : ", params[0].shape) # TODO : here !!!!!!
    # print("ds[0] : ", ds[0]) 
