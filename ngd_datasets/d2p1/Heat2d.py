"""
Adapted from https://github.com/pdebench/PDEBench/tree/main
# TODO : Reaction Diffusion 
"""

import torch
import h5py
import numpy as np
import yaml
import matplotlib.pyplot as plt
import einops

from utils.data import apply_normalization, remove_normalization
from ngd_datasets.abstract import PDE

class Heat2d(PDE):
    def __init__(self, cfg, ntraj, mode='train'):
        super().__init__(cfg, ntraj, mode)

        reduced_resolution_t = cfg.sub_from_t
        reduced_resolution_x = cfg.sub_from_x
        reduced_resolution_y = cfg.sub_from_y

        ds = np.load(f'/data/lise.leboudec/datasets/mae-pdes/data_gen/data/heat_Jmax_{mode}_1000.npz')        
        
        self.labels = torch.from_numpy(ds['u'][0:ntraj, ::reduced_resolution_x, ::reduced_resolution_y, ::reduced_resolution_t]).unsqueeze(-1) # NXYT
        self.datas = torch.from_numpy(ds['params'][0:ntraj, ...]) # N2
        # self.gridxy = torch.from_numpy(ds['xs'][::reduced_resolution_x, ::reduced_resolution_y]) # NX
        self.gridt = torch.from_numpy(ds['ts'][::reduced_resolution_t]) # NT

        self.gridx = torch.linspace(0, 1, 64)[::reduced_resolution_x]
        self.gridy = torch.linspace(0, 1, 64)[::reduced_resolution_y]
        self.gridt = self.gridt / 2

        self.nx = self.gridx.shape[0]
        self.ny = self.gridy.shape[0]
        self.nt = self.gridt.shape[0]

        xx, yy, tt = torch.meshgrid([self.gridx, self.gridy, self.gridt], indexing='ij') # X, Y, T
        self.grid = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1), tt.unsqueeze(-1)), dim=-1) # X, Y, T, 3
        xx, yy, tt = torch.meshgrid([self.gridx, self.gridy, self.gridt[0]], indexing='ij') # X, 1
        self.gridic = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1), tt.unsqueeze(-1)), dim=-1) # X, Y, 1, 3
        self.gridbc = torch.cat((self.grid[0, :].unsqueeze(1), self.grid[:, 0].unsqueeze(1), self.grid[-1, :].unsqueeze(1), self.grid[:, -1].unsqueeze(1)), dim=1) # X, 4, T, 3
        self.gridin = self.grid[1:-1, 1:-1, 1:, ...] # X-2, Y-2, T-1, 3

        self.dim = 3
        self.channels = 1
                    
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        # return items as (params, forcings, ic, bc), grid, u
        params = self.datas[idx, :].unsqueeze(-1) # 3, 1
        ic = self.labels[idx, :, :, 0, :] # X, 1
        bc = self.labels[idx, ...].clone()
        bc[1:-1, 1:-1, ...] = 0
        forcings = torch.empty((0))
        datas = (params, forcings, ic, bc)

        x = (self.grid, self.gridin, self.gridic, self.gridbc)

        u = self.labels[idx, ...] # X, T, C

        return (datas, x, u, idx)

    def pde(self, x, datas, theta, uhat_f):
        pass
    
    def loss_pinns(self, x_in, x_ic, x_b, uhat_f, theta, params, lbd): 
        params, forcings, ic, bc = params

        lpde = self.loss_pde(x_in, uhat_f, theta, params) # B, 1
        lbc = self.loss_bc(x_ic, uhat_f, theta, ic) # B, 1

        return lpde + lbd * lbc, lpde, lbc

    def loss_pde(self, x, uhat_f, theta, params):
        pass

    def loss_bc(self, x_b, uhat_f, theta, ic):
        pass

    def get_sizes(self):
        self.frame_size = self.grid.shape[0] * self.grid.shape[1] * self.grid.shape[2]
        self.paramssize = 1
        self.forcingssize = 0
        self.icsize = self.grid.shape[0] * self.grid.shape[1]
        self.bcsize = (self.grid.shape[1] * 2 + self.grid.shape[0] * 2) * self.grid.shape[2]
        return self.frame_size, self.paramssize, self.forcingssize, self.icsize, self.bcsize

    def get_channels(self):
        self.frame_channels = self.channels
        self.paramschannels = 1
        self.forcingschannels = 0
        self.icchannels = 1
        self.bcchannels = 1
        return self.frame_channels, self.paramschannels, self.forcingschannels, self.icchannels, self.bcchannels


       
if __name__=="__main__":
    from omegaconf import DictConfig

    cfg = DictConfig({'path2data':'/data/lise.leboudec/datasets/Custom/output_datasets', 'name':'advection', 'ntrain': 800, 'normalization':0, 'beta': 0.1, 'sub_from_t': 10, 'sub_from_x': 4, 'sub_from_y': 4})
    ds = Heat2d(cfg, 800, 'train')
    ds[0]
    # print("ds[0] : ", ds[0]) 
    print(ds[0][0][0].shape, ds[0][0][1].shape, ds[0][0][2].shape, ds[0][0][3].shape)
    print(ds[0][1][0].shape, ds[0][1][1].shape, ds[0][1][2].shape, ds[0][1][3].shape)
    print(ds[0][2].shape)


