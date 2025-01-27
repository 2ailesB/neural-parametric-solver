import torch
from torch.utils.data import Dataset

from utils.data import get_normalization_01, get_normalization_N01, apply_normalization, remove_normalization

class PDE(Dataset):
    def __init__(self, cfg, ntraj, mode='train'):
        """ General class for PDEs and datasets 
        
        Args:
            cfg (omegaconf.dictconfig.DictConfig): configuration
            ntraj (int): number of trajectories to use
            mode (str, optional): 'train' or 'test'. Defaults to 'train'.
        ‡"""
        super().__init__()

        self.cfg = cfg
        
        self.mode = mode
        self.ntraj = ntraj
        self.scheduled = False
        self.normalization = cfg.normalization
        self.init_normalization()

        self.m = 0
        self.s = 1

        self.dim = 0
        self.channels = 0

    def __len__(self):
        """ Return the number of trajectories of the dataset """
        return self.ntraj
    
    def __getitem__(self, index):
        """ Return trajectorie(s) at positions/number index 
        Returns:   
            tuple: (datas, x, u, idx) where
                datas (tuple): parameters of the PDE as
                    (params, forcings, ic, bc) where params are the parameters of the PDE, forcings the forcing terms, ic the initial condition and bc the boundary condition
                x (tuple): spatial coordinates as 
                    (x, xin, xic, xbc) where x is the grid, xin the interior points, xic the initial condition and xbc the boundary condition
                u (torch.Tensor): solution of the PDE sampled on the grid
                idx (int): index of the trajectory (to identify PDE and make comparisons)
        """
        return self.datas[index], self.labels[index]
    
    def pde(self, x, datas, theta, uhat_f):
        """ Return the PDE operator applied to a frame u and parameters a """
        pass

    def init_normalization(self):
        if not self.normalization:
            self.norm = 0
        elif self.normalization == 'normal':
            self.norm = get_normalization_N01
        elif self.normalization == '01':
            self.norm = get_normalization_01
        else :
            raise NotImplementedError(f'Normalization {self.normalization} not implemented')
    
    def get_sizes(self):
        """ define the sizes of the different components of the PDE 
        return in a tuple as (frame, params, forcings, ic, bc)"""
        pass

    def get_channels(self):
        """ define the channels of the different components of the PDE
        return in a tuple as (frame, params, forcings, ic, bc) """
        pass
    