import torch
import torch.nn as nn
import einops

from components.base_layer.fourier1d import FNO1d
from components.base_layer.fourier2d import FNO2d
from components.base_layer.fourier3d import FNO3d
from utils.device import get_device

class FNO(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        """ Fourier Neural Operator model """
        super(FNO, self).__init__()

        self.cfg = cfg
        self.dim = self.cfg.dim

        self.cfg.output_dim = 1
        self.model = self.init_FNO()
        self.coord_size = 0 # Updated during training

        self.device = get_device()

    def init_FNO(self):
        if self.dim==1:
            return FNO1d(modes=self.cfg.modes1, layers=self.cfg.layers,
                         fc_dim=self.cfg.fc_dim, in_dim=self.cfg.input_dim, out_dim=self.cfg.output_dim, 
                         act=self.cfg.activation)
        elif self.dim==2:
            return FNO2d(modes1=self.cfg.modes1, modes2=self.cfg.modes2, layers=self.cfg.layers,
                         fc_dim=self.cfg.fc_dim, in_dim=self.cfg.input_dim, out_dim=self.cfg.output_dim, 
                         act=self.cfg.activation, pad_ratio=self.cfg.pad_ratio)
        else:
            assert self.dim == 3, f'FNO not implemented for dim={self.dim}'
            return FNO3d(modes1=self.cfg.modes1, modes2=self.cfg.modes2, modes3=self.cfg.modes3, 
                         layers=self.cfg.layers, fc_dim=self.cfg.fc_dim, 
                         in_dim=self.cfg.input_dim, out_dim=self.cfg.output_dim, 
                         act=self.cfg.activation, pad_ratio=self.cfg.pad_ratio)

    def forward(self, x):

        bsize, sres, channels = x.shape[0], x.shape[1:-1], x.shape[-1]

        # 1d : input : (batch size, x_grid, 1) => (batch size, x_grid, 1)
        # 2d : input : (batch size, x_grid, y_grid, 2) => (batch size, x_grid, y_grid, 1)
        # 3d : input : (batch size, x_grid, y_grid, t_grid, 3) => (batch size, x_grid, y_grid, t_grid, 1)

        yhat = self.model(x)

        return yhat # B, XYZT, C
    
    