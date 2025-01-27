import torch.nn as nn
import torch

from components.base_layer.init import init_layers
from models.abstract import Model


class GD_NN(Model):
    def __init__(self, cfg, pde, *args, **kwargs) -> None:
        """GD_NN model : Directly predicts the gradient steps from inputs

        Args:
            cfg (omegaconf.dictconfig.DictConfig): configuration
            pde (PDE): the PDE configuration as :
            (PDE Physical losses, sizes (frame, params, forcings, ic, bc), channels (frame, params, forcings, ic, bc), PDE dim, PDE channels)
        """
        super().__init__(cfg, pde, *args, **kwargs)

        self.N = self.cfg.N

        assert self.cfg.input_theta + self.cfg.input_gradtheta + self.cfg.input_params + self.cfg.input_splitted_grad + self.cfg.input_forcings + self.cfg.input_bc + self.cfg.input_ic > 0, 'No input in net'

        self.cfg.nn.input_dim = self.get_input_size()
        
        self.cfg.nn.dim=self.dim
        self.nn = init_layers(self.cfg.nn.name, self.cfg.nn)
        
    def forward(self, x):
        if self.cfg.nn.name in ['mlp', 'siren', 'modmlp', 'deeponet', 'mfn_fourier', 'mfn_gabor', 'resnet']:
            return self.nn(x.squeeze(dim=-1)).unsqueeze(-1)
        return self.nn(x)

class GD_NN_step(Model):
    def __init__(self, cfg, *args, **kwargs) -> None:
        """GD_NN_step model : Directly predicts the gradient steps from inputs: one network per step

        Args:
            cfg (omegaconf.dictconfig.DictConfig): configuration
        """
        super().__init__(cfg, *args, **kwargs)

        self.N = self.cfg.N

        assert self.cfg.input_theta + self.cfg.input_gradtheta + self.cfg.input_params + self.cfg.input_splitted_grad > 0, 'No input in net'

        self.cfg.nn.input_dim = self.get_input_size()
        
        self.cfg.nn.dim=self.dim

        self.L = self.cfg.L
        self.nn = nn.ModuleList(init_layers(self.cfg.nn.name, self.cfg.nn) for _ in range(self.L))
        
    def forward(self, x, l=0):
        try:
            nn = self.nn[l]
        except IndexError:
            nn = self.nn[-1]
        if x.shape[2]==1:
            return nn(x.squeeze(dim=-1)).unsqueeze(-1) # B, N, C # voir squeeze selon modèles
        else:
            return nn(x)

