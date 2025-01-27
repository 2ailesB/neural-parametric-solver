import torch
import torch.nn as nn

from components.base_layer.init import init_layers
from models.abstract import Model


class Iterative(Model):
    def __init__(self, cfg, *args, **kwargs) -> None:
        """Iterative model : Predicts conditioning matrix 

        Args:
            cfg (omegaconf.dictconfig.DictConfig): configuration
        """

        super().__init__(cfg, *args, **kwargs)

        self.N = self.cfg.N

        assert self.cfg.input_um1 + self.cfg.input_x + self.cfg.input_params + self.cfg.input_pde > 0, 'No input in net'

        self.cfg.nn.input_dim =  self.get_input_size()
        
        self.cfg.nn.output_dim = (self.N) ** 2

        self.nn = init_layers(self.cfg.nn.name, self.cfg.nn)
        
    def forward(self, x):

        bsize = x.shape[0]
        assert x.shape[2]==1, 'Multi channel solver not implemented'

        return self.get_conditioner(x).reshape((bsize, self.N, self.N))
    
    def get_conditioner(self, x):
        """Simple conditioner P = NN(x)"""

        return self.nn(x.squeeze(dim=-1))
