import torch.nn as nn
import torch

from components.base_layer.init import init_layers
from models.abstract import Model


class Adam_NNs2(Model):
    def __init__(self, cfg, *args, **kwargs) -> None:
        """ Adam_NNs2 model : Use one network per component of the Adam optimizer

        Args:
            cfg (omegaconf.dictconfig.DictConfig): configuration
        """
        super().__init__(cfg, *args, **kwargs)

        self.N = self.cfg.N

        assert self.cfg.input_theta + self.cfg.input_gradtheta + self.cfg.input_params + self.cfg.input_splitted_grad > 0, 'No input in net'

        self.cfg.nn.input_dim = self.get_input_size()

        self.nn1 = init_layers(self.cfg.nn.name, self.cfg.nn)
        self.nn2 = init_layers(self.cfg.nn.name, self.cfg.nn)
        
    def forward(self, x):
        if x.shape[2]==1:
            return (self.nn1(x.squeeze(dim=-1)).unsqueeze(-1), self.nn2(x.squeeze(dim=-1)).unsqueeze(-1)) # B, N, C # voir squeeze selon modèles
        else:
            return (self.nn1(x), self.nn2(x))
    
