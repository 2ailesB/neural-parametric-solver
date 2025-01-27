import torch.nn as nn
import torch

from components.base_layer.init import init_layers
from models.abstract import Model


class RK4_NNs(Model): 
    def __init__(self, cfg, *args, **kwargs) -> None:
        """ RK4_NNs model : One network per RK4 intermediate step. 
        Args:
            cfg (omegaconf.dictconfig.DictConfig): configuration
        """

        super().__init__(cfg, *args, **kwargs)

        self.N = self.cfg.N

        assert self.cfg.input_theta + self.cfg.input_gradtheta + self.cfg.input_params + self.cfg.input_splitted_grad > 0, 'No input in net'

        self.cfg.nn.input_dim = self.get_input_size()
        # self.cfg.nn.output_dim = self.N

        self.nn1 = init_layers(self.cfg.nn.name, self.cfg.nn)
        self.nn2 = init_layers(self.cfg.nn.name, self.cfg.nn)
        self.nn3 = init_layers(self.cfg.nn.name, self.cfg.nn)
        self.nn4 = init_layers(self.cfg.nn.name, self.cfg.nn)
        
    def forward(self, x):
        try:
            x.shape[2]==1
            x = x.squeeze(-1)
        except IndexError:
            print("x.shape : ", x.shape)
        
        thetak = x[:, :-1] # B, N
        k = x[:, [-1]] # B, 1
        
        k1 = self.nn1(torch.cat((thetak, k), 1))
        k2  = self.nn2(torch.cat((thetak + 1/2 * k1, k+1/2), 1))
        k3  = self.nn3(torch.cat((thetak + 1/2 * k2, k+1/2), 1))
        k4  = self.nn4(torch.cat((thetak + k3, k+1), 1))
        return (1 / 6 * (k1 + 2*k2 + 2*k3 + k4)).unsqueeze(-1) # B, N, C # voir squeeze selon modèles
    
    
class RK4_NNs2(Model):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        self.N = self.cfg.N

        assert self.cfg.input_theta + self.cfg.input_gradtheta + self.cfg.input_params + self.cfg.input_splitted_grad > 0, 'No input in net'

        self.cfg.nn.input_dim = self.get_input_size()
        # self.cfg.nn.output_dim = self.N

        self.nn1 = init_layers(self.cfg.nn.name, self.cfg.nn)
        self.nn2 = init_layers(self.cfg.nn.name, self.cfg.nn)
        self.nn3 = init_layers(self.cfg.nn.name, self.cfg.nn)
        self.nn4 = init_layers(self.cfg.nn.name, self.cfg.nn)
        
    def forward(self, x):
        if x.shape[2]==1:
            return (self.nn1(x.squeeze(dim=-1)).unsqueeze(-1), self.nn2(x.squeeze(dim=-1)).unsqueeze(-1), self.nn3(x.squeeze(dim=-1)).unsqueeze(-1), self.nn4(x.squeeze(dim=-1)).unsqueeze(-1)) # B, N, C # voir squeeze selon modèles
        else:
            return (self.nn1(x), self.nn2(x), self.nn3(x), self.nn4(x) )
    
