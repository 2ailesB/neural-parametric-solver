import torch.nn as nn
import torch

from components.base_layer.init import init_layers
from models.abstract import Model

class Preconditioner(Model):
    def __init__(self, cfg, *args, **kwargs) -> None:
        """ Preconditioner model : Predicts conditioning matrix P = NN(x)

        Args:
            cfg (omegaconf.dictconfig.DictConfig): configuration
        """

        super().__init__(cfg, *args, **kwargs)

        self.N = self.cfg.N

        assert self.cfg.input_theta + self.cfg.input_gradtheta + self.cfg.input_params + self.cfg.input_loss > 0, 'No input in net'

        self.cfg.nn.input_dim = self.get_input_size()
        self.cfg.nn.output_dim = (self.N) **2

        self.nn = init_layers(self.cfg.nn.name, self.cfg.nn)
        
    def forward(self, x):

        bsize = x.shape[0]
        assert x.shape[2]==1, 'Multi channel solver not implemented'

        return self.get_conditioner(x).reshape((bsize, self.N, self.N))
    
    def get_conditioner(self, x):
        """Simple conditioner P = NN(x)"""

        return self.nn(x.squeeze(dim=-1))
    
class Preconditioner_PINNs(Preconditioner):
    def __init__(self, cfg, *args, **kwargs) -> None:
        """ Preconditioner model : Predicts conditioning matrix P = tril(NN(x)) @ tril(NN(x))^T + I
        ie use a lower triangular matrix to predict the Cholesky factor of the preconditioner
        Regularized on preconditioner to allow test with identity preconditioner‡

        Args:
            cfg (omegaconf.dictconfig.DictConfig): configuration
        """
        super().__init__(cfg, *args, **kwargs)

        self.regularized = self.cfg.regularized
        

    def get_conditioner(self, x):
        bsize = x.shape[0]
        L = torch.tril(self.nn(x.squeeze(dim=-1)).reshape((bsize, self.N, self.N)))

        return self.regularized * torch.bmm(L.transpose(1, 2), L) + torch.eye(self.N, device=self.device) 
    
class Preconditioner_Cholesky(Preconditioner):
    def __init__(self, cfg, *args, **kwargs) -> None:
        """ Preconditioner model : Predicts conditioning matrix P = tril(NN(x)) @ tril(NN(x))^T + I
        ie use a lower triangular matrix to predict the Cholesky factor of the preconditioner
        Regularized on identity matrix

        Args:
            cfg (omegaconf.dictconfig.DictConfig): configuration
        """
        super().__init__(cfg, *args, **kwargs)

        self.regularized = self.cfg.regularized
        

    def get_conditioner(self, x):
        bsize = x.shape[0]
        L = torch.tril(self.nn(x.squeeze(dim=-1)).reshape((bsize, self.N, self.N)))

        return torch.bmm(L.transpose(1, 2), L) + self.regularized * torch.eye(self.N, device=self.device) 