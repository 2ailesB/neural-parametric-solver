import torch

from optimizer.Optimizer import Optimizer

class Newton(Optimizer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.conditioned = cfg.conditioned

    def update(self, theta, nn, gradtheta=None, loss=None):
        """ Newton optimizer update rule. """
        
        if self.conditioned:
            assert gradtheta != None
            return torch.where(torch.abs(gradtheta)<=1e-8, theta, theta - self.lr * loss.unsqueeze(-1) / gradtheta) 
        else :
            return torch.where(torch.abs(gradtheta)<=1e-8, theta, theta - self.lr * loss.unsqueeze(-1) / gradtheta)