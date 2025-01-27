import torch

from optimizer.Optimizer import Optimizer

class NoOpt(Optimizer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def update(self, theta, nn, gradtheta=None):
        """ No optimization scheme, just return the output of the neural network, ie 
        theta_k+1 = nn(theta_k) """
        
        return nn