import torch

from optimizer.Optimizer import Optimizer

class GD(Optimizer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.conditioned = cfg.conditioned

    def update(self, theta, nn, gradtheta=None):
        """Basic gradient descent update rule.
        2 modes: 
            - conditioned: theta_k+1 = theta_k - lr * nn(theta_k) @ gradtheta, ie NN = precondioning matrix
            - unconditioned: theta_k+1 = theta_k - lr * nn(theta_k). The One used in the paper. """
        
        if self.conditioned:
            assert gradtheta != None
            return theta - self.lr * torch.bmm(nn, gradtheta) 
        else :
            return theta - nn * self.lr