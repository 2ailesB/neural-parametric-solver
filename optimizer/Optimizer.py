import torch

class Optimizer():
    """ Base class for neural solver.
    
    Args:
        cfg (dict): configuration parameters.
    """
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.lr_init = self.lr = cfg.lr

        self.scheduler = cfg.scheduler

        self.device = 'cpu'

    def to(self, device):
        """ Send the optimizer to the device. """
        self.device=device

    def update(self, theta, **kwargs):
        """ Update rule for the parameters theta to be optimized. To be defined in each subclass. """
        pass

    def re_init(self):
        """ Reinitialize the optimizer lr. """
        self.lr = self.lr_init
        pass
        
    def get_update_size(self):
        """ Return the multiplicative factor for NN output size, eg for Adam or RK4 we need 2 or 4 times the size
        of the parameters for each intermediate step. """
        return 1

    def schedule(self):
        """ Attempt of learning rate scheduler. """
        if self.scheduler:
            self.lr *= 0.5


