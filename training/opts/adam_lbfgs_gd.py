from torch.optim import Adam, LBFGS, Optimizer
from .gd import GD

class Adam_LBFGS_GD(Optimizer):
    """ Custom optimizer that switches between Adam, LBFGS and GD optimizers at different epochs.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        switch_epoch1 (int): epoch at which to switch from Adam to LBFGS.
        switch_epoch2 (int): epoch at which to switch from LBFGS to GD.
        adam_params (dict): parameters for Adam optimizer.
        lbfgs_params (dict): parameters for LBFGS optimizer.
        gd_params (dict): parameters for GD optimizer.
    """
    def __init__(self, params, switch_epoch1, switch_epoch2, adam_params, lbfgs_params, gd_params):

        self.switch_epoch1 = switch_epoch1
        self.switch_epoch2 = switch_epoch2
        self.params = list(params)
        self.adam = Adam(self.params, **adam_params)
        self.lbfgs = LBFGS(self.params, **lbfgs_params)
        self.gd = GD(self.params, **gd_params)

        super(Adam_LBFGS_GD, self).__init__(self.params, defaults={})

        self.state['epoch'] = 0

    def step(self, closure=None):
        if self.state['epoch'] < self.switch_epoch1:
            self.adam.step(closure)
            self.state['epoch'] += 1

        elif self.state['epoch'] < self.switch_epoch2:
            if self.state['epoch'] == self.switch_epoch1:
                print(f'Switching to LBFGS optimizer at epoch {self.state["epoch"]}')
            self.lbfgs.step(closure)
            self.state['epoch'] += 1
            
        else:
            if self.state['epoch'] == self.switch_epoch2:
                print(f'Switching to GD optimizer at epoch {self.state["epoch"]}')
            _, grad = self.gd.step(closure)
            self.state['epoch'] += 1
            return grad

        