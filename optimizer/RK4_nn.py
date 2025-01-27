import torch

from optimizer.Optimizer import Optimizer
# https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods

class RK4_nn(Optimizer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def update(self, theta, nn, gradtheta=None):
        """ Runge-Kutta 4th order update rule. 
        One network for ALL intermediate quantities, ie k1, k2, k3, k4."""

        n = theta.shape[1]
        gt = nn
        k1 = nn[:, :n]
        k2 = nn[:, n:2*n]
        k3 = nn[:, 2*n:3*n]
        k4 = nn[:, 3*n:]

        return theta + self.lr / 6 * (k1 + 2*k2 + 2*k3 + k4)

    def re_init(self):
        pass

    def get_update_size(self):
        return 4

class RK4_nns(Optimizer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def update(self, theta, nn, gradtheta=None):
        """ Runge-Kutta 4th order update rule.
        One network for EACH intermediate quantities, ie k1, k2, k3, k4."""

        k1 = nn[0]
        k2 = nn[1]
        k3 = nn[2]
        k4 = nn[3]

        return theta + self.lr / 6 * (k1 + 2*k2 + 2*k3 + k4)

    def re_init(self):
        pass

    def get_update_size(self):
        return 1