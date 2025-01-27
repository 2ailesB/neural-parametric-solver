import torch

from optimizer.Optimizer import Optimizer

class AdaBelief(Optimizer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.beta1 = cfg.beta1
        self.beta2 = cfg.beta2
        self.epsilon = cfg.epsilon
        self.mt = 0
        self.st = 0
        self.t = 0

    def update(self, theta, nn, gradtheta=None):
        """ AdaBelief optimizer update rule.""" 
        gt = nn
        self.t += 1
        self.mt = self.beta1 * self.mt + (1-self.beta1) * gt
        self.st = self.beta2 * self.st + (1-self.beta2) * (gt -self.mt)**2 + self.epsilon
        mth = self.mt / (1 - self.beta1**self.t)
        sth = self.st / (1 - self.beta2**self.t)

        return theta - self.lr * mth / (torch.sqrt(sth + self.epsilon))

    def re_init(self):
        self.t = 0
        self.mt = 0
        self.st = 0