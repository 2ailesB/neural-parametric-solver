import torch

from optimizer.Optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.beta1 = cfg.beta1
        self.beta2 = cfg.beta2
        self.epsilon = cfg.epsilon
        self.mt = torch.zeros(1)
        self.vt = torch.zeros(1)
        self.t = 0

    def to(self, device):
        super().to(device)
        self.mt.to(self.device)
        self.vt.to(self.device)

    def update(self, theta, nn, gradtheta=None):

        """ Adam optimizer update rule.
            Use one ouput of the NN for the update rule.
        """ 

        mtm1 = self.mt.detach().clone()
        vtm1 = self.vt.detach().clone()
        gt = nn

        self.t += 1
        mt = self.beta1 * mtm1 + (1-self.beta1) * gt
        vt = self.beta2 * vtm1 + (1-self.beta2) * gt**2
        mth = mt # / (1 - self.beta1**self.t)
        vth = vt # / (1 - self.beta2**self.t)

        self.mt = mth
        self.vt = vth

        return theta - self.lr * mth / (torch.sqrt(vth + self.epsilon))

    def re_init(self):
        self.t = 0
        self.mt = torch.zeros(1).to(self.device)
        self.vt = torch.zeros(1).to(self.device)


class Adam_nns(Optimizer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.beta1 = cfg.beta1
        self.beta2 = cfg.beta2
        self.epsilon = cfg.epsilon
        self.mt = 0
        self.vt = 0
        self.t = 0

    def update(self, theta, nn, gradtheta=None):
        """ Adam optimizer update rule.
            2 NNs ouptus are needed for the update rule. 
        """
        self.t += 1
        gt = nn[0]
        gt2 = nn[1]
        mt = self.beta1 * self.mt + (1-self.beta1) * gt
        vt = self.beta2 * self.vt + (1-self.beta2) * gt2
        # mth = self.mt / (1 - self.beta1**self.t)
        # vth = self.vt / (1 - self.beta2**self.t)
        self.mt = mt.clone().detach()
        self.vt = vt.clone().detach()

        return theta - self.lr * mt / (torch.sqrt(vt + self.epsilon))

    def re_init(self):
        self.t = 0
        self.mt = 0
        self.vt = 0