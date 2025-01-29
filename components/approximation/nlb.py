import torch
import einops

from components.approximation.FunctionBasis import FunctionBasis
from components.base_layer.mlp import MLP_parametrized
from utils.gradients import gradients
from components.approximation.init import init_1dbasis
from components.base_layer.init import init_layers


class nl_basis(FunctionBasis):
    def __init__(self, cfg) -> None:
        """ Non linear combination of basis functions"""
        super().__init__(cfg)

        self.dim=1
        self.channels=1

        self.basis = init_1dbasis(cfg.name, cfg)

        self.cfg.input_dim = self.basis.N
        self.cfg.output_dim = 1 
        self.hid = self.cfg.units
        self.Ns4tr = (self.basis.N * self.hid + self.hid + self.hid * 1 + 1,)

        self.net = MLP_parametrized(self.basis.N, 1, [self.cfg.units], 'tanh')


    def get_theta_size(self):
        return self.basis.N * self.hid + self.hid + self.hid * 1 + 1 # W1, b1, W2, b2

    def to(self, device):
        self.basis.to(device)
        super().to(device)

    def compute_u(self, x, theta):
        bsize, xsize, c = x.shape
        out = self.basis.get_basis(x.squeeze(-1))
        out = self.net.batched_forward(out, theta.squeeze(-1))
        return out
    
    def compute_uderivativex(self, x, theta):
        bsize, xsize, channels = x.shape
        """

        x.requires_grad_(True)
        u = self.compute_u(x, theta)
        # print("u.shape : ", u.shape)

        ux = gradients(u, x)
        # print("ux.shape : ", ux.shape)
        x.requires_grad_(False)

        return ux

        """

        theta = theta.repeat(xsize, 1, 1)
        out = self.basis.get_basis(x.squeeze(-1))
        out1 = self.basis.get_basis_derivativex(x.squeeze(-1)) # B, X, N
        out = einops.rearrange(out, 'B N C -> (B N) C') # BX N
        out1 = einops.rearrange(out1, 'B N C -> (B N) C') # BX N

        Ws, bs = self.net.get_weights_biases(theta)

        l1 = (torch.einsum('bn, bln -> bl', out, Ws[0]) + bs[0])


        out = torch.einsum('bn, bln -> bl', out1, Ws[0]) * self.actdx(l1)


        return einops.rearrange(torch.einsum('bl, bol -> bo', out, Ws[1]), '(B X) C -> B X C', B = bsize)
    

    def compute_uderivativex2(self, x, theta):
        bsize, xsize, channels = x.shape
        theta = theta.repeat(xsize, 1, 1) # BX
        out = self.basis.get_basis(x.squeeze(-1))
        out1 = self.basis.get_basis_derivativex(x.squeeze(-1)) # B, X, N
        out2 = self.basis.get_basis_derivativex2(x.squeeze(-1)) # B, X, N
        out = einops.rearrange(out, 'B N C -> (B N) C') # BX N
        out1 = einops.rearrange(out1, 'B N C -> (B N) C') # BX N
        out2 = einops.rearrange(out2, 'B N C -> (B N) C') # BX N

        Ws, bs = self.net.get_weights_biases(theta)

        l1 = (torch.einsum('bn, bln -> bl', out, Ws[0]) + bs[0])
        t1 = self.actdx(l1)
        t2 = self.actdx2(l1)
        t11 = torch.einsum('bn, bln -> bl', out2, Ws[0]) # BX, L1

        p1 = t1 * t11

        t21 = torch.einsum('bn, bln -> bl', out1, Ws[0])**2
        p2 = t2*t21

        return einops.rearrange(torch.einsum('bl, bol -> bo', p1+p2, Ws[1]), '(B X) C -> B X C', B = bsize)
        

    def get_basis(self, x): # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        raise NotImplementedError('get_basis not implemented for nl_basis')
    
    def get_basis_derivativex(self, x):
        raise NotImplementedError('get_basis_derivativex not implemented for nl_basis')

    def get_basis_derivativex2(self, x):
        raise NotImplementedError('get_basis_derivativex2 not implemented for nl_basis')

    def act(self, x):
        return torch.tanh(x)
    
    def actdx(self, x):
        return 1 - torch.tanh(x)**2
    
    def actdx2(self, x):
        return -2 * torch.tanh(x) * (1 - torch.tanh(x)**2)

    def projection(self, f, x):
        raise NotImplementedError('Projection not implemented for nl_basis')