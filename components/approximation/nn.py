import torch
import einops

from components.approximation.FunctionBasis import FunctionBasis
from components.base_layer.mlp import MLP_parametrized
from utils.gradients import gradients


class hnet_basis(FunctionBasis):
    def __init__(self, cfg) -> None:
        """ Use NN to reconstruct u as in original PINNs.
        Very simple case with only one hidden layer and one output neuron.
        Work in progress.""" 
        super().__init__(cfg)

        self.dim=1
        self.channels=1 
        if cfg.channels != 1 or self.dim != 1:
            raise NotImplementedError(f'NN basis in {cfg.dim}d and {cfg.channels} channels not implemented')

        self.hid = cfg.N

        self.Ns4tr = (self.get_theta_size(),)

        self.basis = MLP_parametrized(1, 1, [self.N], 'tanh')

    def compute_u(self, x, theta):
        bsize, xsize, channels = x.shape
        x = einops.rearrange(x, 'B X C -> (B X C)').unsqueeze(-1)
        theta = theta.repeat(xsize, 1, 1)
        out = self.basis(x, theta).squeeze(-1)
        return einops.rearrange(out, '(B X) -> B X', B = bsize).unsqueeze(-1)
    
    def compute_uderivativex(self, x, theta):
        bsize, xsize, channels = x.shape
        x = einops.rearrange(x, 'B X C -> (B X C)').unsqueeze(-1)
        theta = theta.repeat(xsize, 1, 1)

        Ws, bs = self.basis.get_weights_biases(theta)

        out = torch.bmm( Ws[1], (Ws[0].squeeze() * (1 - torch.tanh(torch.einsum('bji, bi -> bj', Ws[0], x) + bs[0])**2)).unsqueeze(-1))
        return einops.rearrange(out.reshape((bsize*xsize)), '(B X) -> B X', B = bsize).unsqueeze(-1)
    
    def compute_uderivativex2(self, x, theta):
        bsize, xsize, channels = x.shape
        x = einops.rearrange(x, 'B X C-> (B X C)').unsqueeze(-1)
        theta = theta.repeat(xsize, 1, 1)

        Ws, bs = self.basis.get_weights_biases(theta)
        
        out = torch.bmm( Ws[1], (Ws[0].squeeze()**2 * (-2)*torch.tanh(torch.einsum('bi, bji -> bj', x, Ws[0]) + bs[0]) / torch.cosh(torch.einsum('bi, bji -> bj', x, Ws[0]) + bs[0])**2).unsqueeze(-1))

        return einops.rearrange(out.squeeze(), '(B X) -> B X', B = bsize).unsqueeze(-1)


    def get_theta_size(self):
        return self.hid + self.hid + self.hid + 1 # W1, b1, W2, b2

    def get_basis(self, x): 
        raise NotImplementedError('get_basis not implemented for hnet_basis')
    
    def get_basis_derivativex(self, x):
        raise NotImplementedError('get_basis_derivativex not implemented for hnet_basis')

    def get_basis_derivativex2(self, x):
        raise NotImplementedError('get_basis_derivativex2 not implemented for hnet_basis')

    