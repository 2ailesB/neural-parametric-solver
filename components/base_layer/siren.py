import numpy as np
import torch
import torch.nn as nn


class SIREN(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, weight_init_factor=0.01, **kwargs):
        """
        SIREN layer 
        from the Paper: Implicit Neural Representation with Periodic Activation Function (SIREN) (Sitzman et. al)
        :param in_features: number of input features
        :param out_features: number of output features
        :param bias: add a bias or not to the linear transformation
        :param is_first: first layer
        :param omega_0: pulsation of the sine activation
        """
        super().__init__()
        self.omega_0 = omega_0
        # self.omega_0_init = omega_0
        # self.omega_0 = nn.Parameter(torch.zeros((1)))
        self.weight_init_factor = weight_init_factor
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features * self.weight_init_factor,
                                            1 / self.in_features * self.weight_init_factor)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0 * self.weight_init_factor,
                                            np.sqrt(6 / self.in_features) / self.omega_0 * self.weight_init_factor)

    def forward(self, input):
        # print("input.shape : ", input.shape)
        return torch.sin(self.omega_0 * self.linear(input))


class SIRENs(nn.Module):
    def __init__(self, in_features, out_features, layers=None, bias=True, is_first=False, omega_0=30, weight_init_factor=0.01, **kwargs):
        super(SIRENs, self).__init__()

        self.model = []

        self.layers = layers if layers is not None else []
        self.model = nn.ModuleList([
            nn.Sequential(SIREN(lp, lnext, bias, is_first,
                          omega_0, weight_init_factor))
            for lp, lnext in zip([in_features] + self.layers, self.layers + [out_features])
        ])

    def forward(self, x):
        # x = x.squeeze(-1)
        for idx, layer in enumerate(self.model):
            # print("x.shape : ", x.shape)
            x = layer(x)
        return x # .unsqueeze(-1)


class SIREN_parametrized(nn.Module):
    def __init__(self, in_features, out_features, layers=None, omega_0=30.0, dropout_rate=0.0, **kwargs):
        """
        Usual SIREN module
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(SIREN_parametrized, self).__init__()
        self.layers = layers if layers is not None else []
        self.nlayers = len(layers)
        self.dim_in = in_features
        self.dim_out = out_features
        self.omega_0 = omega_0

    def forward(self, x, out_hnet):
        cpt = 0
        b_size = out_hnet.shape[0]
        # out_hnet : torch.Size([512, 1951])
        for idx, (lp, lnext) in enumerate(zip([self.dim_in] + self.layers, self.layers + [self.dim_out])):
            din = lp
            dout = lnext
            # torch.Size([512, 1951])
            W = out_hnet[:, cpt:cpt + din * dout].reshape(b_size, dout, din)
            cpt += din * dout
            b = out_hnet[:, cpt:cpt + dout]  # torch.Size([512, 1951])
            cpt += dout
            x = self.omega_0 * torch.einsum('bi, bji -> bj', x, W) + b

            # stop at last layer between layer[n-1] and dim_out
            if idx != self.nlayers:
                x = torch.sin(x)  # self.act
            if idx == self.nlayers:
                return x
