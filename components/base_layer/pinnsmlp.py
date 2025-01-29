import torch
import torch.nn as nn

from componnents.activation.init import init_activation

""" Implementation of the modified MLP Layer from 
UNDERSTANDING AND MITIGATING GRADIENT PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS
Sifan Wang, Yujun Teng, Paris Perdikaris
https://arxiv.org/pdf/2001.04536 
"""

class PINNsMLPs(nn.Module):
    def __init__(self, in_features, out_features, layers=None, activation='gelu', dropout_rate=0.0, **kwargs):
        """
        Modified MLP layers
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(PINNsMLPs, self).__init__()
        self.layers = layers if layers is not None else []
        self.U = nn.Linear(in_features, layers[0])
        self.V = nn.Linear(in_features, layers[0])
        self.rec = nn.ModuleList([
            nn.Sequential(nn.Linear(lp, lnext), nn.Dropout(dropout_rate))
            for lp, lnext in zip([in_features] + self.layers, self.layers + [out_features])
        ])

        self.act = init_activation(activation)

    def forward(self, x):
        U = self.act(self.U(x))
        V = self.act(self.V(x))
        h = self.act(self.rec[0](x))

        for idx, layer in enumerate(self.rec[1:], 2):
            z = layer(h)
            if idx != len(self.rec):
                z = self.act(z)
                h = (1 - z) * U + z * V
            if idx >= len(self.rec):
                return z

        return z
