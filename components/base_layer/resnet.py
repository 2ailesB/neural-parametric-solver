import torch
import torch.nn as nn
import numpy as np

from components.activation.init import init_activation

class ResNet(nn.Module):
    def __init__(self, in_features, out_features, layers=None, activation='gelu', dropout_rate=0.0, **kwargs):
        """
        Usual ResNet module
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(ResNet, self).__init__()
        self.layers = layers if layers is not None else []
        self.model = nn.ModuleList([
            nn.Sequential(nn.Linear(lp, lnext), nn.Dropout(dropout_rate))
            for lp, lnext in zip([in_features] + self.layers, self.layers + [out_features])
        ])

        self.act = init_activation(activation)

    def forward(self, x):
        for idx, layer in enumerate(self.model):
            x_prec = x
            x = layer(x_prec)

            if idx != len(self.model) - 1:
                x = x_prec + x
                x = self.act(x)

        return x
