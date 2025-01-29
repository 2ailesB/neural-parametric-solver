import torch
import torch.nn as nn
import numpy as np

from components.activation.init import init_activation

class Conv1ds(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernels=None, activation='gelu', strides=None, paddings=None, dilatations=None, **kwargs):
        """
        Stack of 1D convolutions
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernels: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(Conv1ds, self).__init__()
        self.nlayers = len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernels = kernels 
        assert self.nlayers == len(out_channels) == len(kernels),  f'nlayers muse match in kernels, in_channels, out_channels, but got {len(in_channels)}, {len(out_channels)}, {len(kernels)}'

        self.strides = strides if strides is not None else self.nlayers * [1]
        self.paddings = paddings if paddings is not None else self.nlayers * [0]
        self.dilatations = dilatations if dilatations is not None else self.nlayers * [1]
        
        self.model = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=p, dilation=d), nn.BatchNorm1d(o))
            for (i, o, k, s, p, d) in zip(self.in_channels, self.out_channels, self.kernels, self.strides, self.paddings, self.dilatations)
        ])
        self.act = init_activation(activation) 

    def forward(self, x):
        for idx, layer in enumerate(self.model):
            x = layer(x)

            if idx != len(self.model) - 1:
                x = self.act(x)
        return x.squeeze(1).unsqueeze(-1) # .reshape(bsize, self.out_channels qq chose) # B, out channels, L => B, L, channels


class Conv2ds(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, activation='gelu', frame_shape=None, dropout_rate=0.0, **kwargs):
        """
        Stack of 2D convolutions
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernels: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(Conv2ds, self).__init__()
        self.kernels = kernels if kernels is not None else []
        self.in_channels = in_channels if in_channels is not None else []
        self.out_channels = out_channels if out_channels is not None else []
        self.model = nn.ModuleList([
            nn.Sequential(nn.Conv2d(i, o, k))
            for (i, o, k) in zip(self.in_channels, self.out_channels, self.kernels)
        ])
        self.act = init_activation(activation)
        self.frame_shape = frame_shape if frame_shape is not None else (0, 0)

    def forward(self, x):
        if self.frame_shape[0] == 0:
            self.frame_shape = (np.sqrt(
                x.shape[1] / self.in_channels[0]), np.sqrt(x.shape[1] / self.in_channels[0]))
        x = x.reshape(x.shape[0], self.in_channels[0], int(
            self.frame_shape[0]), int(self.frame_shape[1]))

        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx != len(self.model) - 1:
                x = self.act(x)
        return x
