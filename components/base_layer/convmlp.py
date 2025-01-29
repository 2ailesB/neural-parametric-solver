import torch
import torch.nn as nn
import numpy as np

from components.activation.init import init_activation
from components.base_layer.conv import Conv1ds
from components.base_layer.mlp import MLP

class Conv1dsMLP(nn.Module):
    """ Conv1d followed by MLP """
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.cfg = cfg
        layers = [cfg.units]*(cfg.nlayers + 1)

        self.convnet = Conv1ds(cfg.in_channels, cfg.out_channels, cfg.kernels, cfg.activation, cfg.strides, cfg.paddings, cfg.dilatations)
        self.classifier = MLP(11, cfg.output_dim, layers, cfg.activation) # TODO input en dur

    def forward(self, x):
        o = self.convnet(x)
        o = self.classifier(o.squeeze(-1))

        return o.unsqueeze(-1)

