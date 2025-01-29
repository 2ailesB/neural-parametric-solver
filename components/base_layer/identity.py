import torch.nn as nn
import torch


class IdLayer(nn.Module):
    def __init__(self, **kwargs):
        """
        Identity layer
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(IdLayer, self).__init__()

    def forward(self, x):

        return x
