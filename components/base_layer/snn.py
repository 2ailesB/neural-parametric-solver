import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

from components.activation.init import init_activation


class LeakySurrogate(nn.Module):
    # Leaky neuron model, overriding the backward pass with a custom function
    def __init__(self, beta, threshold=1.0):
        super(LeakySurrogate, self).__init__()

        # initialize decay rate beta and threshold
        self.beta = beta
        self.threshold = threshold
        self.spike_op = self.SpikeOperator.apply

    # the forward function is called each time we call Leaky
    def forward(self, input_, mem):
        # call the Heaviside function
        spk = self.spike_op((mem-self.threshold))
        # removes spike_op gradient from reset
        reset = (spk * self.threshold).detach()
        mem = self.beta * mem + input_ - reset  # Eq (1)
        return spk, mem

    # Forward pass: Heaviside function
    # Backward pass: Override Dirac Delta with the Spike itself
    @staticmethod
    class SpikeOperator(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
            spk = (mem > 0).float()  # Heaviside on the forward pass: Eq(2)
            # store the spike for use in the backward pass
            ctx.save_for_backward(spk)
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            (spk,) = ctx.saved_tensors  # retrieve the spike
            grad = grad_output * spk  # scale the gradient by the spike: 1/0
            return grad


class LIFs(nn.Module):
    def __init__(self, in_features, out_features, layers=None, activation='gelu', beta=0.9, dropout_rate=0.0, **kwargs):
        """
        LIFs module
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(LIFs, self).__init__()

        self.activation = init_activation(activation)
        self.layers = layers if layers is not None else []
        self.model = nn.ModuleList([nn.Linear(in_features, self.layers[0])] + [
            nn.Sequential(snn.Leaky(beta=beta), nn.Linear(
                lp, lnext), self.activation)
            for lp, lnext in zip(self.layers, self.layers[1:] + [out_features])
        ])

    def forward(self, x):
        for idx, layer in enumerate(self.model):
            x = layer(x)
        return x



