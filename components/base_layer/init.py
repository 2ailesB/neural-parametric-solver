import torch

from components.base_layer.identity import IdLayer
from components.base_layer.mlp import MLP
from components.base_layer.conv import Conv1ds, Conv2ds
from components.base_layer.pinnsmlp import PINNsMLPs
from components.base_layer.siren import SIRENs
from components.base_layer.mfn import FourierNet, GaborNet
from components.base_layer.resnet import ResNet
from components.base_layer.convmlp import Conv1dsMLP
from components.base_layer.fno import FNO

def init_layers(name, cfg):
    """ 
    Initialize layers
    """
    layers = [cfg.units]*(cfg.nlayers + 1)
    cfg.layers = layers

    if name == 'mlp':
        return MLP(cfg.input_dim, cfg.output_dim, layers, cfg.activation)
    elif name == 'siren':
        return SIRENs(cfg.input_dim,  cfg.output_dim, layers, omega_0=cfg.omega_0, weight_init_factor=cfg.weight_init_factor)
    elif name == 'resnet':
        if cfg.units != cfg.input_dim:
            print(Warning(f"Warning : The layer size must be the same as input dim for resnet. Set size from {cfg.units} to {cfg.input_dim}."))
            cfg.units = cfg.input_dim
            layers = [cfg.units]*(cfg.nlayers + 1)
            cfg.layers = layers
        return ResNet(cfg.input_dim, cfg.output_dim, layers, activation=cfg.activation)
    elif name == 'pinnsmlp':
        return PINNsMLPs(cfg.input_dim, cfg.output_dim, layers, cfg.activation)
    elif name == 'mfn_fourier':
        return FourierNet(cfg.input_dim, cfg.units, cfg.output_dim, cfg.nlayers)
    elif name == 'id':
        return IdLayer()
    elif name == 'mfn_gabor':
        return GaborNet(cfg.input_dim, cfg.units, cfg.output_dim, cfg.nlayers)
    elif name == 'conv1':
        return Conv1ds(cfg.in_channels, cfg.out_channels, cfg.kernels, cfg.activation, cfg.strides, cfg.paddings, cfg.dilatations)
    elif name == 'conv2':
        return Conv2ds(cfg.in_channels, cfg.out_channels, cfg.kernel) 
    elif name == 'conv1mlp':
        return Conv1dsMLP(cfg)
    elif name == 'fno':
        return FNO(cfg)
    else:
        raise NotImplementedError(
            f'layer {name} not implemented')

