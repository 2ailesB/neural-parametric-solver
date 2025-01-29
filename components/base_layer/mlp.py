import torch
import torch.nn as nn
import numpy as np
import einops
# from complexPyTorch.complexLayers import apply_complex

from components.activation.init import init_activation

class MLP(nn.Module):
    def __init__(self, in_features, out_features, layers=None, activation='gelu', dropout_rate=0.0, **kwargs):
        """
        Usual MLP module
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(MLP, self).__init__()
        self.layers = layers if layers is not None else []
        self.model = nn.ModuleList([
            nn.Sequential(nn.Linear(lp, lnext), nn.Dropout(dropout_rate))
            for lp, lnext in zip([in_features] + self.layers, self.layers + [out_features])
        ])

        self.act = init_activation(activation)

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx != len(self.model) - 1:
                x = self.act(x)
        return x


class MLP_parametrized(nn.Module):
    def __init__(self, in_features, out_features, layers=None, activation='gelu', dropout_rate=0.0, **kwargs):
        """
        Usual MLP module with parameterized weights ie the weights are given as input in the forward 
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(MLP_parametrized, self).__init__()
        self.layers = layers if layers is not None else []
        self.nlayers = len(layers)
        self.dim_in = in_features
        self.dim_out = out_features

        self.act = init_activation(activation)

    def forward(self, x, out_hnet):
        cpt = 0
        b_size = out_hnet.shape[0]
        for idx, (lp, lnext) in enumerate(zip([self.dim_in] + self.layers, self.layers + [self.dim_out])):
            din = lp
            dout = lnext
            W = out_hnet[:, cpt:cpt + din * dout].reshape(b_size, dout, din)
            cpt += din * dout
            b = out_hnet[:, cpt:cpt + dout].reshape(b_size, dout)  # torch.Size([512, 1951])
            cpt += dout
            x = torch.einsum('bi, bji -> bj', x, W) + b
            # stop at last layer between layer[n-1] and dim_out
            if idx != self.nlayers:
                x = self.act(x)
            if idx == self.nlayers:
                return x

    def batched_forward(self, x, out_hnet):
        cpt = 0
        b_size = out_hnet.shape[0]
        sres = x.shape[1]
        # B, X / B, P

        # out_hnet : torch.Size([512, 1951])
        for idx, (lp, lnext) in enumerate(zip([self.dim_in] + self.layers, self.layers + [self.dim_out])):
            din = lp
            dout = lnext
            # torch.Size([512, 1951])
            W = einops.repeat(out_hnet[:, cpt:cpt + din * dout], 'B P -> B X P', X=sres).reshape(b_size, sres, dout, din)
            cpt += din * dout
            b = einops.repeat(out_hnet[:, cpt:cpt + dout], 'B P -> B X P', X=sres).reshape(b_size, sres, dout)  # torch.Size([512, 1951])
            cpt += dout
            x = torch.einsum('bxi, bxji -> bxj', x, W) + b
            # stop at last layer between layer[n-1] and dim_out
            if idx != self.nlayers:
                x = self.act(x)
            if idx == self.nlayers:
                return x
            
    def get_weights_biases(self, out_hnet):
        b_size = out_hnet.shape[0]
        Ws = []
        bs = []
        cpt = 0

        for idx, (lp, lnext) in enumerate(zip([self.dim_in] + self.layers, self.layers + [self.dim_out])):
            din = lp
            dout = lnext
            # torch.Size([512, 1951])
            Ws += [out_hnet[:, cpt:cpt + din * dout].reshape(b_size, dout, din)]
            cpt += din * dout
            bs += [out_hnet[:, cpt:cpt + dout].reshape(b_size, dout)]
            cpt += dout

        return Ws, bs
    
    def count_parameters(self):
        cp = 0
        for lp, lnext in zip([self.dim_in] + self.layers, self.layers + [self.dim_out]):
            cp += lp * lnext + lnext
        return cp


class MLP4SIREN(nn.Module):
    def __init__(self, in_features, out_features, cfg_shape_net, layers=None, activation='gelu', dropout_rate=0.0, **kwargs):
        """
        Hyper network for parametrized SIREN (required specifici initialization, see https://github.com/pswpswpsw/nif)
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(MLP4SIREN, self).__init__()
        self.layers = layers if layers is not None else []
        self.in_features = in_features
        self.out_features = out_features
        self.model = nn.Linear(in_features, out_features)

        self.cfg_shape_net = cfg_shape_net
        self.somega_0 = self.cfg_shape_net['omega_0']
        self.sweight_init_factor = self.cfg_shape_net['weight_init_factor']
        self.swidth = self.cfg_shape_net['units']
        self.snlayers = self.cfg_shape_net['nlayers']
        self.sinput_dim = self.cfg_shape_net['input_dim']
        self.soutput_dim = self.cfg_shape_net['output_dim']
        self.init_weight_hnetsiren()

        self.act = torch.nn.ReLU() if activation == 'relu' else torch.nn.Tanh() if activation == 'tanh' else torch.nn.Sigmoid(
        ) if activation == 'sigmoid' else torch.nn.GELU() if activation == 'gelu' else torch.nn.SiLU() if activation == 'swish' else ValueError

    def forward(self, x):
        x = self.model(x)
        return x

    def init_weight_hnetsiren(self):
        if isinstance(self.model, nn.Linear):
            weight_dist = torch.distributions.uniform.Uniform(-np.sqrt(
                6/self.sinput_dim)*self.sweight_init_factor, np.sqrt(6/self.sinput_dim)*self.sweight_init_factor)
            self.model.weight.data = nn.Parameter(
                weight_dist.sample(self.model.weight.data.shape))

            num_weight_first = self.sinput_dim*self.swidth
            num_weight_hidden = self.snlayers*(self.swidth**2)
            num_weight_last = self.soutput_dim*self.swidth

            scale_matrix = torch.ones((self.out_features))
            # 1st layer weights
            scale_matrix[:num_weight_first] /= self.sinput_dim
            scale_matrix[num_weight_first:
                         num_weight_first + num_weight_hidden] *= np.sqrt(6.0/self.swidth)/self.somega_0  # hidden layer weights
            scale_matrix[num_weight_first + num_weight_hidden:
                         num_weight_first + num_weight_hidden + num_weight_last] *= np.sqrt(6.0/(self.swidth + self.swidth))  # last layer weights, since it is linear layer and no scaling,
            # we choose GlorotUniform
            scale_matrix[num_weight_first + num_weight_hidden +
                         num_weight_last:] /= self.swidth  # all biases

            bias_dist = torch.distributions.uniform.Uniform(
                -scale_matrix, scale_matrix)
            self.model.bias.data = nn.Parameter(bias_dist.sample())


