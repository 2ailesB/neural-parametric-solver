import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from init.init_dataset import init_dataset

name = 'advection'
betas = [0.2, 0.4, 0.7, 1.0, 2.0, 4.0] 
for beta in betas:
    cfg = DictConfig({'name': name, 'ntrain': 10, 'ntest': 10, 'normalization': 0, 'beta': beta, 'nu': 0.5, 'rho': 10.0, 'eta':1e-8, 'zeta':1e-8, 'bc':'trans'})

    dataset, _, _ = init_dataset(name, cfg)
    (params, x, u) = dataset[0]
    print("u.shape : ", u.shape)
    if u.shape[-1]==1:
        plt.plot(u.squeeze(-1))
        # plt.legend(dataset.gridt.numpy())
        plt.savefig(f'xp/vis/ds/vis_{name}_beta{beta}.png')
    else:
        images, axs = plt.subplots(1, u.shape[-1], figsize = (14, 7))
        for channel in range(u.shape[-1]):
            axs[channel].imshow(u[..., channel])
        plt.savefig(f'xp/vis/ds/vis_{name}_beta{beta}.png')