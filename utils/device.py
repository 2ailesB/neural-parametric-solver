import torch
import random
import numpy as np
import subprocess as sp
import os

""" Device utilities """

def get_device():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        device = torch.device(f'cuda:{gpu_id}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')
    # print(f'Running on {device}')

    return device

def set_seed(seed=33):
    """Set all seeds for the experiments.

    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def to4list(list, device):
    return [it.to(device).float() for it in list]