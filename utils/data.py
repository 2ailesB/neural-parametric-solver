import torch
import einops

""" Utils function related to data: normalization, subsampling, etc. """

def get_normalization_01(datas, axis=0):

    if axis is not None:
        dmin = datas.amin(axis)
        dmax = datas.amax(axis)
    else:
        dmin = datas.amin()
        dmax = datas.amax()

    return dmin, dmax-dmin

def get_normalization_N01(datas, axis=0):
    m = datas.mean(axis)
    s = datas.std(axis)

    return m, s

def apply_normalization(datas, m, s):
    return (datas - m) / s

def remove_normalization(datas, m, s):
    return s * datas + m    

def dynamics_different_subsample(u, grid, draw_ratio):
    """

    from: https://github.com/LouisSerrano/coral/blob/main/coral/utils/data/load_data.py

    /!\ written only in 1D static /!\ 

    Performs subsampling for univariate time series
    Args:
        u (torch.Tensor): univariate time series (batch_size, num_points, num_channels)
        grid (torch.Tensor): timesteps coordinates (num_points, input_dim)
        draw_ratio (float): draw ratio
    Returns:
        small_data: subsampled data
        small_grid: subsampled grid
        permutations: draw indexs
    """
    # u = einops.rearrange(u, "b ... c -> b (...) c")
    # grid = einops.rearrange(grid, "b ... c -> b (...) c")

    N = u.shape[0]
    C = u.shape[-1]

    partial_grid_size = int(draw_ratio * grid.shape[0])

    # Create draw indexes
    permutations = [
        torch.randperm(grid.shape[0]-1)[:partial_grid_size-1].unsqueeze(0) + 1 # remove 0 to keep it for BCs
        for ii in range(N)
    ]

    for ii in range(N):
        permutations[ii], _ = torch.sort(permutations[ii])

    permutations = torch.cat(permutations, axis=0) # B, X
    permutations = torch.cat([torch.zeros((N, 1), dtype=torch.int64), permutations], axis=1) # add 0 for BCs
    small_u = torch.gather(u, 1, permutations.unsqueeze(-1).repeat( 1, 1, C))
    small_grid = torch.gather(grid.unsqueeze(0).repeat(N, 1, 1), 1, permutations.unsqueeze(-1))

    return small_u, small_grid, permutations

def dynamics_different_subsample_diffgrid(u, grid, min_ratio, device='cpu'):
    """

    from: https://github.com/LouisSerrano/coral/blob/main/coral/utils/data/load_data.py

    /!\ written only in 1D static /!\ 

    Performs subsampling for univariate time series
    Args:
        u (torch.Tensor): univariate time series (batch_size, num_points, num_channels)
        grid (torch.Tensor): timesteps coordinates (num_points, input_dim)
        draw_ratio (float): draw ratio
    Returns:
        small_data: subsampled data
        small_grid: subsampled grid
        permutations: draw indexs
    """
    # u = einops.rearrange(u, "b ... c -> b (...) c")
    # grid = einops.rearrange(grid, "b ... c -> b (...) c")

    N = u.shape[0]
    C = u.shape[-1]
    draw_ratio = torch.rand(N, device=device) * (1 - min_ratio) + min_ratio # [0, 1] => [min_ratio, 1]

    small_us = []
    small_grids = []
    permutations = []
    for ii in range(N):
        # Create draw indexes
        permutation = torch.randperm(grid.shape[0]-1, device=device)[:int(draw_ratio[ii] * grid.shape[0])-1] + 1 # remove 0 to keep it for BCs
            
        permutation = torch.cat([torch.zeros(1, dtype=torch.int64), permutation], axis=0) # add 0 for BCs
        permutations += [permutation]
        small_us += [torch.gather(u[ii],0, permutation.unsqueeze(-1).repeat(1, C))]
        small_grids += [torch.gather(grid,0, permutation.unsqueeze(-1))]
    return small_us, small_grids, permutations
