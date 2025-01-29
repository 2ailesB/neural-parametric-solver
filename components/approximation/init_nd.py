from components.approximation.init import init_1dbasis

from components.approximation.Basis1d import Basis1d
from components.approximation.Basis2d import Basis2d
from components.approximation.Basis3d import Basis3d
# from components.approximation.xp_Basis4d import Basis4d
from components.approximation.nlb import nl_basis

def init_ndbasis(name, cfg):
    if cfg.nl:
        return nl_basis(cfg)
    if cfg.dim==1:
        return Basis1d(cfg)
    elif cfg.dim==2:
        return Basis2d(cfg)
    elif cfg.dim==3:
        return Basis3d(cfg)
    else: raise NotImplementedError(f'{cfg.dim}d and {cfg.channels} basis not implemented')