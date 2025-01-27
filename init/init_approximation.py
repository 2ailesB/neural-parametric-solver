from components.approximation.MultiChannel import MultiChannel
from components.approximation.init_nd import init_ndbasis


def init_approx(name, cfg):
    if cfg.channels==1:
       return init_ndbasis(name, cfg)
    else : 
        return MultiChannel(cfg)