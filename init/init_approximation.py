from components.approximation.MultiChannel import MultiChannel
from components.approximation.init_nd import init_ndbasis


def init_approx(name, cfg):
    """initialize approximation

    Args:
        name (string): name of the approximation
        cfg (omegaconf.dictconfig.DictConfig): configuration

    Returns: 
        approximation (Approximation): the approximation
    """
    
    if cfg.channels==1:
       return init_ndbasis(name, cfg)
    else : 
        raise NotImplementedError(f'{cfg.channels} channels basis not implemented')
        return MultiChannel(cfg)