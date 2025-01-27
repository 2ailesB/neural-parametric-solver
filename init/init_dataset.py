from ngd_datasets.d1.helmholtz import Helmholtz
from ngd_datasets.d1.poisson import Poisson
from ngd_datasets.d1.forcingmspoisson1d import ForcingMSPoisson1d
from ngd_datasets.d1p1.Advection import Advection
from ngd_datasets.d1p1.AdvectionS import AdvectionS
from ngd_datasets.d2.Darcy import Darcy
from ngd_datasets.d1p1.NlReactionDiffusion import NlReactionDiffusion
from ngd_datasets.d1p1.NlReactionDiffusionICs import NlReactionDiffusionICs
from ngd_datasets.d2p1.Heat2d import Heat2d

datas = {
    "helmholtz": Helmholtz, 
    "poisson": Poisson,
    "helmholtz-hf": Helmholtz,
    "forcingmspoisson": ForcingMSPoisson1d,
    "advection": Advection,
    "advections": AdvectionS,
    "darcy": Darcy,
    "1dnlrd": NlReactionDiffusion,
    "1dnlrdics": NlReactionDiffusionICs,
    "heat2d": Heat2d,
}

def init_dataset(name, cfg):
    """initialize dataset   
    Args:
        name (string): name of the dataset
        cfg (omegaconf.dictconfig.DictConfig): configuration
    Returns:
        dtrain (Dataset): the training dataset
        dtest (Dataset): the test dataset
        pde_cfg (tuple): the pde configuration as :
        (PDE Physical losses, sizes (frame, params, forcings, ic, bc), channels (frame, params, forcings, ic, bc), PDE dim, PDE channels)
    """
    try:
        dtrain = datas[name](cfg, cfg.ntrain, 'train')
        dtest = datas[name](cfg, cfg.ntest, 'test')
        pde_cfg = (dtrain.pde, dtrain.get_sizes(), dtrain.get_channels(), dtrain.dim, dtest.channels)

    except KeyError:
        raise NotImplementedError(f'Dataset {name} does not exist')
    
    return dtrain, dtest, pde_cfg