from optimizer.GD import GD
from optimizer.Adam import Adam_nns, Adam
from optimizer.AdaBelief import AdaBelief
from optimizer.RK4_nn import RK4_nn, RK4_nns
from optimizer.Newton import Newton
from optimizer.NoOpt import NoOpt

optimizer = {'GD': GD,
            'Adam': Adam,
            'Adam_nns': Adam_nns, 
            'Adabelief': AdaBelief,
            'RK4': RK4_nn,
            'RK4_nns': RK4_nns,
            'Newton': Newton,
            'NoOpt': NoOpt}

def init_optimizer(name, cfg):
    try:
        return optimizer[name](cfg)
    except KeyError:
        raise NotImplementedError(f'Optimizer {name} not implemented')
    
    