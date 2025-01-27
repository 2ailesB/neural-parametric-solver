from models.iterative import Iterative
from models.conditioner import Preconditioner, Preconditioner_Cholesky, Preconditioner_PINNs
from models.GD_NN import GD_NN, GD_NN_step
from models.RK4_NN import RK4_NNs, RK4_NNs2
from models.Adam_NN import Adam_NNs2


models = {
    "iterative": Iterative,
    "preconditioner": Preconditioner,
    "preconditioner_cholesky": Preconditioner_Cholesky,
    "preconditioner_pinns": Preconditioner_PINNs,
    "gd_nn": GD_NN,
    "rk4": RK4_NNs,
    "rk4_nn": RK4_NNs2,
    "adam_nn": Adam_NNs2,
    "gd_nn_step": GD_NN_step
}

def init_model(name, cfg, pde):
    try:
        return models[name](cfg, pde)
    except KeyError : 
        raise NotImplementedError(f'Solver {name} not implemented.')
    