from training.trainingsolver_gd import trainingsolver_gd
from training.trainingsolver_hist import trainingsolver_hist
from training.trainingsolver_nd import trainingsolver_nd
from training.trainingsolver_nd_step import trainingsolver_nd_step

def init_training(type, cfg, loss):
    """initialize trainer

    Args:
        cfg (dict): configuration 
        name (string): name of the expe

    Returns:
        trainer(Training): the trainer for the expe
    """
    elif type == 'conditioner':
        return trainingsolver_gd(cfg, loss)
    elif type=='history':
        return trainingsolver_hist(cfg, loss)
    elif type=='nd':
        return trainingsolver_nd(cfg, loss)
    elif type=='nd_step':
        return trainingsolver_nd_step(cfg, loss)
    else:
        NotImplementedError(f"Training not implemented for type {type}")

