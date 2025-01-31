import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import time
from pathlib import Path
from torch.utils.data import DataLoader
# from codecarbon import EmissionsTracker
import torch


from init.init_dataset import init_dataset
from components.base_layer.init import init_layers
from init.init_training import init_training
from init.init_model import init_model
from utils.device import set_seed


@hydra.main(config_path="config/", config_name="base.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.autograd.set_detect_anomaly(True)

    ckpt=0
    if cfg.ckpt:
        ckpt = f'{cfg.path2project}logs/{cfg.ckpt}'
        new_cfg = torch.load(ckpt)['cfg']
        # update new_cfg
        new_cfg.exp.nepoch=cfg.exp.nepoch
        cfg = new_cfg 

    print(HydraConfig.get().overrides.task)

    tic = time.time()

    print(f"Training {cfg.exp.name} with model {cfg.model.name} and arch {cfg.model.nn.name} to fit {cfg.exp.approx.name} on {cfg.data.name}")
    set_seed(cfg.seed)

    dtrain, dtest, pde_cfg = init_dataset(cfg.data.name, cfg.data)
    cfg.exp.approx.dim = dtrain.dim
    cfg.exp.approx.channels = dtrain.channels

    dataloader_train = DataLoader(
        dtrain, batch_size=cfg.exp.batch_size, shuffle=cfg.exp.shuffle, num_workers=1, pin_memory=True)
    dataloader_test = DataLoader(
        dtest, batch_size=cfg.exp.batch_size, shuffle=cfg.exp.shuffle, num_workers=1, pin_memory=True)


    exp = init_training(cfg.exp.name, cfg, dtrain.loss_pinns)
    exp_name = exp.run_name
    cfg.model.N = exp.u.get_theta_size()
    cfg.model.nn.output_dim = exp.inner_optimizer.get_update_size() * cfg.model.N
    cfg.model.L=cfg.exp.L

    model = init_model(cfg.model.name, cfg.model, pde_cfg)
    print("Number of parameters: ", model.count_parameters())

    save_path = f"{cfg.path2project}logs/{cfg.data.name}/{cfg.exp.name}/{exp_name}"
    print("save_path : ", save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    exp.set_save_path(save_path)  
    
    exp.fit(model, dataloader_train, dataloader_test, ckpt)

    tac = time.time()

    print("Training duration : ", tac - tic)


if __name__=="__main__":

    main()