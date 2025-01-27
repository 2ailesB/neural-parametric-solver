import wandb
from omegaconf import OmegaConf
from pathlib import Path

def init_logger(cfg):
    """initialize logger
    Args:
        cfg (omegaconf.dictconfig.DictConfig): configuration
    Returns:
        logger (wandb): the logger
        exp_name (string): the name of the experiment
        save_path (string): the path to save the experiment
    """
    
    logger = wandb.init(
        project=cfg.exp.logger.project, dir=cfg.exp.save_path, entity=cfg.exp.logger.entity,
        tags=[cfg.solver.name, cfg.solver.model.name, cfg.data.name])
    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    wandb.define_metric("batch", hidden=True)
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("lr", hidden=True)

    wandb.define_metric(
        "train_loss", step_metric="epoch", summary="min")
    wandb.define_metric(
        "test_loss", step_metric="epoch", summary="min")
    wandb.define_metric("train_batch_loss", step_metric="batch")
    run_name = wandb.run.name

    print("run_name : ", run_name)

    exp_name = run_name

    # save_path = f"{cfg.path2project}logs/{cfg.data.name}/{cfg.solver.name}/{exp_name}"
    save_path = f"{cfg.exp.logger.save_path}/{cfg.data.name}/{cfg.solver.name}/{exp_name}"
    print("save_path : ", save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    return logger, exp_name, save_path
