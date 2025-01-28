import datetime
import os
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import einops

from utils.device import get_device
from optimizer.nys_newton_cg import NysNewtonCG as nncg_d
from optimizer.opts.adam_lbfgs_gd import Adam_LBFGS_GD
from optimizer.opts.adam_lbfgs_nncg import Adam_LBFGS_NNCG
from optimizer.opts.adam_lbfgs import Adam_LBFGS
from optimizer.opts.alr_mag import ALRMAG
from optimizer.opts.nys_newton_cg import NysNewtonCG
from optimizer.opts.gd import GD
from optimizer.opts.polyak_gd import PolyakGD
from optimizer.opts.polyak_lbfgs import PolyakLBFGS
from optimizer.opts.sketchygn import SketchyGN
from optimizer.opts.sketchysgd import SketchySGD
from optimizer.opts.MultiAdam import MultiAdam


class Experiment():
    def __init__(self, cfg):
        """Training class : need to be re instanciated with a _train_epoch() and _validate() function

        Args:
            cfg (dict): config dictionary for training
            tag (str, optional): name of expe. Defaults to ''.
        """
        # save cfg
        self.all_cfg = cfg
        self.pde_name = cfg.data.name
        self.cfg = cfg.exp

        # training cfg
        self.n_epochs = self.cfg.nepoch
        self.lr = self.cfg.lr
        self.lr_2 = self.cfg.lr_2
        self.lr_3 = self.cfg.lr_3
        self.cfg_opt = self.cfg

        # dataloader cfg
        self.batch_size = self.cfg.batch_size
        self.shuffle = self.cfg.shuffle

        # optimizer
        self.opt_type = self.cfg.opt
        self.scheduler_cfg = self.cfg.scheduler 
        self.scheduler_type = self.cfg.scheduler.name
        self.opt = None
        self.scheduler = None
        self.criterion = self.init_criterion()
        # device
        self.device = get_device()
        # print(f"Running on  {self.device}")

        # logging info
        self.ckpt_save_path = self.cfg.save_path
        self.eval_freq = self.cfg.eval_freq 
        self.save_every_bool=self.cfg.save_every

        self.wandb = self.cfg.wandb
        self.watch_grad = self.cfg.watch_grad
        self.tags = self.cfg.tags
        if self.wandb:
            self.logger = wandb.init(
                project=f"neural-solver", dir=self.ckpt_save_path, entity="spatiotemp-isir",
                tags=[cfg.exp.approx.name, cfg.model.name, cfg.model.nn.name, cfg.data.name, f'{cfg.exp.approx.dim}d'] + self.tags)
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
            self.run_name = wandb.run.name
        else:
            self.run_name='debug'
        
        try :
            self.nvisu = self.cfg.nvisu
        except: 
            self.nvisu=1

        self.verbose_freq = self.cfg.verbose_freq

        # useful stuff that can be needed for during fit
        self.state = {}
        self.best_criterion = {'train_loss': 10 **
                               10, 'val_loss': 10**10, 'mse': 10*10}
        self.best_model = None
        self.best_epoch = 0
        self.start_time = None
        self.n = None


    def _train_epoch(self, dataloader, epoch):
        """train one epoch

        Args:
            dataloader (PyTorch dataloader): dataloader for training
            epoch (int): number of epoch
        """
        pass

    def _validate(self, dataloader, epoch):
        """validate one epoch

        Args:
            dataloader (PyTorch dataloader): dataloader for training
        """
        pass

    def _visual(self, dataloader):
        """validate one epoch

        Args:
            dataloader (PyTorch dataloader): dataloader for training
        """
        pass
    
    # @track_emissions(project_name='neural-solver')
    def fit(self, model, dataloader, validation_data=None, ckpt=None):
        """fit the given model on the data in dataloader

        Args:
            model (nn.Module trainable): the model to be fit
            dataloader (torch.Dataloader): the dataloader containing data for fitting
            validation_data (torch Dataloader, optional): the dataloader containing data for validation. Defaults to None.
            ckpt (_type_, optional): When to dave the model. Defaults to None.
            betas (tuple, optional): betas for adam optimizer. Defaults to (0.0, 0.99).

        Raises:
            ValueError: unknown optimizer
        """
        # tracker = EmissionsTracker(log_level="error")
        # tracker.start()
        self.model = model.to(self.device)
        self.model.device = self.device
        
        assert self.model is not None, 'Training does not have a model, assign the model to the self.model attribute'

        if self.wandb and self.watch_grad:
            wandb.watch(self.model, log='all', log_freq=100, criterion=torch.nn.MSELoss, log_graph=True)

        # init optim
        if self.opt_type != 'adam_lbfgs' and self.opt_type != 'adam_lbfgs_nncg':
            self.opt = self.init_optim(self.opt_type, self.model, self.lr, self.cfg_opt)
        elif self.opt_type == 'adam_lbfgs':
            self.opt = self.init_optim('adam', self.model, self.lr, self.cfg_opt)
            self.opt_2 = self.init_optim('lbfgs', self.model, self.lr_2, self.cfg_opt)
        elif self.opt_type=='adam_lbfgs_nncg':
            self.opt = self.init_optim('adam', self.model, self.lr, self.cfg_opt)
            self.opt_2 = self.init_optim('lbfgs', self.model, self.lr_2, self.cfg_opt)
            self.opt_3 = self.init_optim('nncg', self.model, self.lr_3, self.cfg_opt)
        else:
            self.opt = self.init_optim(self.opt_type, self.model, self.lr, self.cfg_opt)

        self.scheduler = self.init_scheduler(self.cfg.scheduler, self.opt)

        start_time = datetime.datetime.now()
        self.start_time = start_time

        start_epoch = 0

        if ckpt:
            self.load(ckpt)
            # for g in self.opt.param_groups:
            #     print("g.lr : ", g.lr)
            #     g.lr = state['param_groups']['lr']

        t_loss=0
        v_loss=0

        assert self.check_assertions(), 'Custom assertion error'

        for n in range(start_epoch, self.n_epochs):
            self.n = n
            model.train()
            if n % 10 == 0 and dataloader.dataset.scheduled:
                dataloader.dataset.scheduler()
                validation_data.dataset.scheduler()
            # print('trb available mem', torch.cuda.mem_get_info())
            t_loss, t_mse = self._train_epoch(dataloader, n)
            # print('tra available mem', torch.cuda.mem_get_info())
            plt.clf()
            plt.close('all')

            if self.cfg.name=='meta_model_driven':
                mse_loss = t_loss[1]
                t_loss = t_loss[0]

            epoch_result = {'train_loss': t_loss, # torch.sqrt(t_loss),
                            'train_mse': t_mse,
                            'epoch': n, 'lr': self.opt.param_groups[0]['lr']}

            if validation_data is not None and (n % self.eval_freq == 0 or n == self.n_epochs - 1):
                model.eval()
                # with torch.no_grad():
                # print('valb available mem', torch.cuda.mem_get_info())
                v_loss, v_mse = self._validate(validation_data, n)
                # print('vala available mem', torch.cuda.mem_get_info())
                epoch_result.update({'test_loss': v_loss, 'test_mse': v_mse}) # torch.sqrt(v_loss)})
                plt.clf()
                plt.close('all')


            if self.scheduler is not None:
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(t_loss)
                else:
                    # if self.n % 15 == 0:
                    self.scheduler.step()

            if self.wandb:
                self.logger.log(epoch_result)  # , step=n)

            if epoch_result['train_loss'] <= self.best_criterion['train_loss']:
                self.__save_state(epoch_result)
                self.save(self.lr, self.n)
            if self.save_every_bool and self.n%self.save_every_bool==0:
                self.save_every(self.lr, self.n)
            if self.n % self.verbose_freq == 0 or self.n == self.n_epochs - 1:
                print(f'Epoch {self.n} training loss: {t_loss} | Validation loss: {v_loss} | Best epoch {self.best_epoch}') # | mse: {mse_t}
                if self.cfg.name=='meta_model_driven':
                    print(f'mse: {mse_loss}')

        # self.logger.summary.update({'Emissions': emission})

        wandb.finish()

        if self.watch_grad:
            fig = plt.figure()
            plt.imshow(self.model.nn.model[0][0].weight.detach().cpu().numpy().T, cmap='coolwarm')
            plt.colorbar()
            fig.savefig('xp/vis/heatmapl1.png')

        print(
            f'Training completed in {str(datetime.datetime.now() - start_time).split(".")[0]}')

    # @track_emissions
    def _evaluate(self, model, dataloader, ckpt):
        """fit the given model on the data in dataloader

        Args:
            model (nn.Module trainable): the model to be fit
            dataloader (torch.Dataloader): the dataloader containing data for fitting
            validation_data (torch Dataloader, optional): the dataloader containing data for validation. Defaults to None.
            ckpt (_type_, optional): When to dave the model. Defaults to None.
            betas (tuple, optional): betas for adam optimizer. Defaults to (0.0, 0.99).

        Raises:
            ValueError: unknown optimizer
        """

        self.model = model.to(self.device)
        # wandb.watch(model, log_freq=100, log_graph=True)
        assert self.model is not None, 'Training does not have a model, assign the model to the self.model attribute'

        assert self.nvisu <= len(dataloader), f'Too much visualization asked : {self.nvisu} asked and {len(dataloader)} available (depends on the number of batch).'

        start_time = datetime.datetime.now()
        self.start_time = start_time

        state = torch.load(ckpt)
        # self.model.load_state_dict(state.state_dict)
        self.model.load_state_dict(state['model']) 

        model.eval()
        with torch.no_grad():
            v_loss = self._validate(dataloader, 0)

        duration = datetime.datetime.now() - start_time

        print(
            f'Evaluation completed in {str(duration).split(".")[0]}')

        return v_loss, duration

    # @track_emissions : TODO
    def visualize(self, model, dataloader, ckpt, idx=None, idt=None):
        """fit the given model on the data in dataloader

        Args:
            model (nn.Module trainable): the model to be fit
            dataloader (torch.Dataloader): the dataloader containing data for fitting
            validation_data (torch Dataloader, optional): the dataloader containing data for validation. Defaults to None.
            ckpt (_type_, optional): When to dave the model. Defaults to None.
            betas (tuple, optional): betas for adam optimizer. Defaults to (0.0, 0.99).

        Raises:
            ValueError: unknown optimizer
        """


        self.model = model.to(self.device)
        # wandb.watch(model, log_freq=100, log_graph=True)
        assert self.model is not None, 'Training does not have a model, assign the model to the self.model attribute'

        start_time = datetime.datetime.now()
        self.start_time = start_time

        state = torch.load(ckpt)
        start_epoch = state.epoch
        self.model.load_state_dict(state.state_dict)

        model.eval()
        with torch.no_grad():
            v_loss = self._validate(dataloader, 0, vis=True)

        duration = datetime.datetime.now() - start_time

        print(
            f'Visualization completed in {str(duration).split(".")[0]}')

        return v_loss, duration

    def save(self, lr, n):
        """save the model

        Args:
            lr (float): current learning rate
            n (int): current epoch
        """
        self.state['lr'] = lr
        self.state['epoch'] = n
        self.state['best_epoch'] = self.best_epoch
        self.state['model'] = self.best_model.state_dict()
        self.state['optimizer'] = self.opt.state_dict()
        self.state['loss'] = self.best_criterion
        self.state['cfg'] = self.all_cfg

        torch.save(self.state, os.path.join(
            self.ckpt_save_path, f'{self.run_name}.ckpt'))
    
    def save_every(self, lr, n):
        """save the model

        Args:
            lr (float): current learning rate
            n (int): current epoch
        """
        self.state['lr'] = lr
        self.state['epoch'] = n
        self.state['best_epoch'] = self.best_epoch
        self.state['model'] = self.best_model.state_dict()
        self.state['optimizer'] = self.opt.state_dict()
        self.state['loss'] = self.best_criterion
        self.state['cfg'] = self.all_cfg

        torch.save(self.state, os.path.join(
            self.ckpt_save_path, f'{self.run_name}_{n}.ckpt'))

    def load(self, ckpt_path):
        """load model

        Args:
            ckpt_path (string): path to the model to load
        """
        state = torch.load(ckpt_path)

        if self.model:
            self.model.load_state_dict(state['model']) 
        if self.opt:
            self.opt.load_state_dict(state['optimizer'])
        self.best_criterion = state['loss']
        self.n = state['epoch']
        self.state = state
    
    def __save_state(self, loss):
        """
        Save best model in class

        Args:
            n (_type_): _description_
        """
        self.best_epoch = self.n
        self.best_model = self.model
        self.best_criterion = loss


    def get_grid(self, shape):
        """get grid for spatial coordinates

        Args:
            shape (tuple): shape of the data 
        """
        pass

    def set_save_path(self, path):
        self.ckpt_save_path = path

    def set_model(self, model):
        self.model = model

    def set_dim(self, dim):
        self.dim=dim

    def init_scheduler(self, cfg, optimizer):
        """Initialize scheduler

        Returns:
            function: the function for scheduling
        """
        scheduler_type = cfg.name
        if scheduler_type == 'steplr':
            return torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)
        elif scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75,
                                                                patience=cfg.patience, cooldown=0, eps=1e-08, verbose=False,)
        elif scheduler_type == 'exp':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.995)
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs, 1e-5)
        elif scheduler_type==0:
            return None
        else:
            raise NotImplementedError(f'Scheduler type {scheduler_type} not implemented')

    def init_optim(self, opt_type, model, lr, cfg):
        """
        """

        if opt_type == 'sgd':
            return torch.optim.SGD(
                params=model.parameters(), lr=lr)
        elif opt_type == 'adam':
            return torch.optim.Adam(
                params=model.parameters(), lr=lr)
        elif opt_type == 'lbfgs':
            return torch.optim.LBFGS(params=model.parameters(), lr=lr, history_size=100) #, line_search_fn='strong_wolfe')
        elif opt_type == 'nncg':
            return NysNewtonCG(params=model.parameters(), lr=lr)
        elif opt_type == 'adam_lbfgs':
            cfg_adam={'lr': cfg.lr}
            cfg_lbfgs={'lr': cfg.lr_2}
            return Adam_LBFGS(params=model.parameters(), switch_epochs=cfg.switch_epoch, adam_params=cfg_adam, lbfgs_params=cfg_lbfgs)
        elif opt_type == 'adam_lbfgs_nncg':
            cfg_adam={'lr': cfg.lr}
            cfg_lbfgs={'lr': cfg.lr_2}
            cfg_nncg={'lr': cfg.lr_3}
            return Adam_LBFGS_NNCG(params=model.parameters(), switch_epoch1=cfg.switch_epoch, switch_epoch2=cfg.switch_epoch2, precond_update_freq=cfg.precond_update_freq, adam_params=cfg_adam, lbfgs_params=cfg_lbfgs, nncg_params=cfg_nncg)
        else:
            raise ValueError(f'Unknown optimizer {opt_type}')
        
    def check_assertions(self):
        return True

    def init_criterion(self):
        if self.cfg.criterion.name == 'mse':
            return nn.MSELoss(reduction='mean') 
        elif self.cfg.criterion.name == 'bce' : 
            return nn.BCELoss(reduction='mean')  
        elif self.cfg.criterion.name == 'l1':
            return nn.SmoothL1Loss(reduction='mean', beta=self.cfg.criterion.beta) 
        else :
            raise NotImplementedError(f'Criterion {self.cfg.criterion.name} not Implemented')

    def process_coord(self, x):
        """
        x is B, X, (Y, Z, T)
        """
        bsize, coordsize = x.shape[0], x.shape[1:]
        if len(x.shape)==2:
            dim = 1
            x[:, :-1]
            x_in = x[:, 1:]
            x_bc = x[:, [0]]
        else:
            dim = len(x.shape[1:])
            x_ic = x[..., 0]
            x_bc = x[:, 0, :]
            x = einops.rearange()
            # TODO or TOREMOVE ? 

            

            


