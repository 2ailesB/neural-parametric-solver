import torch
import torch.nn.functional as F
import einops
import matplotlib.pyplot as plt
import time

from training.abstract import Experiment
from utils.plot import plot_losses
from utils.device import to4list

class ModelDriven_training(Experiment):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.evolvsol_tr = [[] for i in range(self.nvisu)]
        self.evolvsol_te = [[] for i in range(self.nvisu)]
        self.keysol_tr = []
        self.keysol_te = []
        self.data_loss = self.criterion

        self.lbd = self.cfg.lbd
        
    def fit(self, model, dataloader, validation_data=None, ckpt=None):
        model.set_equation(self.all_cfg.data.name)
        return super().fit(model, dataloader, validation_data, ckpt)

    def _train_epoch(self, dataloader, epoch):
        epoch_loss = 0
        mse_loss = 0
        idxMax = len(dataloader)

        if self.cfg.switch_epoch is not None:
            switch_epoch = self.cfg.switch_epoch
            if self.cfg.switch_epoch_2 is not None:
                switch_epoch_2= self.cfg.switch_epoch_2
            else:
                switch_epoch_2 = self.cfg.nepoch
            if self.cfg.precond_upd_freq is not None:
                precond_upd_freq=self.cfg.precond_upd_freq
        else:
            switch_epoch = self.cfg.nepoch
            switch_epoch_2 = self.cfg.nepoch

        for idx, (params, x, batch_y, _) in enumerate(dataloader):
            
            # load batch and shapes
            params = to4list(params, self.device) # .float() # params = u_b, g_b, omega, T => B, P, channels
            (x, x_in, x_ic, x_bc) = to4list(x, self.device) #  :-1 to remove boundary effects B, X, T, D
            batch_y = batch_y.to(self.device)
            bsize, frame_size, channels = batch_y.shape[0], batch_y.shape[1:-1], batch_y.shape[-1]
            dim = x.shape[-1]

            if dim > 1:
                x = einops.rearrange(x, 'B ... C -> B (...) C')
                x_in = einops.rearrange(x_in, 'B ... C -> B (...) C')
                x_ic = einops.rearrange(x_ic, 'B ... C -> B (...) C')
                x_bc = einops.rearrange(x_bc, 'B ... C -> B (...) C')
                batch_y = einops.rearrange(batch_y, 'B ... C -> B (...) C')

            batch_x = self.model.get_input(params, x)

            ntraj = bsize = batch_x.size(0)
            # sres = batch_x.shape[1]
            shapes_x = batch_x.shape # B, X, C

            batch_y = batch_y.to(self.device).float() 
            batch_x = batch_x.to(self.device).float().requires_grad_(True) 
            if (self.opt_type != 'lbfgs' and self.opt_type != 'nncg') and (self.opt_type not in ['adam_lbfgs','adam_lbfgs_nncg'] or epoch<= switch_epoch):
                self.opt.zero_grad()
                yhat = self.model(batch_x)  
                loss = self.model.physical_loss(yhat, batch_x, params, self.lbd)
                loss.backward()
                self.opt.step()
            elif self.opt_type == 'lbfgs':
                def closure():
                    self.opt.zero_grad()
                    yhat = self.model(batch_x)  
                    loss = self.model.physical_loss(yhat, batch_x, params, self.lbd)
                    loss.backward()
                    return loss
                self.opt.step(closure)
            elif self.opt_type == 'nncg':
                if epoch % 2 == 0:
                    self.opt.zero_grad()
                    yhat = self.model(batch_x)  
                    loss = self.model.physical_loss(yhat, batch_x, params, self.lbd)
                    grad_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                    self.opt.update_preconditioner(grad_tuple)
                def closure():
                    self.opt.zero_grad()
                    yhat = self.model(batch_x)  
                    loss = self.model.physical_loss(yhat, batch_x, params, self.lbd)
                    grad_tuple = torch.autograd.grad(loss, params, create_graph=True)
                    return loss, grad_tuple
                self.opt.step(closure)
            elif epoch>=switch_epoch_2 and self.opt_type == 'adam_lbfgs_nncg':
                if epoch == switch_epoch_2 or epoch - switch_epoch_2 % precond_upd_freq == 0:
                    self.opt_3.zero_grad()
                    yhat = self.model(batch_x)  
                    loss = self.model.physical_loss(yhat, batch_x, params, self.lbd)
                    grad_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                    self.opt_3.update_preconditioner(grad_tuple)
                def closure():
                    self.opt_3.zero_grad()
                    yhat = self.model(batch_x)  
                    loss = self.model.physical_loss(yhat, batch_x, params, self.lbd)
                    grad_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                    return loss, grad_tuple
                self.opt_3.step(closure)
            else:
                def closure():
                    self.opt_2.zero_grad()
                    yhat = self.model(batch_x)  
                    loss = self.model.physical_loss(yhat, batch_x, params, self.lbd)
                    loss.backward()
                    return loss
                self.opt_2.step(closure)
            # Update running losses
            if self.opt_type == 'lbfgs' or (self.opt_type in ['adam_lbfgs', 'adam_lbfgs_nncg'] and epoch>switch_epoch):
                self.opt.zero_grad()
                yhat = self.model(batch_x)  
                loss = self.model.physical_loss(yhat, batch_x, params, self.lbd)
            epoch_loss += loss.item()
            mse_loss += self.data_loss(yhat, einops.rearrange(batch_y, 'B ... C -> B (...) C')).item()
            
            # Save batch loss
            batch_result = {'train_batch_loss': loss.item(),
                            'batch': epoch * idxMax + idx}
           
            # log several solution for better visualization
            if self.wandb and (epoch % self.verbose_freq == 0 or epoch == self.n_epochs - 1) and idx < self.nvisu:
                custom_line2 = self.log_solution(self.evolvsol_tr, self.keysol_tr, epoch, batch_y, x, batch_x, idx, frame_size)
                self.logger.log({f"solution_train_{idx}": custom_line2})

            if self.wandb:
                self.logger.log(batch_result, step=epoch * idxMax + idx)

        return mse_loss / len(dataloader), _
    
    def _validate(self, dataloader, epoch):
            
        epoch_loss = 0
        mse_loss = 0
        idxMax = len(dataloader)

        for idx, (params, x, batch_y, _) in enumerate(dataloader):
            # load batch and shapes
            params = to4list(params, self.device) # .float() # params = u_b, g_b, omega, T => B, P, channels
            (x, x_in, x_ic, x_bc) = to4list(x, self.device) #  :-1 to remove boundary effects B, X, T, D
            batch_y = batch_y.to(self.device)
            bsize, frame_size, channels = batch_y.shape[0], batch_y.shape[1:-1], batch_y.shape[-1]
            dim=x.shape[-1]

            # remove spatial dim
            if dim > 1:
                x = einops.rearrange(x, 'B ... C -> B (...) C')
                x_in = einops.rearrange(x_in, 'B ... C -> B (...) C')
                x_ic = einops.rearrange(x_ic, 'B ... C -> B (...) C')
                x_bc = einops.rearrange(x_bc, 'B ... C -> B (...) C')
                batch_y = einops.rearrange(batch_y, 'B ... C -> B (...) C')

            batch_x = self.model.get_input(params, x)
            ntraj = bsize = batch_x.size(0)
            # sres = batch_x.shape[1]
            shapes_x = batch_x.shape # B, X, C

            batch_y = batch_y.to(self.device).float() 
            batch_x = batch_x.to(self.device).float().requires_grad_(True) 

            self.opt.zero_grad()
            yhat = self.model(batch_x)  
            loss = self.model.physical_loss(yhat, batch_x, params, self.lbd)
            
            # Update running losses
            epoch_loss += loss.item()
            mse_loss += self.data_loss(yhat, einops.rearrange(batch_y, 'B ... C -> B (...) C')).item()

            # Save batch loss
            batch_result = {'test_batch_loss': loss.item(),
                            'batch': epoch * idxMax + idx}
           
            # log several solution for better visualization
            if self.wandb and (epoch % self.verbose_freq == 0 or epoch == self.n_epochs - 1) and idx < self.nvisu:
                custom_line2 = self.log_solution(self.evolvsol_te, self.keysol_te, epoch, batch_y, x, batch_x, idx, frame_size)
                self.logger.log({f"solution_test_{idx}": custom_line2})

            if self.wandb:
                self.logger.log(batch_result, step = epoch * idxMax + idx)

        return mse_loss / len(dataloader), _
    
    def _evaluate(self, model, dataloader, ckpt=0, test_time_opt=0):

        self.model = model.to(self.device)
        self.model.device = self.device
        model.set_equation(self.all_cfg.data.name)

        if ckpt:
            self.load(ckpt)
            
        epoch_loss = 0
        mse_loss = 0
        idxMax = len(dataloader)

        mse = 0
        mae = 0
        rmse = 0
        lpde = 0
        mserel = 0
        rmserel = 0
        l1rel = 0
        ntrajloss = 0

        for idx, (params, x, batch_y, _) in enumerate(dataloader):
            # load batch and shapes
            params = to4list(params, self.device) # .float() # params = u_b, g_b, omega, T => B, P, channels
            (x, x_in, x_ic, x_bc) = to4list(x, self.device) #  :-1 to remove boundary effects B, X, T, D
            batch_y = batch_y.to(self.device)
            bsize, frame_size, channels = batch_y.shape[0], batch_y.shape[1:-1], batch_y.shape[-1]
            dim=x.shape[-1]

            # remove spatial dim
            if dim > 1:
                x = einops.rearrange(x, 'B ... C -> B (...) C')
                x_in = einops.rearrange(x_in, 'B ... C -> B (...) C')
                x_ic = einops.rearrange(x_ic, 'B ... C -> B (...) C')
                x_bc = einops.rearrange(x_bc, 'B ... C -> B (...) C')
                batch_y = einops.rearrange(batch_y, 'B ... C -> B (...) C')

            batch_x = self.model.get_input(params, x)
            ntraj = bsize = batch_x.size(0)
            # sres = batch_x.shape[1]
            shapes_x = batch_x.shape # B, X, C

            batch_y = batch_y.to(self.device).float() 
            batch_x = batch_x.to(self.device).float().requires_grad_(True) 
            tic=time.time()
            if test_time_opt:
                self.opt = self.init_optim('adam', self.model, self.cfg.lr, self.cfg)
                self.load(ckpt) # reload model to start from pre trained point
                self.ld=0
                for st in range(test_time_opt):
                    # detach from graph to make test-time opt
                    self.opt.zero_grad()
                    yhat = self.model(batch_x)  
                    loss = self.model.physical_loss(yhat, batch_x, params, self.lbd)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

            batch_x = batch_x.clone().detach().requires_grad_(True)
            yhat = self.model(batch_x)  
            
            batch_y = einops.rearrange(batch_y, 'B ... C -> B (...) C')
            lpde += self.model.physical_loss(yhat, batch_x, params, self.lbd).item() * bsize
            mse += F.mse_loss(yhat, batch_y).item() * bsize
            mae += F.l1_loss(yhat, batch_y).item() * bsize
            mserel += (torch.linalg.vector_norm(yhat-batch_y, dim=1)**2 / torch.linalg.vector_norm(batch_y, dim=1)**2).sum().item()
            rmserel += (torch.linalg.vector_norm(yhat-batch_y, dim=1) / torch.linalg.vector_norm(batch_y, dim=1)).sum().item()
            l1rel += (torch.linalg.vector_norm(yhat-batch_y, dim=1, ord=1) / torch.linalg.vector_norm(batch_y, dim=1, ord=1)).sum().item()
            ntrajloss += bsize
    
        metrics = {'MSE': mse / ntrajloss, 
                    'MSErel' : mserel / ntrajloss,
                    'RMSE': torch.sqrt(torch.tensor(mse) / ntrajloss).item(), 
                    'RMSErel': rmserel / ntrajloss, 
                    'L1': mae / ntrajloss,
                    'L1rel': l1rel / ntrajloss, 
                    'LPDE': lpde / ntrajloss}
        
        return metrics
    def log_solution(self, seq, leg, epoch, utrue, x, batch_x, idx, frame_size, params=False):
        # logs on only 1 traj : losses (PINNS (gd) + MSE) + solution at the end of training wrt GD steps
        if self.dim==1:
            if epoch == 0 :
                if idx==0:
                    leg.append('true')
                seq[idx].append(utrue[0, :].squeeze(-1).reshape(frame_size))
            us = self.model(batch_x)  
            seq[idx].append(us[0].squeeze().reshape(frame_size))
            if idx==0:
                leg.append(f'{epoch}')
            if self.wandb:
                custom_line2 = plot_losses(
                    xs=x[[0], ..., 0].repeat((len(seq[idx]), 1)),
                    ys=seq[idx],
                    keys=leg,
                    title="Solutions",
                    xname="x",
                    vega_spec="spatiotemp-isir/neural-solver"
                )

                return custom_line2
        
        elif self.dim==2:
            us = self.model(batch_x)  

            images, axs = plt.subplots(1, 2+params, figsize = (14, 7))
            axs[0].imshow(utrue[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy())
            axs[0].set_title('True')
            axs[1].imshow(us[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy())
            axs[1].set_title('Predicted')

            return images
        
        elif self.dim==3:
            ts = frame_size[-1] // 4
            us = self.model(batch_x)  
            images, axs = plt.subplots(2, 1+4, figsize = (14, 7))
            for t in range(5):
                axs[0, t].imshow(utrue[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy()[..., t*ts])
                axs[0, t].set_title(f'True t={t*ts}')
            for t in range(5):
                # print("us[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy().shape : ", us[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy().shape)
                axs[1, t].imshow(us[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy()[..., t*ts])
                axs[1, t].set_title(f'Predictied t={t*ts}')
            # plt.savefig(f'xp/vis/test_{self.all_cfg.data.name}_ppinns.png')
            return images
        
        else:
            raise ValueError(f'Logging of solution for dim {self.dim} not implemented. ')


    