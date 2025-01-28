import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import einops
import wandb

from training.abstract import Experiment
from utils.plot import plot_losses
from utils.device import to4list

class DataDriven_training(Experiment):
    def __init__(self, cfg):
        """Trainer for data in 2d spatial coordinates

        Args:
            cfg (dict): configuration dictionary 
            tag (str, optional): name of expe. Defaults to ''.
        """
        super().__init__(cfg)

        self.spatial_dim = 1
        self.evolvsol_tr = [[] for i in range(self.nvisu)]
        self.evolvsol_te = [[] for i in range(self.nvisu)]
        self.keysol_tr = []
        self.keysol_te = []

    def _train_epoch(self, dataloader, epoch, vis=False):

        epoch_loss = 0
        idxMax = len(dataloader)

        for idx, (params, x, batch_y, _) in enumerate(dataloader):
            
            # load batch and shapes
            xa = (x, x_in, x_ic, x_bc) = to4list(x, self.device) #  cf __getitem__
            datas = to4list(params, self.device) # params = u_b, g_b, omega, T => B, ..., channels
            # x = x.to(self.device)
            batch_y = batch_y.to(self.device)
            bsize, frame_size, channels = batch_y.shape[0], batch_y.shape[1:-1], batch_y.shape[-1]

            ntraj = bsize = batch_y.size(0)
            sres = batch_y.shape[1]
            shapes_x = batch_y.shape # B, X, C

            nspatial = batch_y.shape[1]
            channels = batch_y.shape[2]
            
            x, batch_x = self.model.get_input(datas, xa)

            batch_y = batch_y.to(self.device).float() 
            batch_x = batch_x.to(self.device).float() 

            yhat = self.model(x, batch_x)  
            self.opt.zero_grad()
            batch_y = einops.rearrange(batch_y, 'B ... C -> B (...) C')
            loss = self.criterion(yhat, batch_y)
            loss.backward()
            self.opt.step()
            
            # Update running losses
            epoch_loss += loss.item()

            # Save batch loss
            batch_result = {'train_batch_loss': loss.item(),
                            'train_batch_rel_err': (torch.linalg.vector_norm(yhat-batch_y, dim=1) / torch.linalg.vector_norm(batch_y, dim=1)).sum().item(),
                            'batch': epoch * idxMax + idx}
           
            # log several solution for better visualization
            # if 1:
            if self.wandb and (epoch % self.verbose_freq == 0 or epoch == self.n_epochs - 1) and idx < self.nvisu:
                custom_line2 = self.log_solution(self.evolvsol_tr, self.keysol_tr, epoch, batch_y, yhat, x, idx, frame_size)
                self.logger.log({f"solution_train_{idx}": custom_line2})
                 
            if self.wandb:
                self.logger.log(batch_result, step=epoch * idxMax + idx)

        return epoch_loss / len(dataloader), _

    def _validate(self, dataloader, epoch, vis=False):
        epoch_loss = 0
        idxMax = len(dataloader)

        for idx, (params, x, batch_y, _) in enumerate(dataloader):

            # load batch and shapes
            xa = (x, x_in, x_ic, x_bc) = to4list(x, self.device) #  cf __getitem__
            datas = to4list(params, self.device) # params = u_b, g_b, omega, T => B, ..., channels
            x = x.to(self.device)
            batch_y = batch_y.to(self.device)
            bsize, frame_size, channels = batch_y.shape[0], batch_y.shape[1:-1], batch_y.shape[-1]

            ntraj = bsize = batch_y.size(0)
            sres = batch_y.shape[1]
            shapes_x = batch_y.shape # B, X, C

            nspatial = batch_y.shape[1]
            channels = batch_y.shape[2]
            x, batch_x = self.model.get_input(datas, xa)

            batch_y = batch_y.to(self.device).float() 
            batch_x = batch_x.to(self.device).float() 

            yhat = self.model(x, batch_x)  
            batch_y = einops.rearrange(batch_y, 'B ... C -> B (...) C')
            loss = self.criterion(yhat, batch_y)

            # Update running losses
            epoch_loss += loss.item()

            # Save batch loss
            batch_result = {'test_batch_loss': loss.item(),
                            'test_batch_rel_err': (torch.linalg.vector_norm(yhat-batch_y, dim=1) / torch.linalg.vector_norm(batch_y, dim=1)).sum().item(),
                            'batch': epoch * idxMax + idx}
            
            # log several solution for better visualization
            if self.wandb and (epoch % self.verbose_freq == 0 or epoch == self.n_epochs - 1) and idx < self.nvisu:
                custom_line2 = self.log_solution(self.evolvsol_te, self.keysol_te, epoch, batch_y, yhat, x, idx, frame_size)
                self.logger.log({f"solution_test_{idx}": custom_line2})

        return epoch_loss / len(dataloader), _
    
    def _evaluate(self, model, dataloader, ckpt=0, test_time_opt=0):
        epoch_loss = 0
        idxMax = len(dataloader)

        self.model = model.to(self.device)
        self.model.device = self.device
        model.set_equation(self.all_cfg.data.name)

        if ckpt:
            self.load(ckpt)

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
            xa = (x, x_in, x_ic, x_bc) = to4list(x, self.device) #  cf __getitem__
            datas = to4list(params, self.device) # params = u_b, g_b, omega, T => B, ..., channels
            x = x.to(self.device)
            batch_y = batch_y.to(self.device)
            bsize, frame_size, channels = batch_y.shape[0], batch_y.shape[1:-1], batch_y.shape[-1]

            theta = step = 0

            ntraj = bsize = batch_y.size(0)
            sres = batch_y.shape[1]
            shapes_x = batch_y.shape # B, X, C

            # remove spatial dim
            if self.dim > 1:
                x = einops.rearrange(x, 'B ... C -> B (...) C')
                x_in = einops.rearrange(x_in, 'B ... C -> B (...) C')
                x_ic = einops.rearrange(x_ic, 'B ... C -> B (...) C')
                x_bc = einops.rearrange(x_bc, 'B ... C -> B (...) C')
                batch_y = einops.rearrange(batch_y, 'B ... C -> B (...) C')
            
            nspatial = batch_y.shape[1]
            channels = batch_y.shape[2]
            x, batch_x = self.model.get_input(datas, xa)

            batch_y = batch_y.to(self.device).float() 
            batch_x = batch_x.to(self.device).float() # .requires_grad_(True)

            yhat = self.model(x, batch_x)  

            mse += F.mse_loss(yhat,  batch_y).item() * bsize
            mae += F.l1_loss(yhat,  batch_y).item() * bsize
            mserel += (torch.linalg.vector_norm(yhat - batch_y, dim=1)**2 / torch.linalg.vector_norm(batch_y, dim=1)**2).sum().item()
            rmserel += (torch.linalg.vector_norm(yhat - batch_y, dim=1) / torch.linalg.vector_norm(batch_y, dim=1)).sum().item()
            l1rel += (torch.linalg.vector_norm(yhat - batch_y, dim=1, ord=1) / torch.linalg.vector_norm(batch_y, dim=1, ord=1)).sum().item()
            ntrajloss += bsize

        metrics = {'MSE': mse / ntrajloss, 
                'MSErel' : mserel / ntrajloss,
                'RMSE': torch.sqrt(torch.tensor(mse) / ntrajloss).item(), 
                'L1': mae / ntrajloss,
                'L1rel': l1rel / ntrajloss, 
                'LPDE': lpde / ntrajloss}

        return metrics


    def log_solution(self, seq, leg, epoch, utrue, uhat, x, idx, frame_size, params=False):
        if self.model.u.dim==1:
            if epoch == 0:
                seq[idx].append(utrue[0].squeeze())
                if idx==0:
                    leg.append('true')
            seq[idx].append(uhat[0].squeeze())
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

        elif self.model.u.dim==2:
            images, axs = plt.subplots(1, 2+params, figsize = (14, 7))
            axs[0].imshow(utrue[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy())
            axs[0].set_title('True')
            axs[1].imshow(uhat[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy())
            axs[1].set_title('Predicted')
            return images
        
        elif self.dim==3:
            ts = frame_size[-1] // 4

            images, axs = plt.subplots(2, 1+4, figsize = (14, 7))
            for t in range(4):
                axs[0, t].imshow(utrue[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy()[..., t*ts])
                axs[0, t].set_title(f'True t={t*ts}')
            for t in range(4):
                axs[1, t].imshow(uhat[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy()[..., t*ts])
                axs[1, t].set_title(f'Predictied t={t*ts}')
            axs[0, -1].imshow(utrue[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy()[..., -1])
            axs[0, -1].set_title(f'True t={frame_size[-1]}')
            axs[1, -1].imshow(uhat[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy()[..., -1])
            axs[1, -1].set_title(f'Predictied t={frame_size[-1]}')
            
            # plt.savefig('xp/vis/test_2dwave.png')
            return images
        
        else:
            raise ValueError(f'Logging of solution for dim {self.dim} not implemented. ')


