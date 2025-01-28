import torch
import torch.nn.functional as F
import einops
import matplotlib.pyplot as plt

from training.abstract import Experiment
from utils.plot import plot_losses
from utils.device import to4list

class Hybrid_training(Experiment):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.lp = self.cfg.lp
        self.ld = self.cfg.ld
        self.lbd = self.cfg.lbd

        self.evolvsol_tr = [[] for i in range(self.nvisu)]
        self.evolvsol_te = [[] for i in range(self.nvisu)]
        self.keysol_tr = []
        self.keysol_te = []
        self.data_loss = self.criterion
        
    def fit(self, model, dataloader, validation_data=None, ckpt=None):
        model.set_equation(self.all_cfg.data.name)
        return super().fit(model, dataloader, validation_data, ckpt)

    def _train_epoch(self, dataloader, epoch, vis=False):
        epoch_loss = 0
        mse_loss = 0
        idxMax = len(dataloader)

        for idx, (params, x, batch_y, _) in enumerate(dataloader):
            
            # load batch and shapes
            params = to4list(params, self.device) # params = u_b, g_b, omega, T => B, ..., channels
            (x, x_in, x_ic, x_bc) = to4list(x, self.device)
            batch_y = batch_y.to(self.device).float()   

            bsize, frame_size, channels = batch_y.shape[0], batch_y.shape[1:-1], batch_y.shape[-1]
            self.model.coord_size = frame_size

            batch_x = self.model.get_input(params, x)
            
            batch_x = batch_x.to(self.device).float().requires_grad_(True) 
            # print("batch_x.shape : ", batch_x.shape)

            self.opt.zero_grad()
            yhat = self.model(batch_x) 
            lp=ld=0 
            if self.lp:
                lp = self.lp*self.model.physical_loss(yhat, batch_x, params, self.lbd)
            if self.ld:
                ld = self.ld*self.data_loss(yhat, einops.rearrange(batch_y, 'B ... C -> B (...) C'))
            loss = lp + ld
            loss.backward()
            self.opt.step()

            # Update running losses
            epoch_loss += loss.item()
            mse_loss += self.data_loss(yhat, einops.rearrange(batch_y, 'B ... C -> B (...) C')).item()

            # Save batch loss
            batch_result = {'train_batch_loss': loss.item(),
                            'batch': epoch * idxMax + idx}

            # log several solution for better visualization
            if self.wandb and (epoch % self.verbose_freq == 0 or epoch == self.n_epochs - 1) and idx < self.nvisu:
                custom_line2 = self.log_solution(self.evolvsol_tr, self.keysol_tr, epoch, batch_y, yhat, x, idx, frame_size)
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
            params = to4list(params, self.device) # params = u_b, g_b, omega, T => B, ..., channels
            (x, x_in, x_ic, x_bc) = to4list(x, self.device)

            bsize, frame_size, channels = batch_y.shape[0], batch_y.shape[1:-1], batch_y.shape[-1]

            batch_x = self.model.get_input(params, x) # TODO: to clean

            ntraj = bsize = batch_y.size(0)
            shapes_x = batch_y.shape # B, X, C

            batch_y = batch_y.to(self.device).float()             
            nspatial = batch_y.shape[1]
            channels = batch_y.shape[-1]
            
            batch_x = batch_x.to(self.device).float().requires_grad_(True) 
            self.opt.zero_grad()
            yhat = self.model(batch_x)  
            lp=ld=0 
            if self.lp:
                lp = self.lp*self.model.physical_loss(yhat, batch_x, params, self.lbd)
            if self.ld:
                ld = self.ld*self.data_loss(yhat, einops.rearrange(batch_y, 'B ... C -> B (...) C'))
            loss = lp + ld
            batch_x.requires_grad_(False)
            # Update running losses
            epoch_loss += loss.item()
            mse_loss += self.data_loss(yhat, einops.rearrange(batch_y, 'B ... C -> B (...) C')).item()

            # Save batch loss
            batch_result = {'test_batch_loss': loss.item(),
                            'batch': epoch * idxMax + idx}
            # self.wandb=1
            if self.wandb and (epoch % self.verbose_freq == 0 or epoch == self.n_epochs - 1) and idx < self.nvisu:
                custom_line2 = self.log_solution(self.evolvsol_te, self.keysol_te, epoch, batch_y, yhat, x, idx, frame_size)
                self.logger.log({f"solution_test_{idx}": custom_line2})
                
            if self.wandb:
                self.logger.log(batch_result, step = epoch * idxMax + idx)

        return mse_loss / len(dataloader), _
    
    def _evaluate(self, model, dataloader, ckpt=0, test_time_opt=0, plot_sol=0):
        
        self.model = model.to(self.device)
        self.model.device = self.device
        model.set_equation(self.all_cfg.data.name)

        # new optimizer

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
            params = to4list(params, self.device) # params = u_b, g_b, omega, T => B, ..., channels
            (x, x_in, x_ic, x_bc) = to4list(x, self.device)

            bsize, frame_size, channels = batch_y.shape[0], batch_y.shape[1:-1], batch_y.shape[-1]

            batch_x = self.model.get_input(params, x) # TODO: to clean

            ntraj = bsize = batch_y.size(0)
            shapes_x = batch_y.shape # B, X, C

            batch_y = batch_y.to(self.device).float()             
            
            nspatial = batch_y.shape[1]
            channels = batch_y.shape[-1]
            
            batch_x = batch_x.to(self.device).float().requires_grad_(True) 
            if test_time_opt:
                self.opt = self.init_optim('adam', self.model, self.cfg.lr, self.cfg)
                self.load(ckpt) # reload model to start from pre trained point
                self.ld=0
                for _ in range(test_time_opt):
                    # detach from graph to make test-time opt
                    self.opt.zero_grad()
                    yhat = self.model(batch_x)  
                    loss = self.lp * self.model.physical_loss(yhat, batch_x, params, self.lbd) # + self.ld * self.data_loss(yhat, einops.rearrange(batch_y, 'B ... C -> B (...) C')) # TODO
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

            batch_x = batch_x.clone().detach().requires_grad_(True)
            yhat = self.model(batch_x)  

            batch_y = einops.rearrange(batch_y, 'B ... C -> B (...) C')
            lpde += self.model.physical_loss(yhat, batch_x, params, self.lbd).item() * bsize
            mse += F.mse_loss(yhat, batch_y).item() * bsize
            mae += F.l1_loss(yhat, batch_y).item() * bsize
            
            mserel += (torch.linalg.vector_norm(yhat - batch_y, dim=1)**2 / torch.linalg.vector_norm(batch_y, dim=1)**2).sum().item()
            rmserel += (torch.linalg.vector_norm(yhat - batch_y, dim=1) / torch.linalg.vector_norm(batch_y, dim=1)).sum().item()
            l1rel += (torch.linalg.vector_norm(yhat - batch_y, dim=1, ord=1) / torch.linalg.vector_norm(batch_y, dim=1, ord=1)).sum().item()
            ntrajloss += bsize
            
        metrics = {'MSE': mse / ntrajloss, 
                'MSErel' : mserel / ntrajloss,
                'RMSE': torch.sqrt(torch.tensor(mse) / ntrajloss).item(), 
                'RMSErel': rmserel / ntrajloss, 
                'L1': mae / ntrajloss,
                'L1rel': l1rel / ntrajloss, 
                'LPDE': lpde / ntrajloss}
        
        return metrics


    def log_solution(self, seq, leg, epoch, utrue, uhat, x, idx, frame_size, params=False):
        if self.model.dim==1:
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

        elif self.model.dim==2:
            images, axs = plt.subplots(1, 2+params, figsize = (14, 7))
            axs[0].imshow(utrue[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy())
            axs[0].set_title('True')
            axs[1].imshow(uhat[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy())
            axs[1].set_title('Predicted')
            return images
        elif self.model.dim==3:
            pass
        else:
            raise ValueError(f'Logging of solution for model.u.dim {self.model.u.dim} not implemented. ')