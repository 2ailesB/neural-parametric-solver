import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops
import time
import wandb

from training.abstract import Experiment
from utils.gradients import gradients, gradients_NoGraph, gradients_clipped, gradients_normalized, gradients_normalizedAdam, gradients_adam
from init.init_approximation import init_approx
from utils.plot import plot_losses
from utils.device import to4list
from optimizer.init import init_optimizer
from training.physical_losses_Nd import Physical_Loss
from utils.data import dynamics_different_subsample_diffgrid

class trainingsolver_nd(Experiment):
    def __init__(self, cfg, loss_theta):
        """Trainer for solver gd in 1d

        Args:
            cfg (dict): configuration dictionary 
            tag (str, optional): name of expe. Defaults to ''.
        """
        super().__init__(cfg)

        self.adaptative = self.cfg.adaptative
        self.L = self.cfg.L
        self.schedule_L = self.cfg.schedule_L
        self.lbd = self.cfg.lbd

        self.u = init_approx(self.cfg.approx.name, self.cfg.approx)
        self.u.to(self.device)
        self.N = self.u.get_theta_size()
        # print("self.N : ", self.N)
        # print("trainer self.u.Ns4tr : ", self.u.Ns4tr)
        self.theta0 = (0.1)**(1/2) * torch.randn(self.u.Ns4tr+(self.u.channels,)).detach() 
        self.theta0[0, 0] = 0 # bias # TODO moche 
        self.theta0net = torch.nn.Linear(4, self.N).to(self.device)
        # print("self.theta0 : ", self.theta0)
        # exit()
        self.theta_init = self.cfg.theta_init
        self.theta_comp = self.cfg.theta_comp
        self.theta_noise = self.cfg.theta_noise
        self.theta_buffer = torch.zeros((100,) + tuple(self.u.Ns4tr) + (1,), device=self.device)

        self.inner_optimizer = init_optimizer(self.cfg.inner_optimizer.name, self.cfg.inner_optimizer)
        self.inner_optimizer.to(self.device)
        self.optsum = self.cfg.optsum
        self.optingd = self.cfg.optingd
        self.beta1 = self.cfg.inner_optimizer.beta1

        self.loss_theta = Physical_Loss(self.pde_name)
        self.loss_theta.set_reconstructionf(self.u)
        self.loss_theta = self.loss_theta.loss

        self.evolvgd_tr = []
        self.evolvgd_te = []
        self.evolvmse_tr = []
        self.evolvmse_te = []
        self.evolvsol_tr = [[] for i in range(self.nvisu)]
        self.evolvsol_te = [[] for i in range(self.nvisu)]
        self.keys_tr = []
        self.keys_te = []
        self.keysol_tr = []
        self.keysol_te = []

        self.dim = self.cfg.approx.dim

    def compute_gradtheta(self, outputs, inputs):
        if (self.theta_comp=='default') or (self.theta_comp=='true'):
            return gradients(outputs, inputs)
        elif self.theta_comp=='no_graph':
            return gradients_NoGraph(outputs, inputs)
        elif self.theta_comp=='clipped':
            return gradients_clipped(outputs, inputs)
        elif self.theta_comp=='normalized':
            return gradients_normalized(outputs, inputs)
        elif self.theta_comp=='adamnorm':
            return gradients_normalizedAdam(outputs, inputs)
        elif self.theta_comp=='adamgrad':
            return gradients_adam(outputs, inputs, self.beta1, self.mtm1)
        else:
            raise NotImplementedError(f'Grad theta computation with function {self.theta_comp} not implemented')

    def scheduler_L(self, epoch):
        if self.schedule_L == 0:
            return self.L
        elif isinstance(self.schedule_L, int) :
            if self.schedule_L > 0 and epoch % self.schedule_L == 0 and epoch > 0 :
                self.L += 1
                return self.L + 1
            else:
                return self.L
        elif self.schedule_L == 'random':
            return int(torch.randint(1, self.L, (1,)).detach())
    
    def get_theta0(self, bsize):
        if self.theta_init == 'fixed_random':
            return einops.repeat(self.theta0.clone(), '... -> B ...', B=bsize)
        elif self.theta_init == 'fixed_zeros':
            return torch.zeros((bsize,) + self.u.Ns4tr +(1,)).detach() * 1/self.N
        elif self.theta_init == 'fixed_ones':
            return torch.ones((bsize,) + self.u.Ns4tr +(1,)).detach() * 1/self.N
        elif self.theta_init == 'random':
            return torch.randn((bsize,) + self.u.Ns4tr +(1,)).detach() * 1/self.N
        else:
            raise ValueError(f'theta zeros function {self.theta_init} unknown')
    
    def fit(self, model, dataloader, validation_data=None, ckpt=None):
        return super().fit(model, dataloader, validation_data, ckpt)

    def _train_epoch(self, dataloader, epoch, vis=False):

        epoch_loss = 0
        mse_loss = 0
        idxMax = len(dataloader)
        L = self.scheduler_L(epoch)

        for idx, (params, x, utrue, _) in enumerate(dataloader):

            params = to4list(params, self.device) # .float() # params = u_b, g_b, omega, T => B, P, channels
            xa = (x, x_in, x_ic, x_bc) = to4list(x, self.device) #  B, X, T, D
                
            utrue = utrue.to(self.device)
            # print("utrue.shape : ", utrue.shape)
            bsize, frame_size, channels = utrue.shape[0], utrue.shape[1:-1], utrue.shape[-1]
            # print("frame_size : ", frame_size)

            theta = self.get_theta0(bsize).to(self.device) # B, 2N+1, 1
            # print("theta : ", theta)
            # print("self.all_cfg : ", self.all_cfg)
            # exit()
            # print("theta.shape : ", theta.shape)
            # theta = self.theta0net(params.squeeze(-1).float()).unsqueeze(-1)
            theta.requires_grad_(True)

            if not self.adaptative:
                losses = (loss_gd, lpde, lbc) = self.loss_theta(x_in, x_ic, x_bc, self.u, theta, params, self.lbd) # B, 1
                gradtheta = self.compute_gradtheta(loss_gd, theta) # B, 2N+1, C 

                input = self.model.get_input(theta, gradtheta, params, xa, self.u, losses, self.compute_gradtheta, 0)
                out = self.model(input)
            
            losses_gd = []
            mse_gd = []
            us_gd = []

            self.inner_optimizer.re_init()

            self.mtm1=0
            lossbw = 0
            for step in range(L):
                tic = time.time()
                if self.optingd: 
                    theta = theta.clone().detach().requires_grad_(True) # theta.grad=0? 
                
                losses = (loss_gd, lpde, lbc) = self.loss_theta(xa, theta
                , params, self.lbd)
                gradtheta = self.compute_gradtheta(loss_gd, theta)
                self.mtm1 = gradtheta.detach().clone()
                
                if self.adaptative:
                    input = self.model.get_input(theta, gradtheta, params, xa, self.u, losses, self.compute_gradtheta, step / self.L)
                    out = self.model(input)
                theta = self.inner_optimizer.update(theta, out, gradtheta) # + self.theta_noise * torch.randn(theta.shape, device=self.device) # , loss_gd) 

                self.inner_optimizer.schedule()

                thetac = theta
                if self.cfg.approx.dim > 1:
                    x = einops.rearrange(x, 'B ... C -> B (...) C')
                    thetac = einops.rearrange(theta, 'B ... -> B (...)').unsqueeze(-1)[:, 1:]
                uhat = self.u.compute_u(x, thetac)# B, X, C
                losses_gd.append(loss_gd[0].cpu().item())
                if self.dim==1:
                    us_gd.append(uhat[0, :].squeeze(-1).cpu())

                mse_gd.append(self.criterion(uhat, einops.rearrange(utrue, 'B ... C -> B (...) C')).cpu().item())

                if self.optingd and step < self.L-1: # last step optimize outside the loop
                    self.opt.zero_grad()
                    loss_net = self.criterion(uhat, einops.rearrange(utrue, 'B ... C -> B (...) C')) # * (step+1) / self.L #  + self.optsum * lossbw # + self.cfg.criterion.add_physics * self.loss_theta(x_in, x_ic, x_bc, self.u, theta, params, self.lbd)[0].mean() # B, 1 .mean() 

                    loss_net.backward(retain_graph=self.optsum)

                    self.opt.step()

                if self.optsum  and step < self.L-1: # accumulate losses along optimization
                    lossbw += self.criterion(uhat, einops.rearrange(utrue, 'B ... C -> B (...) C')) * (step+1) / self.L

                    if self.optingd: # optimize at each steps along the entire traj
                        self.opt.zero_grad()
                        loss_net.backward(retain_graph=self.optsum)
                        self.opt.step()

            self.opt.zero_grad()
            if self.cfg.approx.dim > 1:
                   utrue = einops.rearrange(utrue, 'B ... C -> B (...) C ')
            loss_net = self.criterion(uhat, utrue) + self.cfg.criterion.add_physics * loss_gd.mean()
            lossbw += loss_net
            if self.optsum:
                loss_net = lossbw
            loss_net.backward()

            self.opt.step()

            epoch_loss += self.criterion(uhat, utrue).item() # / ((self.optingd) * self.L + (not self.optingd)) # cf au-dessus, on divise par L si optingd = true
            mse_loss += torch.nn.functional.mse_loss(uhat, utrue).item() # / ((self.optingd) * self.L + (not self.optingd)) # cf au-dessus, on divise par L si optingd = true
            # Save batch loss
            batch_result = {'train_batch_loss': loss_net.item(),
                            'train_batch_rel_err': (torch.linalg.vector_norm(uhat-einops.rearrange(utrue, 'B ... C -> B (...) C'), dim=1) / torch.linalg.vector_norm(einops.rearrange(utrue, 'B ... C -> B (...) C'), dim=1)).sum().item(),
                            'batch': epoch * idxMax + idx}
            # logs on only 1 traj : losses (PINNS (gd) + MSE) + solution at the end of training wrt GD steps
            if (epoch % self.verbose_freq == 0 or epoch == self.n_epochs - 1) and idx == 0:
                self.evolvgd_tr.append(losses_gd)
                self.keys_tr.append(epoch)
                if self.wandb:
                    custom_line2 = plot_losses(
                        xs=list(range(self.L)),
                        ys=self.evolvgd_tr,
                        keys=self.keys_tr,
                        title="GD Losses ",
                        xname="GD steps",
                        vega_spec="spatiotemp-isir/neural-solver"
                    )
                    self.logger.log({"losses_gd_train": custom_line2})
                    # print(f"Epoch {epoch} : loss {loss_net.item()}")
                self.evolvmse_tr.append(mse_gd)
                if self.wandb:
                    custom_line2 = plot_losses(
                        xs=list(range(self.L)),
                        ys=self.evolvmse_tr,
                        keys=self.keys_tr,
                        title="GD MSE ",
                        xname="GD steps",
                        vega_spec="spatiotemp-isir/neural-solver"
                    )
                    self.logger.log({"mse_gd_train": custom_line2})
                    # print(f"Epoch {epoch} : loss {loss_net.item()}")
                
                if self.dim==1 and epoch == self.n_epochs - 1: # TODO : faire pour la 2d # last epoch plot sol wrt GD steps
                    if self.wandb:
                        us_gd.insert(0, utrue[0, :].squeeze(-1))
                        custom_line2 = plot_losses(
                            xs=x[[0], ..., 0].repeat((self.L + 1, 1)),
                            ys=us_gd,
                            keys=['true'] + [f'{i}' for i in range(self.L)],
                            title="GD solutions",
                            xname="x",
                            vega_spec="spatiotemp-isir/neural-solver"
                        )
                        self.logger.log({"solution_gd_train": custom_line2})

            # log several solution for better visualization
            if self.wandb and (epoch % self.verbose_freq == 0 or epoch == self.n_epochs - 1) and idx < self.nvisu:
            # if 1:
                custom_line2 = self.log_solution(self.evolvsol_tr, self.keysol_tr, epoch, utrue, thetac, x, idx, frame_size)
                self.logger.log({f"solution_train_{idx}": custom_line2})
                
            if self.wandb:
                self.logger.log(batch_result, step=epoch * idxMax + idx)
        return epoch_loss / len(dataloader), mse_loss / len(dataloader)

    def _validate(self, dataloader, epoch, vis=False):
        epoch_loss = 0
        mse_loss = 0
        idxMax = len(dataloader)

        for idx, (params, x, utrue, pidx) in enumerate(dataloader):
            params = to4list(params, self.device) # params = u_b, g_b, omega, T => B, ..., channels
            xa = (x, x_in, x_ic, x_bc) = to4list(x, self.device) #  cf __getitem__
            utrue = utrue.to(self.device) # :-1 to remove boundary effects
            bsize, frame_size, channels = utrue.shape[0], utrue.shape[1:-1], utrue.shape[-1]

            theta = self.get_theta0(bsize).to(self.device) # B, 2N+1, 1
            theta.requires_grad_(True)

            if not self.adaptative:
                losses = (loss_gd, lpde, lbc) = self.loss_theta(x_in, x_ic, x_bc, self.u, theta, params, self.lbd)
                gradtheta = self.compute_gradtheta(loss_gd, theta)

                input = self.model.get_input(theta, gradtheta, params, xa, self.u, losses, self.compute_gradtheta, 0)
                out = self.model(input)
            
            losses_gd = []
            mse_gd = []
            us_gd = []

            self.inner_optimizer.re_init()
            self.mtm1 = 0

            for step in range(self.L):
                if self.optingd: 
                    theta = theta.clone().detach().requires_grad_(True)
                losses = (loss_gd, lpde, lbc) = self.loss_theta(xa, theta, params, self.lbd)
                gradtheta = self.compute_gradtheta(loss_gd, theta)
                self.mtm1 = gradtheta# .detach().clone()

                if self.adaptative:
                    input = self.model.get_input(theta, gradtheta, params, xa, self.u, losses, self.compute_gradtheta, step / self.L)
                    out = self.model(input)

                thetac = theta = self.inner_optimizer.update(theta, out, gradtheta) # , loss_gd)

                self.inner_optimizer.schedule()

                if self.cfg.approx.dim > 1:
                    x = einops.rearrange(x, 'B ... C -> B (...) C')
                    thetac = einops.rearrange(theta, 'B ... -> B (...)').unsqueeze(-1)[:, 1:]
                uhat = self.u.compute_u(x, thetac)# B, X, C

                losses_gd.append(loss_gd[0].cpu().item())
                if self.dim==1:
                    us_gd.append(uhat[0, :].squeeze(-1))
                mse_gd.append(self.criterion(uhat, einops.rearrange(utrue, 'B ... C -> B (...) C')).cpu().item())

            if self.cfg.approx.dim > 1:
                   utrue = einops.rearrange(utrue, 'B ... C -> B (...) C ')
            loss_net = self.criterion(uhat, utrue) + self.cfg.criterion.add_physics * loss_gd.mean() # B, 1 .mean() 
            epoch_loss += loss_net.item()
            mse_loss += torch.nn.functional.mse_loss(uhat, utrue).item()

            batch_result = {'test_batch_loss': loss_net.item(),
                            'test_batch_rel_err': (torch.linalg.vector_norm(uhat-einops.rearrange(utrue, 'B ... C -> B (...) C'), dim=1) / torch.linalg.vector_norm(einops.rearrange(utrue, 'B ... C -> B (...) C'), dim=1)).sum().item(),
                            'batch': epoch * idxMax + idx}
            
            if (epoch % self.verbose_freq == 0 or epoch == self.n_epochs - 1) and idx == 0:
                self.keys_te.append(epoch)
                self.evolvgd_te.append(losses_gd)
                if self.wandb:
                    custom_line2 = plot_losses(
                        xs=list(range(self.L)),
                        ys=self.evolvgd_te,
                        keys=self.keys_te,
                        title="GD Losses - test",
                        xname="GD steps",
                        vega_spec="spatiotemp-isir/neural-solver"
                    )
                    self.logger.log({"losses_gd_test": custom_line2})
                self.evolvmse_te.append(mse_gd)
                if self.wandb:
                    custom_line2 = plot_losses(
                        xs=list(range(self.L)),
                        ys=self.evolvmse_te,
                        keys=self.keys_te,
                        title="GD MSE - test ",
                        xname="GD steps",
                        vega_spec="spatiotemp-isir/neural-solver"
                    )
                    self.logger.log({"mse_gd_test": custom_line2})

                # print(f"Epoch {epoch} : loss {loss_net.item()}")
                if self.dim==1 and epoch == self.n_epochs - 1: # last epoch plot sol wrt GD steps
                    if self.wandb:
                        us_gd.insert(0, utrue[0, :].squeeze(-1))
                        custom_line2 = plot_losses(
                            xs=x[[0], ..., 0].repeat((self.L + 1, 1)),
                            ys=us_gd,
                            keys=['true'] + [f'{i}' for i in range(self.L)],
                            title="GD solutions",
                            xname="x",
                            vega_spec="spatiotemp-isir/neural-solver"
                        )
                        self.logger.log({"solution_gd_test": custom_line2})

            # log several solution for better visualization
            if self.wandb and (epoch % self.verbose_freq == 0 or epoch == self.n_epochs - 1) and idx < self.nvisu:
                custom_line2 = self.log_solution(self.evolvsol_te, self.keysol_te, epoch, utrue, thetac, x, idx, frame_size)
                self.logger.log({f"solution_test_{idx}": custom_line2})
                
            if self.wandb:
                self.logger.log(batch_result, step=epoch * idxMax + idx)
        
        return epoch_loss / len(dataloader), mse_loss / len(dataloader)
    
    def _evaluate(self, model, dataloader, ckpt=0, test_time_opt=0, plot_sol=0):
        # print("self.all_cfg : ", self.all_cfg)
        epoch_loss = 0
        idxMax = len(dataloader)
        self.model = model.to(self.device)
        self.model.device = self.device

        if ckpt:
            self.load(ckpt)

        mse = 0
        mse2 = 0
        mae = 0
        rmse = 0
        lpinns = 0
        mserel = 0
        rmserel = 0
        l1rel = 0
        inf_time = 0
        ntrajloss = 0

        i=0

        for idx, (params, x, utrue, pidx) in enumerate(dataloader):
            
            params = to4list(params, self.device) # params = u_b, g_b, omega, T => B, ..., channels
            xa = (x, x_in, x_ic, x_bc) = to4list(x, self.device) #  cf __getitem__
            utrue = utrue.to(self.device) # :-1 to remove boundary effects
            bsize, frame_size, channels = utrue.shape[0], utrue.shape[1:-1], utrue.shape[-1]

            theta = self.get_theta0(bsize).to(self.device) # B, 2N+1, 1
            
            theta.requires_grad_(True)

            if not self.adaptative:
                losses = (loss_gd, lpde, lbc) = self.loss_theta(x_in, x_ic, x_bc, self.u, theta, params, self.lbd)
                gradtheta = self.compute_gradtheta(loss_gd, theta)

                input = self.model.get_input(theta, gradtheta, params, xa, self.u, losses, self.compute_gradtheta, 0)
                out = self.model(input)
            
            tic = time.time()

            self.inner_optimizer.re_init()
            self.mtm1 = 0

            for step in range(self.L):
                if self.optingd: 
                    theta = theta.clone().detach().requires_grad_(True)
                losses = (loss_gd, lpde, lbc) = self.loss_theta(xa, theta, params, self.lbd)
                gradtheta = self.compute_gradtheta(loss_gd, theta)
                self.mtm1 = gradtheta# .detach().clone()

                if self.adaptative:
                    input = self.model.get_input(theta, gradtheta, params, xa, self.u, losses, self.compute_gradtheta, step / self.L)
                    out = self.model(input)

                thetac = theta = self.inner_optimizer.update(theta, out, gradtheta) # , loss_gd)

                self.inner_optimizer.schedule()

                if self.cfg.approx.dim > 1:
                    x = einops.rearrange(x, 'B ... C -> B (...) C')
                    thetac = einops.rearrange(theta, 'B ... -> B (...)').unsqueeze(-1)[:, 1:]
                uhat = self.u.compute_u(x, thetac)# B, X, C

                if self.cfg.approx.dim > 1:
                   utrue = einops.rearrange(utrue, 'B ... C -> B (...) C ')

            lpinns += lpde.sum().item()
            if plot_sol:
                plt.plot(uhat.squeeze(-1).detach().cpu().numpy().T)
                plt.plot(utrue.squeeze(-1).detach().cpu().numpy().T)
                plt.legend(['Reconstruction', 'Ground truth'])
                plt.xlabel('x')
                plt.ylabel('Solution')
                plt.savefig('xp/vis/reb/ICLR/compyhaty_gdnn_reg.pdf')
                exit()
            if self.cfg.approx.dim > 1:
                   utrue = einops.rearrange(utrue, 'B ... C -> B (...) C ')
            
            mse += F.mse_loss(uhat, utrue).item() * bsize 
            mse2 += self.criterion(uhat, utrue).item()
            mae += F.l1_loss(uhat, utrue).item() * bsize
            mserel += (torch.linalg.vector_norm(uhat-utrue, dim=1)**2 / torch.linalg.vector_norm(utrue, dim=1)**2).sum().item()
            rmserel += (torch.linalg.vector_norm(uhat-utrue, dim=1) / torch.linalg.vector_norm(utrue, dim=1)).sum().item()
            l1rel += (torch.linalg.vector_norm(uhat-utrue, dim=1, ord=1) / torch.linalg.vector_norm(utrue, dim=1, ord=1)).sum().item()
            ntrajloss += bsize
    
        metrics = {'trainingCrit': mse2 / len(dataloader),
                    'MSE': mse / ntrajloss, 
                    'MSErel' : mserel / ntrajloss,
                    'RMSE': torch.sqrt(torch.tensor(mse) / ntrajloss).item(), 
                    'RMSErel': rmserel / ntrajloss, 
                    'L1': mae / ntrajloss,
                    'L1rel': l1rel / ntrajloss, 
                    'LPDE': lpinns / ntrajloss}
        
        return metrics

    def forward(self, params, x):
        params = to4list(params, self.device) # params = u_b, g_b, omega, T => B, ..., channels
        xa = (x, x_in, x_ic, x_bc) = to4list(x, self.device) #  cf __getitem__
        
        bsize = x.shape[0]
        theta = self.get_theta0(bsize).to(self.device) # B, 2N+1, 1
        theta.requires_grad_(True)

        if not self.adaptative:
            losses = (loss_gd, lpde, lbc) = self.loss_theta(x_in, x_ic, x_bc, self.u, theta, params, self.lbd)
            gradtheta = self.compute_gradtheta(loss_gd, theta)

            input = self.model.get_input(theta, gradtheta, params, xa, self.u, losses, self.compute_gradtheta, 0)
            out = self.model(input)
        
        losses_gd = []

        self.inner_optimizer.re_init()
        
        for step in range(self.L): # in forward mode, no need of optingd bc this parameter only specify the training mode of the net
            losses = (loss_gd, lpde, lbc) = self.loss_theta(x_in, x_ic, x_bc, self.u, theta, params, self.lbd) # + self.criterion(uhat, utrue)# B, 1
            gradtheta = self.compute_gradtheta(loss_gd, theta)

            if self.adaptative:
                input = self.model.get_input(theta, gradtheta, params, xa, self.u, losses, self.compute_gradtheta, step / self.L)
                out = self.model(input)

            theta = self.inner_optimizer.update(theta, out, gradtheta) 

        uhat = self.u.compute_u(x, theta)# B, X, C

        return uhat
    
    def check_assertions(self):
        if self.model.cfg.name == 'gd_nn' and self.cfg.inner_optimizer.conditioned==1:
            print('Warning using GD_NN if conditioned=True (because no conditionement involved in this model)')
        if self.model.cfg.name in ['preconditioner', 'preconditioner_cholesky', 'preconditioner_pinns']:
            assert self.cfg.inner_optimizer.conditioned == 1, f'Cannot use {self.model.cfg.name} if conditioned=False (because conditionement required in this model)'

        return super().check_assertions()

    def log_solution(self, seq, leg, epoch, utrue, theta, x, idx, frame_size, params=False):
        # logs on only 1 traj : losses (PINNS (gd) + MSE) + solution at the end of training wrt GD steps

        if self.dim==1:
            if epoch == 0 :
                if idx==0:
                    leg.append('true')
                seq[idx].append(utrue[0, :].squeeze(-1).reshape(frame_size))
            us = self.u.compute_u(x, theta)
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
            us = self.u.compute_u(x, theta)

            images, axs = plt.subplots(1, 2+params, figsize = (14, 7))
            axs[0].imshow(utrue[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy())
            axs[0].set_title('True')
            axs[1].imshow(us[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy())
            axs[1].set_title('Predicted')

            return images
        
        elif self.dim==3:
            us = self.u.compute_u(x, theta) # B, XYT, 1 frame_size=X, Y, T
            ts = frame_size[-1] // 4

            images, axs = plt.subplots(2, 1+4, figsize = (14, 7))
            for t in range(4):
                axs[0, t].imshow(utrue[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy()[..., t*ts])
                axs[0, t].set_title(f'True t={t*ts}')
            for t in range(4):
                axs[1, t].imshow(us[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy()[..., t*ts])
                axs[1, t].set_title(f'Predictied t={t*ts}')
            axs[0, -1].imshow(utrue[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy()[..., -1])
            axs[0, -1].set_title(f'True t={-1}')
            axs[1, -1].imshow(us[0].squeeze(-1).reshape(frame_size).detach().cpu().numpy()[..., -1])
            axs[1, -1].set_title(f'Predictied t={-1}')
            plt.savefig('xp/vis/test_2dns.png')
            return images
        
        else:
            raise ValueError(f'Logging of solution for dim {self.dim} not implemented. ')

    def load(self, ckpt_path):
        super().load(ckpt_path)

        self.theta0 = self.state['init']

    def save(self, lr, n):
        self.state['init'] = self.theta0
        super().save(lr, n)



