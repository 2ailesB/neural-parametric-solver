import torch.nn as nn
import torch
import einops
import matplotlib.pyplot as plt
import time

from utils.gradients import gradients

class Model(nn.Module):
    def __init__(self, cfg, pde_cfg, *args, **kwargs) -> None:
        """Abstract class for models

        Args:
            cfg (omegaconf.dictconfig.DictConfig): configuration
            pde_cfg (tuple): the pde configuration as :
            (PDE Physical losses, sizes (frame, params, forcings, ic, bc), channels (frame, params, forcings, ic, bc), PDE dim, PDE channels)
            PDE informations are used to define the input size of the model
        """
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.pde_cfg = pde_cfg
        self.pde = pde_cfg[0]

        self.frame_size = pde_cfg[1][0]
        self.paramssize = pde_cfg[1][1]
        self.forcingssize = pde_cfg[1][2]
        self.icsize = pde_cfg[1][3]
        self.bcsize = pde_cfg[1][4]

        self.frame_channels = pde_cfg[2][0]
        self.paramschannels = pde_cfg[2][1]
        self.forcingschannels = pde_cfg[2][2]
        self.icchannels = pde_cfg[2][3]
        self.bcchannels = pde_cfg[2][4]

        self.dim=pde_cfg[3]
        self.channels = pde_cfg[4]

    def get_input_size(self):
        """ get the input size of the model
            Used to create layers with the right input size automatically depending on the chosen inputs
        Returns:
            int: the input size
        """
        if self.cfg.nn.name=='conv1' or self.cfg.nn.name=="fno":
            return self.get_input_size_fno()
        return self.get_input_size_mlp()

    def get_input_size_mlp(self):
        input_dim = 0
        if self.cfg.input_theta:
            input_dim += self.N
        if self.cfg.input_gradtheta:
            input_dim += self.N
        if self.cfg.input_params:
            input_dim += self.paramssize
        if self.cfg.input_loss:
            input_dim += 1
        if self.cfg.input_x:
            input_dim += self.frame_size
        if self.cfg.input_residual:
            input_dim += self.frame_size
        if self.cfg.input_bcloss:
            input_dim += self.frame_size
        if self.cfg.input_step:
            input_dim += 1
        if self.cfg.input_splitted_grad:
            input_dim += 2*self.N
        if self.cfg.input_splitted_losses:
            input_dim += 2
        if self.cfg.input_forcings:
            input_dim += self.forcingssize
        if self.cfg.input_ic:
            input_dim += self.icsize 
        if self.cfg.input_bc:
            input_dim += self.bcsize 
        if self.cfg.input_signlog:
            input_dim += 2*self.N

        return input_dim

    def get_input_size_fno(self):
        input_dim = 0
        if self.cfg.input_theta:
            input_dim += self.channels
        if self.cfg.input_gradtheta:
            input_dim += self.channels
        if self.cfg.input_params:
            input_dim += self.paramschannels
        if self.cfg.input_loss:
            input_dim += 1
        if self.cfg.input_x:
            input_dim += self.frame_size
        if self.cfg.input_residual:
            input_dim += self.frame_size
        if self.cfg.input_bcloss:
            input_dim += self.frame_size
        if self.cfg.input_step:
            input_dim += 1
        if self.cfg.input_splitted_grad:
            input_dim += 2*self.channels
        if self.cfg.input_splitted_losses:
            input_dim += 2
        if self.cfg.input_forcings:
            input_dim += self.forcingschannels
        if self.cfg.input_ic:
            input_dim += self.icchannels 
        if self.cfg.input_bc:
            input_dim += self.bcchannels
        if self.cfg.input_idx:
            input_dim += 1
        return input_dim

    def get_input(self, theta, gradtheta, params, x, uhat_f, losses, compute_grad, step):
        """" Get the input of the model depending on the configuration

        Args:
            theta (torch.Tensor): the theta input (B, L, C)
            gradtheta (torch.Tensor): the gradtheta input (B, L, C)
            params (Tuple): the PDE params input tuple (params, forcings, ic, bc)
            x (torch.Tensor): the x grid 
            uhat_f (function): the uhat_f function to compute PDE residuals if needed
            losses (torch.Tensor): the PDE losses input
            compute_grad (function): how to compute the gradients of the physical loss
            step (torch.Tensor): the GD step 

        Returns:
            torch.Tensor: the input of the model
        """

        if self.cfg.nn.name=='conv1' or self.cfg.nn.name == 'conv1mlp':
            return self.get_input_conv(theta, gradtheta, params, x, uhat_f, losses, compute_grad, step)
        elif self.cfg.nn.name=='fno':
            return self.get_input_fno(theta, gradtheta, params, x, uhat_f, losses, compute_grad, step)
        else:
            return self.get_input_mlp(theta, gradtheta, params, x, uhat_f, losses, compute_grad, step)
        

    def get_input_mlp(self, thetas, gradthetas, datas, x, uhat_f, losses, compute_grad, step):
        """ 
        theta, gradtheta, datas : B, L, C
        out : B, I, C (cat  on I dim)
        """
        theta = thetas # [-1]
        gradtheta = gradthetas # [-1]
        bsize = theta.shape[0]
        channels = theta.shape[2]
        input = torch.empty((bsize, 0, channels), device=self.device)
        loss, lpde, lbc = losses
        if self.cfg.input_theta:
            input = torch.cat((input, theta), dim=1)
        if self.cfg.input_gradtheta:
            input = torch.cat((input, gradtheta), dim=1)
        if self.cfg.input_params:
            params = datas[0]
            input = torch.cat((input, params[:, :self.cfg.input_params].float()), dim=1)
        if self.cfg.input_loss:
            loss = loss.unsqueeze(-1).repeat(1, 1, channels)
            input = torch.cat((input, loss.float()), dim=1)
        if self.cfg.input_x:
            # x = einops.rearrange(x, 'B ... C -> B (...) C')
            x = x.unsqueeze(-1).repeat(1, 1, channels)
            input = torch.cat((input, x.float()), dim=1)
        if self.cfg.input_residual:
            residual, _ = self.pde(x.squeeze(-1), params, theta, uhat_f)
            residual = residual.repeat(1, 1, channels)
            input = torch.cat((input, residual.float()), dim=1)
        if self.cfg.input_bcloss:
            _, bc = self.pde(x.squeeze(-1), params, theta, uhat_f)
            bc = bc.unsqueeze(-1).repeat(1, 1, channels) # add channel dim B, X, C here is B, 1, 1
            input = torch.cat((input, bc.float()), dim=1)
        if self.cfg.input_splitted_grad:
            gpde = compute_grad(lpde, theta)
            gbc = compute_grad(lbc, theta)
            denom = torch.clamp((gpde+gbc).norm(2, 1, keepdim=True), min=1e-12).expand_as(gpde+gbc)
            gpden = torch.div(gpde, denom)
            gbcn = torch.div(gbc, denom)
            input = torch.cat((input, gpden, gbcn), dim=1)
        if self.cfg.input_splitted_losses:
            lpde = lpde.unsqueeze(-1).repeat(1, 1, channels)
            input = torch.cat((input, lpde.float()), dim=1)
            lbc = lbc.unsqueeze(-1).repeat(1, 1, channels)
            input = torch.cat((input, lbc.float()), dim=1)
        if self.cfg.input_forcings:
            forcings = datas[1]
            input = torch.cat((input, forcings), dim=1)
        if self.cfg.input_ic:
            ic = datas[2]
            ic = einops.rearrange(ic, 'B ... C -> B (...) C') 
            input = torch.cat((input, ic), dim=1)
        if self.cfg.input_bc:
            bc = datas[3]
            bc = einops.rearrange(bc, 'B ... C -> B (...) C') 
            input = torch.cat((input, bc), dim=1)
        if self.cfg.input_theta_hist:
            input = torch.cat((input, thetas), dim=1)
        if self.cfg.input_gradtheta_hist:
            input = torch.cat((input, gradthetas), dim=1)
        if self.cfg.input_step:
            step = torch.tensor([step]).unsqueeze(-1).repeat(bsize, 1, channels).to(self.device)
            input = torch.cat((input, step.float()), dim=1)
        if self.cfg.input_signlog:
            input = torch.cat((input, torch.sign(gradtheta)), dim=1)
            input = torch.cat((input, torch.log(torch.abs(gradtheta))), dim=1)
        return input
    

    def get_input_conv(self, theta, gradtheta, datas, x, uhat_f, losses, compute_grad, step):
        """ 
        theta, gradtheta, params : B, L, C
        out = B, C, L (for convd torch, cat on L dim)
        """
        bsize = theta.shape[0]
        length = theta.shape[1]
        channels = theta.shape[2]
        input = torch.empty((bsize, 0, length), device=self.device)
        loss, lpde, lbc = losses

        if self.cfg.input_theta:
            input = torch.cat((input, theta.transpose(1, 2)), dim=1)
        if self.cfg.input_gradtheta:
            input = torch.cat((input, gradtheta.transpose(1, 2)), dim=1)
        if self.cfg.input_params:
            params = datas[0]
            input = torch.cat((input, params[:, :self.cfg.input_params].repeat(1, 1, length).float()), dim=1)
        if self.cfg.input_step:
            step = torch.tensor([step]).unsqueeze(-1).repeat(bsize, channels, length).to(self.device)
            input = torch.cat((input, step.float()), dim=1)
        if self.cfg.input_loss:
            raise NotImplementedError(f'input loss not implemented for conv-like layers')
            loss = loss.unsqueeze(-1).repeat(1, 1, channels)
            input = torch.cat((input, loss.float()), dim=1)
        if self.cfg.input_x:
            raise NotImplementedError(f'input x not implemented for conv-like layers')
            x = x.unsqueeze(-1).repeat(1, 1, channels)
            input = torch.cat((input, x.float()), dim=1)
        if self.cfg.input_residual:
            raise NotImplementedError(f'input residual not implemented for conv-like layers')
            residual, _ = self.pde(x.squeeze(-1), params, theta, uhat_f)
            residual = residual.repeat(1, 1, channels)
            input = torch.cat((input, residual.float()), dim=1)
        if self.cfg.input_bcloss:
            raise NotImplementedError(f'input bc not implemented for conv-like layers')
            _, bc = self.pde(x.squeeze(-1), params, theta, uhat_f)
            bc = bc.unsqueeze(-1).repeat(1, 1, channels) # add channel dim B, X, C here is B, 1, 1
            input = torch.cat((input, bc.float()), dim=1)
        if self.cfg.input_splitted_grad:
            gpde = compute_grad(lpde, theta)
            gbc = compute_grad(lbc, theta) # torch.nn.functional.normalize(gradients, p=2, dim=1)
            input = torch.cat((input, gpde.transpose(1, 2), gbc.transpose(1, 2)), dim=1)
        if self.cfg.input_splitted_losses:
            raise NotImplementedError(f'input splitted losses not implemented for conv-like layers')
            lpde = lpde.unsqueeze(-1).repeat(1, 1, channels)
            input = torch.cat((input, lpde.float()), dim=1)
            lbc = lbc.unsqueeze(-1).repeat(1, 1, channels)
            input = torch.cat((input, lbc.float()), dim=1)
        if self.cfg.input_forcings:
            forcings = datas[1]
            input = torch.cat((input, forcings.repeat(1, 1, length).float()), dim=1)
        if self.cfg.input_ic:
            ic = datas[2]
            input = torch.cat((input, ic.repeat(1, 1, length).float()), dim=1)
        if self.cfg.input_bc:
            bc = datas[3]
            input = torch.cat((input, bc.repeat(1, 1, length).float()), dim=1)

        return input

    def get_input_fno(self, theta, gradtheta, datas, x, uhat_f, losses, compute_grad, step):
        """ 
        theta, gradtheta, params : B, N, C
        out = B, N, I (cat on last dim)
        """
        
        (x, x_in, x_ic, x_bc) = x # for training_nd 
        bsize = theta.shape[0]
        length = tuple(theta.shape[1:-1])
        channels = theta.shape[-1]
        input = torch.empty((bsize,) + length + (0,), device=self.device)
        loss, lpde, lbc = losses

        if self.cfg.input_theta:
            input = torch.cat((input, theta), dim=-1)

        if self.cfg.input_gradtheta:
            input = torch.cat((input, gradtheta), dim=-1)

        if self.cfg.input_params:
            params = datas[0]
            if len(params.shape) == 3:
                params = params[:, :self.paramschannels].reshape((bsize, ) + len(length) * (1, ) + (self.paramschannels, ))
                input = torch.cat((input, params.repeat((1, ) + length + (1, )).float()), dim=-1)
            else:
                paramsproj = uhat_f.projection(einops.rearrange(params, 'B ... D -> B (...) D'), einops.rearrange(x_in, 'B ... D -> B (...) D'))
                proj = torch.cat((torch.zeros((bsize, 1, 1), device=self.device), paramsproj), dim=1).reshape((bsize, ) + length + (channels,))
                input = torch.cat((input, proj), dim=-1)

        if self.cfg.input_step:
            step = torch.tensor([step]).unsqueeze(-1).repeat((bsize,)+ length + (channels, )).to(self.device)
            input = torch.cat((input, step.float()), dim=-1)

        if self.cfg.input_loss:
            raise NotImplementedError(f'input loss not implemented for fno-like layers')
            loss = loss.unsqueeze(-1).repeat(1, length, 1)
            input = torch.cat((input, loss.float()), dim=-1)

        if self.cfg.input_x:
            raise NotImplementedError(f'input x not implemented for fno-like layers')
            x_mod = x.unsqueeze(-1).repeat(1, 1, length)
            x_mod = x_mod.transpose(1,2)
            input = torch.cat((input, x_mod.float()), dim=-1)

        if self.cfg.input_residual:
            raise NotImplementedError(f'input residual not implemented for fno-like layers')
            residual, _ = self.pde(x.squeeze(-1), params, theta, uhat_f)
            residual = residual.repeat(1, length, 1)
            input = torch.cat((input, residual.float()), dim=-1)

        if self.cfg.input_bcloss:
            raise NotImplementedError(f'input bc not implemented for fno-like layers')
            _, bc = self.pde(x.squeeze(-1), params, theta, uhat_f)
            bc = bc.unsqueeze(-1).repeat(1, length, 1) # add channel dim B, X, C here is B, 1, 1
            input = torch.cat((input, bc.float()), dim=-1)

        if self.cfg.input_splitted_grad:
            raise NotImplementedError(f'input splitted_grad implemented for fno-like layers')
            gpde = compute_grad(lpde, theta)
            gbc = compute_grad(lbc, theta)
            input = torch.cat((input, gpde, gbc), dim=-1)

        if self.cfg.input_splitted_losses:
            raise NotImplementedError(f'input splitted losses not implemented for fno-like layers')
            lpde = lpde.unsqueeze(-1).repeat(1, length, 1)
            input = torch.cat((input, lpde.float()), dim=-1)
            lbc = lbc.unsqueeze(-1).repeat(1, length, 1)
            input = torch.cat((input, lbc.float()), dim=-1)

        if self.cfg.input_forcings:
            forcings = datas[1]
            forcings_proj = uhat_f.projection(einops.rearrange(forcings, 'B ... C -> B (...) C'), einops.rearrange(x, 'B ... D -> B (...) D'))
            if len(forcings.shape)>3:
                forcings_proj = torch.cat((torch.zeros((bsize, 1, 1), device=self.device), forcings_proj), dim=1).reshape((bsize, ) + length + (channels,))

            input = torch.cat((input, forcings_proj), dim=-1)
            
        if self.cfg.input_ic:
            ic = datas[2]

            proj = uhat_f.projection(einops.rearrange(ic, 'B ... C -> B (...) C'), einops.rearrange(x_ic, 'B ... D -> B (...) D'))
            proj = torch.cat((torch.zeros((bsize, 1, 1), device=self.device), proj), dim=1).reshape((bsize, ) + length + (channels,)) # cat bias + reshape
            input = torch.cat((input, proj), dim=-1)

        if self.cfg.input_bc:
            bc = datas[3]
            bc = bc.transpose(1,2)
            input = torch.cat((input, bc.repeat((1,) + length +(1,)).float()), dim=-1)

        if self.cfg.input_idx:
            idx = torch.arange(0, 1, 1/length[0], device=self.device).unsqueeze(0).unsqueeze(-1).repeat(bsize, 1, 1)
            input = torch.cat((input, idx), dim=-1)

        return input

    def count_parameters(self):
        """ Count the number of parameters in the model """

        pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params
        
        