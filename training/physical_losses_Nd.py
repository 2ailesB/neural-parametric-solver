import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops

from components.base_layer.init import init_layers
from utils.device import get_device
from utils.gradients import gradients
from models.abstract import Model


class Physical_Loss():
    def __init__(self, name):
        self.equation = name

        self.loss = self.init_physical_loss(self.equation)
        # self.gradfunc = self.init_gradient_func(cfg.gradfunc) # plus tard, on implémente d'abord avec le u.

    def init_physical_loss(self, equation): 
        if equation=='helmholtz' or equation=='helmholtz-hf':
            return self.helmholtz_loss
        elif equation=='poisson':
            return self.poisson_loss
        elif equation=='forcingmspoisson':
            return self.poisson_loss_f
        elif equation=='darcy':
            return self.darcy_loss
        elif equation=='advection' or equation=='advections':
            return self.advection_loss
        elif equation=='1dnlrd' or equation=='1dnlrdics':
            return  self.nlrd_loss
        elif equation=='heat2d':
            return self.heat2d_loss
        else:
            raise NotImplementedError(f'PDE loss {equation} not implemented')

    def set_reconstructionf(self, u):
        self.u = u

    def helmholtz_loss(self, x, theta, params, lbd):
        """ 
        in PINNs physical losses, batch mus be the input that was used for computing u
        => this because we need to be capable of computing gradient with autograd
        """
        # u is B, X, C
        # batch is B, X, D+P, C

        params, forcings, ic, bc = params # B, 1, 1 / 0 / B, X, 1 / B, T, 1 (X=256, T=25)
        (x, x_in, x_ic, x_bc) = x # B, X, T, D
        bsize, Ns, dim = x.shape[0], x.shape[1:-1], x.shape[1]
        
        u_b, g_b = bc[:, 0], bc[:, 1] # B, P
        omega, T = params[:, 0], params[:, 1] # B, P
        # x = x[:, :, 0]
        u = self.u.compute_u(x_in, theta) # [:, :, 0] # B, X, C
        u_x = self.u.compute_uderivativex(x_in, theta) # [:, :, 0] # B, X, C
        u_xx = self.u.compute_uderivativex2(x_in, theta) # [:, :, 0] # B, X, C
        # u_xx = self.gradfunc(u_x, x) # [:, :, 0]

        wuhat = torch.einsum('bn, bxn -> bxn', omega**2, u)
        F_g = u_xx + wuhat # B, Xi, C
        dx = (x[:, 1]-x[:, 0]) # B 1
        loss_in = (F_g ** 2).sum(dim=1) * dx / T

        u_bc = self.u.compute_u(x_bc, theta) # [:, :, 0] # B, X, C
        ux_bc = self.u.compute_uderivativex(x_bc, theta) # [:, :, 0] # B, X, C
        loss_ext = ((u_bc[:, 0] - u_b) ** 2) / 2 # initial condition on u
        loss_d = ((ux_bc[:, 0] - g_b) ** 2) / 2

        return (loss_in + lbd * (loss_ext + loss_d)), loss_in, loss_ext+loss_d

    def poisson_loss(self, x, theta, params, lbd):
        
        params, forcings, ic, bc = params # B, 1, 1 / 0 / B, X, 1 / B, T, 1 (X=256, T=25)
        (x, x_in, x_ic, x_bc) = x # B, X, T, D
        bsize, Ns, dim = x.shape[0], x.shape[1:-1], x.shape[1]
        u_b, g_b = bc[:, 0], bc[:, 1]
        rhs, T = forcings[:, 0], params[:, 0]  

        u_xx = self.u.compute_uderivativex(x_in, theta)
        F_g = u_xx[:, 1:] - rhs.unsqueeze(1).repeat(1, u_xx.shape[1] - 1, 1) # B, X, C + remove 1 in spatial dim for BC

        # Riemann approx
        dx = (x[:, 1]-x[:, 0]).unsqueeze(-1)
        loss_in = (F_g ** 2).sum(dim=1) * dx / T

        ubc = self.u.compute_u(x_bc, theta)
        uxbc = self.u.compute_uderivativex(x_bc, theta)
        loss_ext = ((ubc[:, 0] - u_b) ** 2) / 2 # initial condition on u
        loss_d = ((uxbc[:, 0] - g_b) ** 2) / 2 # initial condition on du/dx

        return (loss_in + lbd * (loss_ext + loss_d)), loss_in, loss_ext+loss_d

    def poisson_loss_f(self,  x, theta, params, lbd):
        params, forcings, ic, bc = params # B, 1, 1 / 0 / B, X, 1 / B, T, 1 (X=256, T=25)
        (x, x_in, x_ic, x_bc) = x # B, X, T, D
        bsize, Ns, dim = x.shape[0], x.shape[1:-1], x.shape[1]
        u_b, g_b = bc[:, 0], bc[:, 1]
        ais, T = params[:, :-1], params[:, -1]

        u_xx = self.u.compute_uderivativex2(x_in, theta)

        F_g = u_xx - forcings[:, 1:] # B, X, C et B, X, C

        # Riemann approx 
        dx = (x[:, 1]-x[:, 0])
        loss_in = (F_g ** 2).sum(dim=1) * dx / T

        ubc = self.u.compute_u(x_bc, theta)
        uxbc = self.u.compute_uderivativex(x_bc, theta)
        loss_ext = ((ubc[:, 0] - u_b) ** 2) / 2 # initial condition on u
        loss_d = ((uxbc[:, 0] - g_b) ** 2) / 2 # initial condition on du/dx

        return (loss_in + lbd * (loss_ext + loss_d)), loss_in, loss_ext+loss_d


    def darcy_loss(self, x, theta, params, lbd):
        params, forcings, ic, bc = params # B, 1, 1 / 0 / B, X, 1 / B, T, 1 (X=256, T=25)
        (x, x_in, x_ic, x_bc) = x # B, X, T, D
        bsize, Ns, dim = x.shape[0], x.shape[1:-1], x.shape[1]

        a = einops.rearrange(params, 'B X Y C -> B (X Y) C')
        forcings = einops.rearrange(forcings, 'B X Y C -> B (X Y) C')
        x_in = einops.rearrange(x_in, 'B X Y C -> B (X Y) C')
        theta = einops.rearrange(theta, 'B ... C -> B (...) C')[:, 1:]

        Ns_in = x_in.shape[1]

        x_in = x_in.detach().clone()
        x_in.requires_grad_(True)
        u_x, u_y = self.u.compute_uderivativex(x_in, theta, True) # B, XT
        dx = gradients(a*u_x.unsqueeze(-1), x_in)[:, :, 0] # B, XT, 2
        dy = gradients(a*u_y.unsqueeze(-1), x_in)[:, :, 1] # B, XT, 2
        x_in.requires_grad_(False)
        Du = dx + dy
        lpde = Du + forcings.squeeze(-1) # B, XT

        x_bc = einops.rearrange(x_bc, 'B X Y C -> B (X Y) C')
        Ns_bc = x_bc.shape[1]
        uhat_bc = self.u.compute_u(x_bc, theta) # B, X, C
        bc = torch.zeros(uhat_bc.shape, device=uhat_bc.get_device())
        lbc = uhat_bc - bc

        loss_in = (lpde.unsqueeze(-1) ** 2).sum(dim=1) / Ns_in
        loss_bc = (lbc ** 2).sum(1) / Ns_bc # TODO essayer avec bc = data

        return (loss_in + lbd * loss_bc), loss_in, loss_bc

    def advection_loss(self, x, theta, params, lbd):
        """
        params: B, 1, 1 / 0 / B, X, 1 / B, T, 1 (X=256, T=25)
        x: B, X, T, 2
        theta: B, NX, NT, 1
        """

        beta, forcings, ic, bc = params # B, 1, 1 / 0 / B, X, 1 / B, T, 1 (X=256, T=25)
        (x, x_in, x_ic, x_bc) = x # B, X, T, D
        bsize, Ns, dim = x.shape[0], x.shape[1:-1], x.shape[1]

        u_x, u_t = self.u.compute_uderivativex(einops.rearrange(x_in, 'B ... C -> B (...) C'), einops.rearrange(theta, 'B ... C -> B (...) C')[:, 1:]) # B, XT, P, 1
        F_g = u_t + beta.squeeze(-1) * u_x # B, X
        Ns = F_g.shape[1]
        # Riemann approx
        loss_in = (F_g.unsqueeze(-1) ** 2).sum(dim=1) / Ns # B, 1
        uhat_ic = self.u.compute_u(einops.rearrange(x_ic, 'B ... C -> B (...) C'), einops.rearrange(theta, 'B ... C -> B (...) C')[:, 1:])
        Ns = ic.shape[1]
        loss_bc = ((uhat_ic - ic) ** 2).sum(1) / Ns # b, 1

        return (loss_in + lbd * loss_bc), loss_in, loss_bc
    
    def nlrd_loss(self, x, theta, params, lbd):
        p, forcings, ic, bc = params # B, 1, 1 / 0 / B, X, 1 / B, T, 1 (X=256, T=25)
        (x, x_in, x_ic, x_bc) = x # B, X, T, D

        nu, rho = p[:, 0], p[:, 1] # b, 1, 1

        u = self.u.compute_u(einops.rearrange(x, 'B ... C -> B (...) C'), einops.rearrange(theta, 'B ... -> B (...)').unsqueeze(-1)[:, 1:]).squeeze(-1) # B, X
        u_x, u_t = self.u.compute_uderivativex(einops.rearrange(x, 'B ... C -> B (...) C'), einops.rearrange(theta, 'B ... -> B (...)').unsqueeze(-1)[:, 1:]) # B, X
        u_xx, u_xt, u_tx, u_tt = self.u.compute_uderivativex2(einops.rearrange(x, 'B ... C -> B (...) C'), einops.rearrange(theta, 'B ... -> B (...)').unsqueeze(-1)[:, 1:]) # B, X

        F_g = u_t + nu * u_xx - rho * u * (1 - u) # B, XT
        Ns = F_g.shape[1]

        # Riemann approx
        loss_in = (F_g.unsqueeze(-1) ** 2).sum(dim=1) / Ns

        uhat_ic = self.u.compute_u(einops.rearrange(x_ic, 'B ... C -> B (...) C'), einops.rearrange(theta, 'B ... -> B (...)').unsqueeze(-1)[:, 1:]) # B, X, C
        Ns = uhat_ic.shape[1]
        loss_ic = (uhat_ic - ic) 
        loss_ic = (loss_ic ** 2).sum(1) / Ns

        return (loss_in + lbd * loss_ic), loss_in, loss_ic
    
    def heat2d_loss(self, x, theta, params, lbd):
        p, forcings, ic, bc = params # B, 1, 1 / 0 / B, X, 1 / B, T, 1 (X=256, T=25)
        (x, x_in, x_ic, x_bc) = x # B, X, T, D
        bsize, Ns, dim = x.shape[0], x.shape[1:-1], x.shape[1]
        Ns_in = x_in.shape[1]*x_in.shape[2]

        # PDE 
        d1 = self.u.compute_uderivativex(einops.rearrange(x_in, 'B ... C -> B (...) C'), einops.rearrange(theta, 'B ... -> B (...)').unsqueeze(-1)[:, 1:])
        u_t = d1[2]
        d2 = self.u.compute_uderivativex2(einops.rearrange(x_in, 'B ... C -> B (...) C'), einops.rearrange(theta, 'B ... -> B (...)').unsqueeze(-1)[:, 1:])
        u_xx, u_yy = d2[0][0], d2[1][1]

        c = p[:, 0, :]
        F_g = u_t - c * (u_xx + u_yy)
        loss_in = (F_g ** 2).sum(dim=1) / Ns_in 

        # IC 
        uhat_ic = self.u.compute_u(einops.rearrange(x_ic, 'B ... C -> B (...) C'), einops.rearrange(theta, 'B ... -> B (...)').unsqueeze(-1)[:, 1:]) # B, X, C
        Ns = uhat_ic.shape[1]
        icu = einops.rearrange(ic, 'B ... C -> B (...) C')
        lossic = ((uhat_ic - icu)** 2).sum(1) / Ns

        loss_bc = lossic + lossbc

        return (loss_in.unsqueeze(-1) + lbd * loss_bc), loss_in.unsqueeze(-1), loss_bc
    
