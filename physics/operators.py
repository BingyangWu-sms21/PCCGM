'''
Implementation of PDE operators to enfore physical consistency
Does not solve the system, only computes residuals
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
import torch.nn as nn
import numpy as np
from data_generation.rve.fem_solver import PeriodicBCRVE2D
from utils import instantiate_from_config

class DarcyFlow(nn.Module):
    def __init__(self, dx, eps, mu_p, sigma_p, mu_k, sigma_k):
        super().__init__()
        # returns all components of Darcy flow residual

        self.eps = eps # for FD approximation of d/du[du/dx]
        self.dx = dx
        self.mu_p = mu_p
        self.sigma_p = sigma_p
        self.mu_k = mu_k
        self.sigma_k = sigma_k

    def F(self, u):
        # source term on domain [0,1]x[0,1]
        # u is used as input only to determine sizes and devices -> (b, n, n)

        f = torch.zeros_like(u)

        xv = np.linspace(0, 1, f.shape[-1])
        for i in range(len(xv)):
            for j in range(len(xv)):
                x = xv[i]
                y = xv[j]

                if (np.abs(x-0.0625)<=0.0625) and (np.abs(y-0.0625)<=0.0625):
                    f[:, i, j] = 10
                elif (np.abs(x-1+0.0625)<=0.0625) and (np.abs(y-1+0.0625)<=0.0625):
                    f[:, i, j] = -10

        return f

    def int(self, p):
        # integral of p over domain [0, 1]x[0, 1]
        # trapezoid rule in 2d
        # inputs: p -> (b, n, n)
        n = p.shape[-1]
        s1 = torch.sum(p[:, 1:-1, 1:-1], dim=(1, -1))
        s2 = torch.sum(p[:, 0, :] + p[:, -1, :], dim=-1)
        s3 = torch.sum(p[:, :, 0] + p[:, :, -1], dim=-1)

        integral = self.dx**2/4*(p[:, 0, 0] + p[:, 0, -1] + p[:, -1, 0] + p[:, -1, -1] + 4*s1 + 2*s2 + 2*s3)
        return integral.repeat(1, n*n).reshape(-1, n, n)

    def dudy(self, u):
        # computes du/dx at each point in the domain using
        #  2nd order finite differences
        # inputs: u -> (b, n, n)

        # interior points
        dudx = (torch.roll(u, -1, 2) - torch.roll(u, 1, 2))/(2*self.dx)

        # x1 = 0 forward FD
        dudx[:, :, 0] = (-3*u[:, :, 0] + 4*u[:, :, 1] - u[:, :, 2])/(2*self.dx)

        # x1 = L backward FD
        dudx[:, :, -1] = (3*u[:, :, -1] - 4*u[:, :, -2] + u[:, :, -3])/(2*self.dx)

        return dudx

    def dudx(self, u):
        # computes du/dy at each point in the domain using
        #  2nd order finite differences
        # inputs: u -> (b, n, n)

        # interior points
        dudy = (torch.roll(u, -1, 1) - torch.roll(u, 1, 1))/(2*self.dx)

        # x1 = 0 forward FD
        dudy[:, 0, :] = (-3*u[:, 0, :] + 4*u[:, 1, :] - u[:, 2, :])/(2*self.dx)

        # x1 = L backward FD
        dudy[:, -1, :] = (3*u[:, -1, :] - 4*u[:, -2, :] + u[:, -3, :])/(2*self.dx)

        return dudy

    def d2udy2(self, u):
        # Second order FD approximation of second derivative d^2u/dx^2
        # inputs: u -> (b, n, n)
        d2udx2 = (torch.roll(u, -1, 2) - 2*u + torch.roll(u, 1, 2))/(self.dx**2)

        d2udx2[:, :, 0] = (2*u[:, :, 0] - 5*u[:, :, 1] + 4*u[:, :, 2] - u[:, :, 3])/(self.dx**2)
        d2udx2[:, :, -1] = (2*u[:, :, -1] - 5*u[:, :, -2] + 4*u[:, :, -3] - u[:, :, -4])/(self.dx**2)

        return d2udx2

    def d2udx2(self, u):
        # Second order FD approximation of second derivative d^2u/dy^2
        # inputs: u -> (b, n, n)
        d2udy2 = (torch.roll(u, -1, 1) - 2*u + torch.roll(u, 1, 1))/(self.dx**2)

        d2udy2[:, 0, :] = (2*u[:, 0, :] - 5*u[:, 1, :] + 4*u[:, 2, :] - u[:, 3, :])/(self.dx**2)
        d2udy2[:, -1, :] = (2*u[:, -1, :] - 5*u[:, -2, :] + 4*u[:, -3, :] - u[:, -4, :])/(self.dx**2)

        return d2udy2

    def get_inds(self, method, u):
        # seperates the rows or columns by index for computing d/du[grad_u] efficiently

        if method == 'dudx':
            inds = torch.zeros(u.shape[1], device=u.device, dtype=torch.uint8)
            for i in range(len(inds)):
                if (i%3) == 0:
                    inds[i] = 0
                elif ((i-1)%3) == 0:
                    inds[i] = 1
                elif ((i-2)%3) == 0:
                    inds[i] = 2
        elif method == 'dudy':
            inds = torch.zeros(u.shape[2], device=u.device, dtype=torch.uint8)
            for i in range(len(inds)):
                if (i%3) == 0:
                    inds[i] = 0
                elif ((i-1)%3) == 0:
                    inds[i] = 1
                elif ((i-2)%3) == 0:
                    inds[i] = 2
        elif method == 'd2udx2':
            inds = torch.zeros(u.shape[1], device=u.device, dtype=torch.uint8)
            for i in range(len(inds)):
                if (i%4) == 0:
                    inds[i] = 0
                elif ((i-1)%4) == 0:
                    inds[i] = 1
                elif ((i-2)%4) == 0:
                    inds[i] = 2
                elif ((i-3)%4) == 0:
                    inds[i] = 3
        elif method == 'd2udy2':
            inds = torch.zeros(u.shape[2], device=u.device, dtype=torch.uint8)
            for i in range(len(inds)):
                if (i%4) == 0:
                    inds[i] = 0
                elif ((i-1)%4) == 0:
                    inds[i] = 1
                elif ((i-2)%4) == 0:
                    inds[i] = 2
                elif ((i-3)%4) == 0:
                    inds[i] = 3

        return inds

    def d_du(self, method, grad_u, u):
        # computes finite differences of d/du[d^2u/dx^2] among others
        # inputs: method -> (str) ['d2udx2', 'd2udy2', 'dudx', dudy']
        #         grad_u -> (b, n, n) gradient du/dx or other from method
        #         u ->      (b, n, n) u
        # outputs: ddp -> (b, n, n) d/du[grad_u]

        ddu = torch.zeros_like(grad_u)
        inds = self.get_inds(method, u)
        n = torch.unique(inds)

        if method == 'dudx':
            # compute finite difference in row or column-wise batches
            for i in range(len(n)):
                u_plus = u + 0.
                u_plus[:, inds==n[i], :] = u[:, inds==n[i], :] + self.eps
                grad_u_plus = self.dudx(u_plus)
                ddu_group = (grad_u_plus - grad_u)/self.eps
                ddu[:, inds==n[i], :] = ddu_group[:, inds==n[i], :]

        elif method == 'dudy':
            # compute finite difference in row or column-wise batches
            for i in range(len(n)):
                u_plus = u + 0.
                u_plus[:, :, inds==n[i]] = u[:, :, inds==n[i]] + self.eps
                grad_u_plus = self.dudy(u_plus)
                ddu_group = (grad_u_plus - grad_u)/self.eps
                ddu[:, :, inds==n[i]] = ddu_group[:, :, inds==n[i]]

        elif method == 'd2udx2':
            for i in range(len(n)):
                u_plus = u + 0.
                u_plus[:, inds==n[i], :] = u[:, inds==n[i], :] + self.eps
                grad_u_plus = self.d2udx2(u_plus)
                ddu_group = (grad_u_plus - grad_u)/self.eps
                ddu[:, inds==n[i], :] = ddu_group[:, inds==n[i], :]

        elif method == 'd2udy2':
            for i in range(len(n)):
                u_plus = u + 0.
                u_plus[:, :, inds==n[i]] = u[:, :, inds==n[i]] + self.eps
                grad_u_plus = self.d2udy2(u_plus)
                ddu_group = (grad_u_plus - grad_u)/self.eps
                ddu[:, :, inds==n[i]] = ddu_group[:, :, inds==n[i]]

        return ddu


    def drdp(self, x):
        # gradient of residual w.r.t. p : dr/dp
        # Computing d/dp[dp/dx] and other quantities using finite differences
        p = x[:, 0, :, :]*self.sigma_p + self.mu_p
        k = x[:, 1, :, :]*self.sigma_k + self.mu_k

        r, d2pdx2, d2pdy2, dkdx, dpdx, dkdy, dpdy = self.r_diff(p, k)

        d2pdx2_dp = self.d_du('d2udx2', d2pdx2, p)
        d2pdy2_dp = self.d_du('d2udy2', d2pdy2, p)
        dpdx_dp = self.d_du('dudx', dpdx, p)
        dpdy_dp = self.d_du('dudy', dpdy, p)

        dr_dp = 2*r*(k*d2pdx2_dp+dkdx*dpdx_dp+k*d2pdy2_dp+dkdy*dpdy_dp)
        return dr_dp

    def r_diff(self, p, k):
        # compute diffusion operator residual and components
        # inputs: p (pressure) -> (b, n, n)
        #         k (permeability) -> (b, n, n)
        # TEST

        f = self.F(p)
        d2pdx2 = self.d2udx2(p)
        d2pdy2 = self.d2udy2(p)
        dkdx = self.dudx(k)
        dpdx = self.dudx(p)
        dkdy = self.dudy(k)
        dpdy = self.dudy(p)
        residual = f+k*d2pdx2+dkdx*dpdx+k*d2pdy2+dkdy*dpdy
        return residual, d2pdx2, d2pdy2, dkdx, dpdx, dkdy, dpdy

    def forward(self, x):
        # inputs are (b, 4, n, n) where channels are (pressure, u1, u2, K)
        # returns non normalized residual
        residual = torch.zeros_like(x)
        '''
        integral = self.int((x[:, 0]*self.sigma) + self.mu)
        residual[:, 0, :, :] = integral
        '''
        residual[:, 0, :, :] = self.drdp(x)

        return residual


class ViscousBurgers(nn.Module):
    def __init__(self, nu, dx, dt, mu, sigma):
        super().__init__()
        # forward returns residual at every point in the domain (second order approx.)

        self.mu = mu  # mean of dataset
        self.sigma = sigma # std of dataset

        self.nu = nu
        self.dx = dx
        self.dt = dt

    def dudt(self, u):
        # Second order FD approximation of first derivative.
        # Endpoints treated with forward/backward FD

        # interior points
        dudx = (torch.roll(u, -1, 3) - torch.roll(u, 1, 3))/(2*self.dt)
        dudx[:, :, :, 0] = 0.0
        dudx[:, :, :, -1] = 0.0

        # t = 0 forward FD
        dudx[:, :, :, 0] = (-3*u[:, :, :, 0] + 4*u[:, :, :, 1] - u[:, :, :, 2])/(2*self.dt)

        # t = T backward FD
        dudx[:, :, :, -1] = (3*u[:, :, :, -1] - 4*u[:, :, :, -2] + u[:, :, :, -3])/(2*self.dt)

        return dudx


    def dudx(self, u):
        # periodic boundary conditions. Second order centered FD approximation of first derivative
        return (torch.roll(u, -1, 2) - torch.roll(u, 1, 2))/(2*self.dx)

    def d2udx2(self, u):
        # periodic boundardy conditions. Second order centered FD approximation of second derivative
        return (torch.roll(u, -1, 2) - 2*u + torch.roll(u, 1, 2))/(self.dx**2)

    def forward(self, u):
        # data u is of shape (b, 1, n, n) corresponding to (batch, u channel, x coord, t coord)
        return self.sigma*self.dudt(u) + (self.sigma*u+self.mu)*self.sigma*self.dudx(u) - self.sigma*self.nu*self.d2udx2(u)


class RVE(nn.Module):
    def __init__(self, properties, corner, n_cells, fourier=False):
        super().__init__()
        if len(properties) != 4:
            raise ValueError("Properties must be a list of length 4")
        Em, nu_m, Ei, nu_i = properties
        self.lmbda_m = Em * nu_m / ((1. + nu_m) * (1. - 2. * nu_m))
        self.mu_m = Em / (2. * (1. + nu_m))
        self.lmbda_i = Ei * nu_i / ((1. + nu_i) * (1. - 2. * nu_i))
        self.mu_i = Ei / (2. * (1. + nu_i))
        global_strain_list = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
        self.fourier = fourier
        self.solver = PeriodicBCRVE2D(
            corner=corner, n_cells=n_cells, global_strain_list=global_strain_list)

    def forward(self, x, moduli_gt):
        r"""Compute the residual of the RVE problem

        Args:
            x (torch.Tensor): The RVE field. Shape (b, 1, *n_cells) if fourier=False,
                (b, 2, *n_cells) if fourier=True
            moduli_gt (torch.Tensor): The ground truth moduli. Shape (b, 9)

        Returns:
            torch.Tensor: The residual. Shape (b, 9)
        """
        # reshape x to (b, *n_cells)
        if self.fourier:
            # compute the inverse Fourier transform
            x_complex = x[:, 0] + 1j * x[:, 1]
            x = torch.fft.ifft(x_complex).real
        else:
            x = x.squeeze(1)
        lmbda = (1. - x) * self.lmbda_m + x * self.lmbda_i
        mu = (1. - x) * self.mu_m + x * self.mu_i
        lmbda = lmbda.cpu().double()
        mu = mu.cpu().double()
        # _, moduli = self.solver.run_and_process(lmbda.cpu().double(), mu.cpu().double())
        moduli = torch.zeros(moduli_gt.shape[0], 3, 3, device=moduli_gt.device, dtype=moduli_gt.dtype)
        # solve sample by sample
        for i in range(lmbda.shape[0]):
            try:
                _, moduli[i] = self.solver.run_and_process(lmbda[i].unsqueeze(0), mu[i].unsqueeze(0))
            except Exception as e:
                print("Error in RVE forward: ", e)
        moduli = moduli.reshape(-1, 9)
        error = moduli - moduli_gt
        return error


class RVESurrogate(torch.autograd.Function):
    r"""Compute the RVE problem using a surrogate model. The
    surrogate model is a pre-trained neural network that predicts the moduli
    and the gradient of the moduli with respect to the RVE field.
    """

    @staticmethod
    def forward(ctx, x, model):
        # Shape of x: [b, in_channels, *n_cells]
        fval, grad = model(x)  # Shape [b, out_dim], [b, out_dim*in_channels, *n_cells]
        ctx.save_for_backward(grad, x)
        return fval

    @staticmethod
    def backward(ctx, grad_output):
        grad, x = ctx.saved_tensors
        in_channels = x.shape[1]
        out_dim = grad_output.shape[1]
        grad = grad.view(-1, out_dim, in_channels, *x.shape[2:])  # Shape [b, out_dim, in_channels, *n_cells]
        grad_output = grad_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape [b, out_dim, 1, 1, 1]
        grad_input = grad * grad_output  # Shape [b, out_dim, in_channels, *n_cells]
        grad_input = grad_input.sum(dim=1)  # Shape [b, in_channels, *n_cells]
        return grad_input, None


class RVESurrogateResidual(nn.Module):
    r"""Compute the residual of the RVE problem using a surrogate model. The
    surrogate model is a pre-trained neural network that predicts the moduli
    and the gradient of the moduli with respect to the RVE field.
    """

    def __init__(self, model_config):
        super().__init__()
        self.model = instantiate_from_config(model_config)

    def forward(self, x, moduli_gt):
        r"""Compute the residual of the RVE problem

        Args:
            x (torch.Tensor): The RVE field. Shape (b, 1, *n_cells)
            moduli_gt (torch.Tensor): The ground truth moduli. Shape (b, 9)

        Returns:
            torch.Tensor: The residual. Shape (b, 9)
        """
        moduli_pred = RVESurrogate.apply(x, self.model)
        error = moduli_pred - moduli_gt
        return error


class Regularizer(nn.Module):
    def __init__(self, weight=None, fourier=False):
        super().__init__()
        if weight is None:
            weight = [0., 0., 0.]
        if len(weight) != 3:
            raise ValueError("Regularization weights must be a list of length 2")
        self.reg_weight = weight
        self.fourier = fourier

    def forward(self, x):
        r"""Compute the regularization term for samples x

        Args:
            x (torch.Tensor): The RVE field. Shape (b, 1, *n_cells)

        Returns:
            torch.Tensor: The regularization term. Shape (b)
        """
        # reshape x to (b, *n_cells)
        if self.fourier:
            # compute the inverse Fourier transform
            x_complex = x[:, 0] + 1j * x[:, 1]
            x = torch.fft.ifft(x_complex).real
        else:
            x = x.squeeze(1)
        # periodic TV (Total Variation) regularization
        periodic_tv = torch.sum(torch.abs(x - torch.roll(x, 1, 1)), dim=(1, 2)) + \
                      torch.sum(torch.abs(x - torch.roll(x, 1, 2)), dim=(1, 2))
        # constrain the values to be either 0 or 1
        binary_loss = torch.mean(torch.abs(x * (1. - x)), dim=(1, 2))
        # periodic contraint
        periodic_loss = torch.mean(torch.abs(x[:, 0, :] - x[:, -1, :]), dim=1) + \
                        torch.mean(torch.abs(x[:, :, 0] - x[:, :, -1]), dim=1)
        return self.reg_weight[0] * periodic_tv + self.reg_weight[1] * binary_loss + \
            self.reg_weight[2] * periodic_loss
