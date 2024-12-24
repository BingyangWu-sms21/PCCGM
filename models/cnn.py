"""CNN model for regression tasks. Input is a 2D image and output is a 1D vector."""
from typing import Dict, Tuple, List
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (CosineAnnealingLR, CosineAnnealingWarmRestarts,
                                      MultiStepLR, ExponentialLR)
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import sys
import pytorch_lightning as pl
import os
import os.path as osp
import argparse


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False, padding_mode: str = 'zeros',
        activation: str = 'gelu'
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        if activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            self.activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros', activation='gelu') -> None:
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels, padding_mode=padding_mode, activation=activation
                                    ), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros', activation='gelu') -> None:
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels, padding_mode=padding_mode, activation=activation),
            ResidualConvBlock(out_channels, out_channels, padding_mode=padding_mode, activation=activation),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class CNNRegressor(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 input_size: List[int],
                 out_dim: int,
                 n_feat: int = 64,
                 padding_mode: str = 'zeros',
                 activation: str = 'gelu',
                 lr: float = 1e-4,
                 optimizer_type: str = 'adamw',
                 scheduler_type: str = 'none',
                 milestones: List[int] = [30, 80],
                 gamma: float = 0.5,
                 dropout: float = 0.,
                 share_weights: bool = True,
                 grad: List[bool] = [False, False],
                 grad_weight: float = 0.1,
                 ignore_keys=[],
                 ckpt_path=None,
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        self.out_dim = out_dim
        self.n_feat = n_feat
        self.padding_mode = padding_mode
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.milestones = milestones
        self.gamma = gamma
        self.dropout = dropout
        self.grad = grad  # [train, val & test]
        self.grad_weight = grad_weight
        if activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # [B, in_channels, H, W] -> [B, n_feat, H, W]
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True, padding_mode=padding_mode,
                                           activation=activation)

        # [B, n_feat, H, W] -> [B, n_feat*2, H/2, W/2]
        self.down1 = UnetDown(n_feat, n_feat * 2, padding_mode=padding_mode, activation=activation)
        # [B, n_feat*2, H/2, W/2] -> [B, n_feat*4, H/4, W/4]
        self.down2 = UnetDown(n_feat * 2, n_feat * 4, padding_mode=padding_mode, activation=activation)
        # [B, n_feat*4, H/4, W/4] -> [B, n_feat*8, H/8, W/8]
        self.down3 = UnetDown(n_feat * 4, n_feat * 8, padding_mode=padding_mode, activation=activation)
        # [B, n_feat*8, H/8, W/8] -> [B, n_feat*16, H/16, W/16], only for label output
        self.down4_label = UnetDown(n_feat * 8, n_feat * 16, padding_mode=padding_mode, activation=activation)

        if not share_weights:
            self.down1_label = UnetDown(n_feat, n_feat * 2, padding_mode=padding_mode, activation=activation)
            self.down2_label = UnetDown(n_feat * 2, n_feat * 4, padding_mode=padding_mode, activation=activation)
            self.down3_label = UnetDown(n_feat * 4, n_feat * 8, padding_mode=padding_mode, activation=activation)

        fc_in_dim = n_feat * 16 * (input_size[0] // 16) * (input_size[1] // 16)
        self.fc_out = nn.Sequential(
            nn.Linear(fc_in_dim, n_feat * 16),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(n_feat * 16, n_feat * 4),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(n_feat * 4, out_dim),
        )

        # [B, n_feat*8, H/8, W/8] -> [B, n_feat*4, H/4, W/4]
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(n_feat * 8, n_feat * 4, 2, 2),
            ResidualConvBlock(n_feat * 4, n_feat * 4, padding_mode=padding_mode, activation=activation),
            ResidualConvBlock(n_feat * 4, n_feat * 4, padding_mode=padding_mode, activation=activation),
        )
        # [B, n_feat*4 + n_feat*4, H/4, W/4] -> [B, n_feat*2, H/2, W/2]
        self.up1 = UnetUp(n_feat * 4 + n_feat * 4, n_feat * 2, padding_mode=padding_mode, activation=activation)
        # [B, n_feat*2 + n_feat*2, H/2, W/2] -> [B, n_feat, H, W]
        self.up2 = UnetUp(n_feat * 2 + n_feat * 2, n_feat, padding_mode=padding_mode, activation=activation)
        # [B, n_feat + n_feat, H, W] -> [B, n_feat, H, W] -> [B, out_dim * in_channels, H, W]
        self.out = nn.Sequential(
            nn.Conv2d(n_feat + n_feat, n_feat, 3, 1, 1, padding_mode=padding_mode),
            nn.BatchNorm2d(n_feat),
            self.activation,
            nn.Conv2d(n_feat, out_dim * in_channels, 3, 1, 1, padding_mode=padding_mode),
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x = self.init_conv(x)
        # x = self.down1(x)
        # x = self.down2(x)
        # x = self.down3(x)
        # x = self.down4(x)

        # x = x.view(x.size(0), -1)
        # x = F.gelu(self.fc1(x))
        # x = F.gelu(self.fc2(x))
        # x = self.fc3(x)

        x0 = self.init_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        d1 = self.up0(x3)
        d2 = self.up1(d1, x2)
        d3 = self.up2(d2, x1)
        grad_out = self.out(torch.cat([d3, x0], 1))

        if self.share_weights:
            x4_label = self.down4_label(x3)
            value_out = x4_label.view(x4_label.size(0), -1)
            value_out = self.fc_out(value_out)
        else:
            x0_label = self.init_conv(x)
            x1_label = self.down1_label(x0_label)
            x2_label = self.down2_label(x1_label)
            x3_label = self.down3_label(x2_label)
            x4_label = self.down4_label(x3_label)
            value_out = x4_label.view(x4_label.size(0), -1)
            value_out = self.fc_out(value_out)

        return value_out, grad_out

    def init_from_ckpt(self, path, ignore_keys=list()):
        params = torch.load(path, map_location='cpu')
        if "state_dict" in list(params.keys()):
            params = params["state_dict"]
        keys = list(params.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del params[k]

        missing, unexpected = self.load_state_dict(params, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch['input']  # shape (B, C, H, W)
        use_grad = self.grad[0]
        if use_grad:
            x = x.clone().detach().requires_grad_(True)
            x.retain_grad()
        y = batch['label']
        y_hat, grad_hat = self.forward(x)  # shape (B, out_dim)
        # if use_grad:
        #     y_grad = batch['grad']  # shape (B, C * out_dim, H, W)
        #     y_hat_grad = torch.zeros_like(y_grad, requires_grad=False)
        #     for i in range(self.out_dim):
        #         y_hat[:, i].backward(torch.ones_like(y_hat[:, i]), retain_graph=True)
        #         y_hat_grad[:, i*self.in_channels:(i+1)*self.in_channels] = x.grad.clone()
        #         x.grad.zero_()
        #     print(y_grad.mean(), y_hat_grad.mean())

        loss = F.mse_loss(y_hat, y)
        if use_grad:
            # loss_grad = F.mse_loss(y_hat_grad, y_grad)
            y_grad = batch['grad']  # shape (B, C * out_dim, H, W)
            loss_grad = F.mse_loss(grad_hat, y_grad)
            self.log('train/loss_label', loss, prog_bar=True, logger=True,
                     on_step=True, on_epoch=True)
            self.log('train/loss_grad', loss_grad, prog_bar=True, logger=True,
                     on_step=True, on_epoch=True)
            loss = loss + loss_grad * self.grad_weight

        self.log('train/loss', loss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True,
                 on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        with torch.enable_grad():
            x = batch['input']
            use_grad = self.grad[1]
            if use_grad:
                x = x.clone().detach().requires_grad_(True)
                x.retain_grad()
            y = batch['label']
            y_hat, grad_hat = self.forward(x)  # shape (B, out_dim), (B, C * out_dim, H, W)
            # if use_grad:
            #     y_grad = batch['grad']  # shape (B, C * out_dim, H, W)
            #     y_hat_grad = torch.zeros_like(y_grad, requires_grad=False)
            #     for i in range(self.out_dim):
            #         y_hat[:, i].backward(torch.ones_like(y_hat[:, i]), retain_graph=True)
            #         y_hat_grad[:, i*self.in_channels:(i+1)*self.in_channels] = x.grad.clone()
            #         x.grad.zero_()
            loss = F.mse_loss(y_hat, y)
            if use_grad:
                # loss_grad = F.mse_loss(y_hat_grad, y_grad)
                y_grad = batch['grad']  # shape (B, C * out_dim, H, W)
                loss_grad = F.mse_loss(grad_hat, y_grad)
                self.log('val/loss_label', loss, prog_bar=True, logger=True,
                        on_step=True, on_epoch=True)
                self.log('val/loss_grad', loss_grad, prog_bar=True, logger=True,
                        on_step=True, on_epoch=True)
                loss = loss + loss_grad * self.grad_weight

            self.log('val/loss', loss, prog_bar=True, logger=True,
                     on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        with torch.enable_grad():
            x = batch['input']
            use_grad = self.grad[1]
            if use_grad:
                x = x.clone().detach().requires_grad_(True)
                x.retain_grad()
            y = batch['label']
            y_hat, grad_hat = self.forward(x)  # shape (B, out_dim)
            # if use_grad:
            #     y_grad = batch['grad']  # shape (B, C * out_dim, H, W)
            #     y_hat_grad = torch.zeros_like(y_grad, requires_grad=False)
            #     for i in range(self.out_dim):
            #         y_hat[:, i].backward(torch.ones_like(y_hat[:, i]), retain_graph=True)
            #         y_hat_grad[:, i*self.in_channels:(i+1)*self.in_channels] = x.grad.clone()
            #         x.grad.zero_()

            loss = F.mse_loss(y_hat, y)
            if use_grad:
                # loss_grad = F.mse_loss(y_hat_grad, y_grad)
                y_grad = batch['grad']  # shape (B, C * out_dim, H, W)
                loss_grad = F.mse_loss(grad_hat, y_grad)
                self.log('test/loss_label', loss, prog_bar=True, logger=True,
                        on_step=True, on_epoch=True)
                self.log('test/loss_grad', loss_grad, prog_bar=True, logger=True,
                        on_step=True, on_epoch=True)
                loss = loss + loss_grad * self.grad_weight

            self.log('test/loss', loss, prog_bar=True, logger=True,
                     on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # choose optimizer based on optimizer type
        params = list(self.parameters())
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.lr)
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # if scheduler_type is 'none', then no lr scheduler is used
        if self.scheduler_type == 'none':
            return {'optimizer': optimizer}

        # choose lr scheduler based on scheduler type
        if self.scheduler_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
        elif self.scheduler_type == 'cos-warm':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
        elif self.scheduler_type == 'mstep':
            scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        elif self.scheduler_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=self.gamma)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train/loss'}
