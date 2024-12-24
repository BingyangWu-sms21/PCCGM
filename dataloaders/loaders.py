'''
Pytorch lightning dataloaders
Author: Christian Jacobsen, University of Michigan 2023
'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pytorch_lightning as pl
import os
import os.path as osp
import argparse


class Darcy_Dataset(Dataset):
    def __init__(self, path):
        self.root = path

        # load the sample names
        sample_names = os.listdir(osp.join(path, "data"))
        self.P_names, self.U1_names, self.U2_names = self.seperate_img_names(sample_names)
        self.P_names.sort()
        self.U1_names.sort()
        self.U2_names.sort() # all files are stored as P_xxx.npy, U1_xxx.npy, U2_xxx.npy
        self.img_mean = np.array([0, 0.194094975, 0.115737872]) # P, U1, U2
        self.img_std = np.array([0.08232874, 0.27291843, 0.12989907])

        # load permeability fields
        self.perm_names = os.listdir(osp.join(path, "permeability"))
        self.perm_names.sort()
        self.perm_mean = 1.14906847
        self.perm_std = 7.81547992

        # load the parameter values
        self.param_names = os.listdir(osp.join(path, "params"))
        self.param_names.sort()
        self.param_mean = 1.248473
        self.param_std = 0.7208982

    def seperate_img_names(self, names):
        P, U1, U2 = [], [], []
        for name in names:
            if name[0] == "P":
                P.append(name)
            elif name[0:2] == "U1":
                U1.append(name)
            elif name[0:2] == "U2":
                U2.append(name)
            else:
                raise Exception("File "+name+" isn't a pressure or velocity field!")

        return P, U1, U2

    def __len__(self):
        return len(self.P_names)

    def __getitem__(self, idx):

        W = torch.from_numpy(np.load(osp.join(self.root, "params", self.param_names[idx]))).float()
        W = (np.squeeze(W) - self.param_mean) / self.param_std
        W = W

        K = torch.from_numpy(np.load(osp.join(self.root, "permeability", self.perm_names[idx]))).float()
        K = (np.expand_dims(K, axis=0) - self.perm_mean) / self.perm_std

        P = torch.from_numpy(np.load(osp.join(self.root, "data", self.P_names[idx]))).float()
        P = (np.expand_dims(P, axis=0) - self.img_mean[0]) / self.img_std[0]

        '''
        U1 = torch.from_numpy(np.load(osp.join(self.root, "data", self.U1_names[idx]))).float()
        U1 = (np.expand_dims(U1, axis=0) - self.img_mean[1]) / self.img_std[1]

        U2 = torch.from_numpy(np.load(osp.join(self.root, "data", self.U2_names[idx]))).float()
        U2 = (np.expand_dims(U2, axis=0) - self.img_mean[2]) / self.img_std[2]
        '''

        Data = np.concatenate([P, K], axis=0)

        return Data, W


class DarcyLoader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_dataset = Darcy_Dataset(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class Burgers_Dataset(Dataset):
    def __init__(self, path):
        self.root = path

        # load the sample names
        self.sample_names = os.listdir(osp.join(path, "data"))
        self.img_mean = -0.751762357#-3.010598882
        self.img_std = 8.041401807#49.02098157

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):

        W = torch.tensor([0.0]).float()

        Data = torch.from_numpy(np.load(osp.join(self.root, "data", self.sample_names[idx]))).float()
        Data = (np.expand_dims(Data, axis=0) - self.img_mean) / self.img_std

        return Data, W


class BurgersLoader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_dataset = Burgers_Dataset(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class RVEDataset(Dataset):
    def __init__(self, path, idx_list, moduli_mean, moduli_std, fourier=False, grad=False):
        self.root = path
        self.idx_list = idx_list
        self.moduli_mean = moduli_mean
        self.moduli_std = moduli_std
        self.fourier = fourier
        self.grad = grad
        if self.fourier and self.grad:
            raise ValueError("Fourier and grad cannot be used together")

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        if self.fourier:
            # Shape (2, 64, 64)
            rve = np.load(osp.join(self.root, "rves_fourier", f"rve_{self.idx_list[idx]}.npy"))
        else:
            # Shape (64, 64) -> (1, 64, 64)
            rve = np.load(osp.join(self.root, "rves", f"rve_{self.idx_list[idx]}.npy"))
            rve = np.expand_dims(rve, axis=0)

        moduli_raw = np.load(osp.join(self.root, "moduli", f"moduli_{self.idx_list[idx]}.npy"))
        moduli_raw = moduli_raw.reshape(-1)
        moduli = (moduli_raw - self.moduli_mean) / self.moduli_std  # shape (9,)
        grad_raw = np.load(osp.join(self.root, "grads", f"grad_{self.idx_list[idx]}.npy"))  # shape (3, 3, 64, 64)
        grad_raw = grad_raw.reshape(-1, 64, 64)  # shape (9, 64, 64)
        grad = (grad_raw - self.moduli_mean[:, None, None]) / self.moduli_std[:, None, None]  # shape (9, 64, 64)
        rve = torch.from_numpy(rve).float()
        moduli = torch.from_numpy(moduli).float()
        moduli_raw = torch.from_numpy(moduli_raw).float()
        grad = torch.from_numpy(grad).float()
        grad_raw = torch.from_numpy(grad_raw).float()
        out_dict = {"input": rve, "label": moduli, "label_raw": moduli_raw,
                    "grad": grad, "grad_raw": grad_raw}

        return out_dict


class RVELoader(pl.LightningDataModule):
    def __init__(self, data_dir, total_samples, train_samples, val_samples=0,
                 test_samples=0, batch_size=32, num_workers=8, fourier=False,
                 grad=False):
        super().__init__()
        self.data_dir = data_dir
        self.total_samples = total_samples
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.moduli_mean, self.moduli_std = self._compute_moduli_mean_std()
        self.fourier = fourier
        self.grad = grad
        if self.fourier:
            self._compute_fourier_transform()

    def setup(self, stage=None):
        self.train_dataset = RVEDataset(self.data_dir, list(range(self.train_samples)),
                                        self.moduli_mean, self.moduli_std, self.fourier, self.grad)
        self.val_dataset = RVEDataset(
            self.data_dir, list(range(self.train_samples, self.train_samples + self.val_samples)),
            self.moduli_mean, self.moduli_std, self.fourier, self.grad)
        self.test_dataset = RVEDataset(
            self.data_dir, list(range(self.total_samples - self.test_samples,
                                        self.total_samples)),
            self.moduli_mean, self.moduli_std, self.fourier, self.grad)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def _compute_moduli_mean_std(self):
        r"""Go through the dataset and compute the mean and std of the moduli"""
        moduli = []
        for idx in range(self.total_samples):
            moduli.append(np.load(osp.join(self.data_dir, "moduli", f"moduli_{idx}.npy")))
        moduli = np.array(moduli)
        moduli_mean = np.mean(moduli, axis=0).reshape(-1)  # shape (3, 3) -> (9,)
        moduli_std = np.std(moduli, axis=0).reshape(-1)  # shape (3, 3) -> (9,)

        return moduli_mean, moduli_std

    def _compute_fourier_transform(self):
        r"""Compute the Fourier transform of the RVEs in the dataset"""
        os.makedirs(osp.join(self.data_dir, "rves_fourier"), exist_ok=True)
        for idx in range(self.total_samples):
            filename = osp.join(self.data_dir, "rves_fourier", f"rve_{idx}.npy")
            if osp.exists(filename):
                continue
            rve = np.load(osp.join(self.data_dir, "rves", f"rve_{idx}.npy"))
            rve_freq = np.fft.fft2(rve)
            # transform to real and imaginary parts
            # shape (2, 64, 64)
            np.save(filename, np.stack([rve_freq.real, rve_freq.imag], axis=0))
