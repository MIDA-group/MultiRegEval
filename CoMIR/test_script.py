# -*- coding: utf-8 -*-
# Python Standard Libraries
from datetime import datetime
import glob
import itertools
import math
import os
import random
import re
import time
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Deep Learning libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision

# Other libraries
# ~ Scientific
import numpy as np
import scipy.stats as st
# ~ Image manipulation / visualisation
import imgaug
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage
import skimage.io as skio
import skimage.transform as sktr
# ~ Other
from tqdm import tqdm

# Local libraries
from utils.image import *
from utils.plotting import *
from utils.torch import *

# IPython
from IPython.display import clear_output, HTML

# %%
count = torch.cuda.device_count()
print(f"{count} GPU device(s) available.")
print()
print("List of GPUs:")
for i in range(count):
    print(f"* {torch.cuda.get_device_name(i)}")
# %%
# METHOD RELATED
# The place where the models will be saved
export_folder = "results" # Add this path to the .gitignore
# The number of channels in the latent space (best is 1 or 3 for visualization)
latent_channels = 1
# Modality slicing
# You can choose a set of channels per modality (RGB for instance)
# Modality A
modA = slice(0, 1)
modA_name = "SHG"
modA_len = modA.stop - modA.start
# Modality B
modB = slice(1, 4)
modB_name = "BF"
modB_len = modB.stop - modB.start
# Distance function
simfunctions = {
    "euclidean" : lambda x, y: -torch.norm(x - y, p=2, dim=1).mean(),
    "L1"        : lambda x, y: -torch.norm(x - y, p=1, dim=1).mean(),
    "MSE"       : lambda x, y: -(x - y).pow(2).mean(),
    "L3"        : lambda x, y: -torch.norm(x - y, p=3, dim=1).mean(),
    "Linf"      : lambda x, y: -torch.norm(x - y, p=float("inf"), dim=1).mean(),
    "soft_corr" : lambda x, y: F.softplus(x*y).sum(axis=1),
    "corr"      : lambda x, y: (x*y).sum(axis=1),
    "cosine"    : lambda x, y: F.cosine_similarity(x, y, dim=1, eps=1e-8).mean(),
    "angular"   : lambda x, y: F.cosine_similarity(x, y, dim=1, eps=1e-8).acos().mean() / math.pi,
}
sim_func = simfunctions["MSE"]
# Temperature (tau) of the loss
tau =  0.5
# L1/L2 activation regularization
act_l1 = 1e-4 # settings for MSE, MSElong, MSEverylong
act_l2 = 1e-4 # settings for MSE, MSElong, MSEverylong
#act_l1 = 0
#act_l2 = 0 
# p4 Equivariance (should always be True, unless you want to see how everything breaks visually otherwise)
equivariance = True

# DEEP LEARNING RELATED
# Device to train on (inference is done on cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use two GPUs?
device1 = device2 = device # 1 gpu for 2 modalities
#device1, device2 = "cuda:0", "cuda:1" # 1 gpu per modality
# Arguments for the tiramisu neural network
tiramisu_args = {
    # Number of convolutional filters for the first convolution
    "init_conv_filters": 32,
    # Number and depth of down blocks
    "down_blocks": (4, 4, 4, 4, 4, 4),
    # Number and depth of up blocks
    "up_blocks": (4, 4, 4, 4, 4, 4),
    # Number of dense layers in the bottleneck
    "bottleneck_layers": 4,
    # Upsampling type of layer (upsample has no grid artefacts)
    "upsampling_type": "upsample",
    # Type of max pooling, blurpool has better shift-invariance
    "transition_pooling": "max",
    # Dropout rate for the convolution
    "dropout_rate": 0.2,
    # Early maxpooling to reduce the input size
    "early_transition": False,
    # Activation function at the last layer
    "activation_func": None,
    # How much the conv layers should be compressed? (Memory saving)
    "compression": 0.75,
    # Memory efficient version of the tiramisu network (trades memory for computes)
    # Gains of memory are enormous compared to the speed decrease.
    # See: https://arxiv.org/pdf/1707.06990.pdf
    "efficient": True,
}
# Epochs
epochs = 23
# Batch size
batch_size = 32
# Steps per epoch
steps_per_epoch = 32
# Number of steps
steps = steps_per_epoch * epochs
# How many unique patches are fed duriing one epoch
samples_per_epoch = steps_per_epoch * batch_size
# Optimiser
#from lars.lars import LARS
#optimiser = LARS
optimiser = optim.SGD
# Optimizer arguments
opt_args = {
    "lr": 1e-2,
    "weight_decay": 1e-5,
    "momentum": 0.9
}
# Gradient norm. (limit on how big the gradients can get)
grad_norm = 1.0

# DATASET RELATED
def worker_init_fn(worker_id):
    base_seed = int(torch.randint(2**32, (1,)).item())
    lib_seed = (base_seed + worker_id) % (2**32)
    imgaug.seed(lib_seed)
    np.random.seed(lib_seed)

dataloader_args = {
    "batch_size": batch_size,
    "shuffle": False,
    "num_workers": 16,
    "pin_memory": True,
    "worker_init_fn": worker_init_fn,
}
# %%
class SlideDataset(Dataset):
    def __init__(self, folder_path, name_regex=r"(?P<name>.*_(?P<type>.*)\.", logSHG=True, transform=None):
        self.transform = transform
        if not isinstance(folder_path, list):
            folder_path = [folder_path]
        self.path = folder_path
        self.filenames = [glob.glob(path) for path in folder_path]
        self.filenames = list(itertools.chain(*self.filenames))

        dataset = {}
        pbar = tqdm(total=len(self.filenames))
        for pathname in self.filenames:
            filename = os.path.basename(pathname)
            pbar.set_description(filename)
            m = re.search(name_regex, filename, flags=re.IGNORECASE)
            assert m is not None, f"Couldn't find filename in {filename}."
            file_id = m.group("name")
            file_type = m.group("type")

            if file_id not in dataset.keys():
                dataset[file_id] = {}

            img = skio.imread(pathname)
            img = skimage.img_as_float(img)

            if file_type == "SHG" and logSHG:
                img = np.log(1.+img)

            if img.ndim == 2:
                img = img[..., np.newaxis]
            dataset[file_id][file_type] = img
            pbar.update(1)

        self.images = []
        for image_set in dataset:
            try:
                self.images.append(
                    np.block([
                        dataset[image_set]["SHG"],
                        dataset[image_set]["BF"]
                    ]).astype(np.float32)
                )
            except KeyError:
                continue
            except ValueError:
                print(f"Failed concatenating set {image_set}. Shapes are {dataset[image_set]['SHG'].shape} and {dataset[image_set]['BF'].shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx, augment=True):
        if augment and self.transform:
            return self.transform(self.images[idx])
        return self.images[idx]

class ImgAugTransform:
    def __init__(self, testing=False):
        if not testing:
            self.aug = iaa.Sequential([
                iaa.CropToFixedSize(128,128),
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-180, 180), order=[0, 1, 3], mode="symmetric"),
            ])
        else:
            self.aug = iaa.Sequential([
                iaa.CropToFixedSize(128,128),
            ])

    def call(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)
# %%
from pytorch_tiramisu.tiramisu import DenseUNet

class ModNet(DenseUNet):
    def __init__(self, **args):
        super(ModNet, self).__init__(**args, include_top=False)
        out_channels = self.get_channels_count()[-1]
        self.final_conv = torch.nn.Conv2d(out_channels, latent_channels, 1, bias=False)

    def forward(self, x):
        # Penultimate layer
        L_hat = super(ModNet, self).forward(x)
        # Final convolution
        return self.final_conv(L_hat)

torch.manual_seed(0)
modelA = ModNet(in_channels=modA_len, nb_classes=latent_channels, **tiramisu_args).to(device1)
modelB = ModNet(in_channels=modB_len, nb_classes=latent_channels, **tiramisu_args).to(device2)

optimizerA = optimiser(modelA.parameters(), **opt_args)
optimizerB = optimiser(modelB.parameters(), **opt_args)

print("*** MODEL A ***")
modelA.summary()
# %%

# %% Converting the test set
#import time

# Models avaible in the github release
checkpoint = torch.load("models/model_biodata_mse.pt")
#checkpoint = torch.load("models/model_biodata_cosine.pt")
modelA = checkpoint['modelA']
modelB = checkpoint['modelB']
device = "cpu"
modelA.to(device)
modelB.to(device)
modelA.eval()
modelB.eval()

# Number of threads to use
# It seems to be best at the number of physical cores when hyperthreading is enabled
# In our case: 18 physical + 18 logical cores
torch.set_num_threads(0)
# %%
dset = 'Eliceiri'

if dset == 'Stefan':
    dset_registration = SlideDataset([
        "../Datasets/Stefan/SHG/*_SHG.tif",
        "../Datasets/Stefan/BF/*_BF.tif"
    ], name_regex=r"(?P<name>[a-z0-9_]+)_(?P<type>[A-Z]+)", transform=ImgAugTransform(testing=True))
elif dset == 'Eliceiri':
    dset_registration = SlideDataset([
        "./BioData/TrainSet/*_SHG.tif",
        "./BioData/TrainSet/*_BF.tif"
    ], name_regex=r"(?P<name>[a-z0-9_]+)_(?P<type>[A-Z]+)", transform=ImgAugTransform(testing=True))

# %%
def sigmoid(x, tau=1.0):
    """Sigmoid function for ndarrays, safe for big values."""
    return np.exp(-np.logaddexp(
        np.zeros_like(x), - x / tau)
    )

temperature = 1.0
# %%
# How many images to compute in one iteration?
batch_size = 1

N = len(dset_registration)
l, r = 0, batch_size
idx = 1
for i in tqdm(range(int(np.ceil(N / batch_size)))):
    batch = []
    for j in range(l, r):
        batch.append(dset_registration.get(j, augment=False))
    batch = torch.tensor(np.stack(batch), device=device).permute(0, 3, 1, 2)

    newdim = (np.array(batch.shape[2:]) // 128) * 128
    L1 = modelA(batch[:, modA, :newdim[0], :newdim[1]])
    L2 = modelB(batch[:, modB, :newdim[0], :newdim[1]])
    
    for j in range(L1.shape[0]):
        path1 = os.path.basename(dset_registration.filenames[idx]).replace('_SHG.tif', '_SHG_R1.tif')
        L1_hat = L1[j].permute(1, 2, 0).detach()
        if device == 'cuda':
            L1_hat = L1_hat.cpu()
        L1_hat = sigmoid(L1_hat.numpy(), temperature)
        skio.imsave(f"results/{dset}/" + path1, L1_hat)
#        skio.imsave(f"results/{dset}/" + path1, L1[j].permute(1, 2, 0).detach().numpy())
        path2 = os.path.basename(dset_registration.filenames[idx]).replace('_SHG.tif', '_BF_R2.tif')
        L2_hat = L2[j].permute(1, 2, 0).detach()
        if device == 'cuda':
            L2_hat = L2_hat.cpu()
        L2_hat = sigmoid(L2_hat.numpy(), temperature)
        skio.imsave(f"results/{dset}/" + path2, L2_hat)        
#        skio.imsave(f"results/{dset}/" + path2, L2[j].permute(1, 2, 0).detach().numpy())
        idx += 1

    l, r = l+batch_size, r+batch_size
    if r > N:
        r = N