
#
# Script for performing inference of CoMIR:s
# Authors: Nicolas Pielawski, Elisabeth Wetzer, Johan Ofverstedt
# Published under the MIT License
# 2020
#

# Python Standard Libraries
from datetime import datetime
import glob
import itertools
import math
import os
import time
import sys
import random
import re
import warnings

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
import scipy
import scipy.stats as st
import scipy.special
# ~ Image manipulation
import imgaug
from imgaug import augmenters as iaa
import skimage
import skimage.io as skio
import skimage.transform as sktr

# Local libraries
from utils.image import *
from utils.torch import *

#logTransformA = True
#logTransformB = False

apply_sigmoid = True

# %%
# Create model

from models.tiramisu import DenseUNet

class ModNet(DenseUNet):
    def __init__(self, **args):
        super(ModNet, self).__init__(**args, include_top=False)
        out_channels = self.get_channels_count()[-1]
        self.final_conv = torch.nn.Conv2d(out_channels, latent_channels, 1, bias=False)
        # This is merely for the benefit of the serialization (so it will be known in the inference)
        self.log_transform = False

    def set_log_transform(self, flag):
        # This is merely for the benefit of the serialization (so it will be known in the inference)
        self.log_transform = flag

    def forward(self, x):
        # Penultimate layer
        L_hat = super(ModNet, self).forward(x)
        # Final convolution
        return self.final_conv(L_hat)


# Create 

def filenames_to_dict(filenamesA, filenamesB):
    d = {}
    for i in range(len(filenamesA)):
        basename = os.path.basename(filenamesA[i])
        d[basename] = (i, None)
    for i in range(len(filenamesB)):
        basename = os.path.basename(filenamesB[i])
        # filter out files only in B
        if basename in d:
            d[basename] = (d[basename][0], i)

    # filter out files only in A
    d = {k:v for k,v in d.items() if v[1] is not None}
    return d

class MultimodalDataset(Dataset):
    def __init__(self, pathA, pathB, logA=False, logB=False, transform=None):
        self.transform = transform

        if not isinstance(pathA, list):
            pathA = [pathA]
        if not isinstance(pathB, list):
            pathB = [pathB]
        self.pathA = pathA
        self.pathB = pathB
        self.filenamesA = [glob.glob(path) for path in pathA]
        self.filenamesA = list(itertools.chain(*self.filenamesA))
        self.filenamesB = [glob.glob(path) for path in pathB]
        self.filenamesB = list(itertools.chain(*self.filenamesB))

        self.channels = [None, None]

        self.base_names = []

        filename_index_pairs = filenames_to_dict(self.filenamesA, self.filenamesB)
        
        filenames = [self.filenamesA, self.filenamesB]
        log_flags = [logA, logB]

        dataset = {}
        for mod_ind in range(2):
            # Read all files from modality
            for filename, inds in filename_index_pairs.items():
                pathname = filenames[mod_ind][inds[mod_ind]]

                filename = os.path.basename(pathname)
                
                if filename not in dataset.keys():
                    dataset[filename] = [None, None]

                img = skio.imread(pathname)
                img = skimage.img_as_float(img)

                if log_flags[mod_ind]:
                    img = np.log(1.+img)

                if img.ndim == 2:
                    img = img[..., np.newaxis]

                if self.channels[mod_ind] is None:
                    self.channels[mod_ind] = img.shape[2]

                dataset[filename][mod_ind] = img

        self.images = []
        for image_set in dataset:
            try:
                self.images.append(
                    np.block([
                        dataset[image_set][0],
                        dataset[image_set][1]
                    ]).astype(np.float32)
                )
                self.base_names.append(image_set)
            except ValueError:
                print(f"Failed concatenating set {image_set}. Shapes are {dataset[image_set][0].shape} and {dataset[image_set][1].shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx, augment=True):
        if augment and self.transform:
            return self.transform(self.images[idx])
        return self.images[idx]
    
    def get_name(self, idx):
        return self.base_names[idx]

# %%
print(len(sys.argv), sys.argv)
if len(sys.argv) < 6:
    print('Use: inference_comir.py model_path mod_a_path mod_b_path mod_a_out_path mod_b_out_path')
    sys.exit(-1)

model_path = sys.argv[1]
modA_path = sys.argv[2]
modB_path = sys.argv[3]
modA_out_path = sys.argv[4]
modB_out_path = sys.argv[5]

if modA_path[-1] != '/':
    modA_path += '/'
if modB_path[-1] != '/':
    modB_path += '/'
if modA_out_path[-1] != '/':
    modA_out_path += '/'
if modB_out_path[-1] != '/':
    modB_out_path += '/'

if not os.path.exists(modA_out_path):
    os.makedirs(modA_out_path)
if not os.path.exists(modB_out_path):
    os.makedirs(modB_out_path)
#os.system('mkdir -p ' + modA_out_path)
#os.system('mkdir -p ' + modB_out_path)

checkpoint = torch.load(model_path)

modelA = checkpoint['modelA']
modelB = checkpoint['modelB']

print("Loading dataset...")
dset = MultimodalDataset(modA_path + '*', modB_path + '*', logA=modelA.log_transform, logB=modelB.log_transform, transform=None)

# Modality slicing
# You can choose a set of channels per modality (RGB for instance)
# Modality A
modA_len = modelA.in_channels #dset.channels[0]
modA = slice(0, modA_len)
modA_name = "A"
# Modality B
modB_len = modelB.in_channels #dset.channels[1]
modB = slice(modA_len, modA_len + modB_len)
modB_name = "B"
print('Modality A has ', modA_len, ' channels.', sep='')
print('Modality B has ', modB_len, ' channels.', sep='')
if modelA.log_transform:
    print('Modality A uses a log transform.')
if modelB.log_transform:
    print('Modality B uses a log transform.')

if torch.cuda.is_available():
    device = torch.device('cuda')
    modelA.to(device)
    modelB.to(device)
#    modelA.half()
#    modelB.half()
else:
    device = torch.device('cpu')
    modelA.to(device)
    modelB.to(device)
modelA.eval()
modelB.eval()

# Number of threads to use
# It seems to be best at the number of physical cores when hyperthreading is enabled
# In our case: 18 physical + 18 logical cores
torch.set_num_threads(5)

def compute_padding(sz, alignment=128):
    if sz % alignment == 0:
        return 0
    else:
        return alignment - (sz % alignment)

# How many images to compute in one iteration?
batch_size = 1

#all_paths = []

N = len(dset)
l, r = 0, batch_size
idx = 1
for i in range(int(np.ceil(N / batch_size))):
    batch = []
    names = []
    for j in range(l, r):
        batch.append(dset.get(j, augment=False))
        names.append(dset.get_name(j))

    with torch.cuda.amp.autocast(): # pytorch>=1.6.0 required
        batch = torch.tensor(np.stack(batch), device=device).permute(0, 3, 1, 2)
    
    #    if device.type == 'cuda':
    #        batch = batch.half()
    
        padsz = 128
        orig_shape = batch.shape
        pad1 = compute_padding(batch.shape[-2])
        pad2 = compute_padding(batch.shape[-1])
    
        padded_batch = F.pad(batch, (padsz, padsz+pad1, padsz, padsz + pad2), mode='reflect')
        #newdim = (np.array(batch.shape[2:]) // 128) * 128
        #L1 = modelA(batch[:, modA, :newdim[0], :newdim[1]])
        #L2 = modelB(batch[:, modB, :newdim[0], :newdim[1]])
        L1 = modelA(padded_batch[:, modA, :, :])
        L2 = modelB(padded_batch[:, modB, :, :])
        
        L1 = L1[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3]]
        L2 = L2[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3]]
    
        for j in range(len(batch)):#L1.shape[0]):
            path1 = modA_out_path + names[j]
            path2 = modB_out_path + names[j]
    
    #        print(path1)
    #        print(path2)
    
    #        all_paths.append(path1)
    #        all_paths.append(path2)
    
            if device.type == 'cuda':
                im1 = L1[j].permute(1, 2, 0).cpu().detach().numpy()
                im2 = L2[j].permute(1, 2, 0).cpu().detach().numpy()
            else:
                im1 = L1[j].permute(1, 2, 0).detach().numpy()
                im2 = L2[j].permute(1, 2, 0).detach().numpy()
            if apply_sigmoid:
                im1 = np.round(scipy.special.expit(im1) * 255).astype('uint8')
                im2 = np.round(scipy.special.expit(im2) * 255).astype('uint8')
                skio.imsave(path1, im1)
                skio.imsave(path2, im2)
            else:
                skio.imsave(path1, im1)
                skio.imsave(path2, im2)
            print(f'Encodeing... {idx}/{N}')
            idx += 1

    del L1
    del L2
    
    l, r = l+batch_size, r+batch_size
    if r > N:
        r = N

#all_paths = sorted(all_paths)
#for i in range(len(all_paths)):
#    print(all_paths[i])
