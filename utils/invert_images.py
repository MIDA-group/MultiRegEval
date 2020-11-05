# -*- coding: utf-8 -*-
# Python Standard Libraries
from glob import glob
import os, argparse, shutil

# ~ Scientific
import numpy as np
# ~ Image manipulation / visualisation
import matplotlib.pyplot as plt
import skimage.io as skio
# ~ Other
from tqdm import tqdm

# %%

def create_inverted_dataset(dataroot):
#    dataroot = '../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_train'
    
    tar_dirA = f'{dataroot}_Ainverted/trainA'
    if not os.path.exists(tar_dirA):
        os.makedirs(tar_dirA)

    img_paths = glob(f'{dataroot}/trainA/*')
    
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path)
        img = skio.imread(img_path)
        img = 255 - img
        skio.imsave(f'{tar_dirA}/{img_name}', img)
    
    shutil.copytree(f'{dataroot}/trainB', f'{dataroot}_Ainverted/trainB')
    
    return

# %%
parser = argparse.ArgumentParser(description='Invert images in "trainA"')
parser.add_argument(
        '--path', '-p', 
        required=True,
        help="dataroot containing 'trainA' and 'trainB'")

args = parser.parse_args()

create_inverted_dataset(args.path)

