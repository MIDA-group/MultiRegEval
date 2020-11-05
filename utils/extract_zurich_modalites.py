# -*- coding: utf-8 -*-
# extract Balvan data from tifs
import pandas as pd
import skimage.io as skio
import skimage.transform as skt
import skimage.util as sku
from tqdm import tqdm
from glob import glob
import os, random, math, cv2, re
import numpy as np
import matplotlib.pyplot as plt

# %%
src_dir = './Datasets/Zurich_dataset_v1.0/images_tif'
tar_dir = './Datasets/Zurich'


dir_IR = f'{tar_dir}/IR'
dir_RGB = f'{tar_dir}/RGB'
if not os.path.exists(dir_IR):
    os.makedirs(dir_IR)
if not os.path.exists(dir_RGB):
    os.makedirs(dir_RGB)


names_tif = [os.path.basename(p) for p in glob(f'{src_dir}/*.tif')]

#patt_tif = re.compile(r'zh(\d+).tif')

# find golbal max and min
maxis = []
minis = []
shapes = []
for id_tif in range(1, len(names_tif)+1):

    name_tif = f'zh{id_tif}.tif'
    
#    m = patt_tif.match(name_tif)
#    if m:
#        (id_tif, ) = m.groups()
    
    img = skio.imread(f'{src_dir}/{name_tif}')
    maxis.append(img.max(axis=(0,1)))
    minis.append(img.min(axis=(0,1)))
    shapes.append(img.shape)

ch_maxis = np.asarray(maxis).max(axis=0)    # maximum value in each channel
ch_minis = np.asarray(minis).min(axis=0)    # minimum value in each channel
shapes = np.asarray(shapes)

for id_tif in tqdm(range(1, len(names_tif)+1)):
    name_tif = f'zh{id_tif}.tif'    
    img = skio.imread(f'{src_dir}/{name_tif}')
    img_norm = (img - ch_minis) / (ch_maxis - ch_minis + 1e-15)   # channel-wise normalise to 0-1
    img_norm = sku.img_as_ubyte(img_norm)   # convert to uint8
#    skio.imshow(img_norm[..., :0:-1])
    img_RGB = img_norm[..., :-1]
    img_IR = img_norm[..., -1]
    name_save = f'zh{id_tif}.png'
    skio.imsave(f'{dir_RGB}/{name_save}', img_RGB)
    skio.imsave(f'{dir_IR}/{name_save}', img_IR)
    
#    # visualise
#    p_low_RGB = np.percentile(img_RGB, 1, axis=None)
#    p_high_RGB = np.percentile(img_RGB, 99, axis=None)
#    p_low_IR = np.percentile(img_IR, 1, axis=None)
#    p_high_IR = np.percentile(img_IR, 99, axis=None)
#    
#    img_en = np.empty(img.shape, dtype='uint8')
#    img_en[..., :-1] = cv2.normalize(
#            src=np.clip(img_RGB, p_low_RGB, p_high_RGB), 
#            dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#    img_en[..., -1] = img_IR = cv2.normalize(
#            src=np.clip(img_IR, p_low_IR, p_high_IR), 
#            dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#    
#    skio.imshow(img_en[..., :0:-1])


