# -*- coding: utf-8 -*-
# extract Balvan data from tifs
import pandas as pd
import skimage.io as skio
import skimage.transform as skt
from tqdm import tqdm
from glob import glob
import os, random, math, cv2, re
import numpy as np
import matplotlib.pyplot as plt

# %%
src_dir = './Datasets/data_skluzavky'
tar_dir = './Datasets/Balvan'
n_frames = 60

dir_GREEN = f'{tar_dir}/GREEN'
dir_QPI = f'{tar_dir}/QPI'
if not os.path.exists(dir_GREEN):
    os.makedirs(dir_GREEN)
if not os.path.exists(dir_QPI):
    os.makedirs(dir_QPI)


names_GREEN = [os.path.basename(p) for p in glob(f'{src_dir}/GREEN*.tif')]

## traverse all tif stacks with the same modality and recored the min and max intensity
#GREEN_max, QPI_max = -math.inf, -math.inf
#GREEN_min, QPI_min = math.inf, math.inf
#for name_GREEN in tqdm(names_GREEN):
##    name_GREEN = names_GREEN[3]  # for debugging
#    name_QPI = name_GREEN.replace('GREEN', 'QPI')
#    tif_GREEN = skio.imread(f'{src_dir}/{name_GREEN}')
#    tif_QPI = skio.imread(f'{src_dir}/{name_QPI}')
#    GREEN_max_temp = tif_GREEN.max()
#    GREEN_min_temp = tif_GREEN.min()
#    QPI_max_temp = tif_QPI.max()
#    QPI_min_temp = tif_QPI.min()
#    if GREEN_max_temp > GREEN_max:
#        GREEN_max = GREEN_max_temp
#    if GREEN_min_temp < GREEN_min:
#        GREEN_min = GREEN_min_temp
#    if QPI_max_temp > QPI_max:
#        QPI_max = QPI_max_temp
#    if QPI_min_temp < QPI_min:
#        QPI_min = QPI_min_temp
#    


for name_GREEN in tqdm(names_GREEN):
#    name_GREEN = names_GREEN[3]  # for debugging
    name_QPI = name_GREEN.replace('GREEN', 'QPI')
    tif_GREEN = skio.imread(f'{src_dir}/{name_GREEN}')
    tif_QPI = skio.imread(f'{src_dir}/{name_QPI}')
    
#    # global normalisation
#    tif_GREEN = (tif_GREEN - GREEN_min) / (GREEN_max - GREEN_min)
#    tif_QPI = (tif_QPI - QPI_min) / (QPI_max - QPI_min)
    tif_GREEN = cv2.normalize(src=tif_GREEN, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#    p_low_QPI = np.percentile(tif_QPI, 1, axis=None)
#    p_high_QPI = np.percentile(tif_QPI, 99.9, axis=None)
#    tif_QPI = np.clip((tif_QPI - p_low_QPI) / (p_high_QPI - p_low_QPI + 1e-15), p_low_QPI, p_high_QPI)
    tif_QPI = cv2.normalize(src=tif_QPI, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # extract i_frame 0, 20, 40, 60 backwards from the last
    for i in range(0, n_frames+1, 15):
        img_GREEN = tif_GREEN[-(i+1)]
#        img_GREEN = cv2.normalize(src=img_GREEN, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_QPI = tif_QPI[-(i+1)]
        # normalise QPI to percentile 1-99.9
#        p_low_QPI = np.percentile(img_QPI, 1, axis=None)
#        p_high_QPI = np.percentile(img_QPI, 99.9, axis=None)
#        img_QPI = np.clip((img_QPI - p_low_QPI) / (p_high_QPI - p_low_QPI + 1e-15), p_low_QPI, p_high_QPI)
#        img_QPI = cv2.normalize(src=img_QPI, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#        skio.imshow(img_QPI)
        name_save = name_GREEN.replace('GREEN_','').replace('.tif', f'_f{"0"*(2-len(str(i)))+str(i)}.tif')
        skio.imsave(f'{dir_GREEN}/{name_save}', img_GREEN)
        skio.imsave(f'{dir_QPI}/{name_save}', img_QPI)
    


