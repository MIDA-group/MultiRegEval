# -*- coding: utf-8 -*-
# 
import pandas as pd
import skimage.io as skio
import skimage.transform as skt
from tqdm import tqdm
from glob import glob
import os, random, math, cv2, re
import numpy as np
import matplotlib.pyplot as plt

# %% extract Balvan data from tifs
src_dir = './Datasets/data_skluzavky'
tar_dir = './Datasets/Balvan'
n_frames = 50

dir_GREEN = f'{tar_dir}/GREEN'
dir_QPI = f'{tar_dir}/QPI'
if not os.path.exists(dir_GREEN):
    os.makedirs(dir_GREEN)
if not os.path.exists(dir_QPI):
    os.makedirs(dir_QPI)


names_GREEN = [os.path.basename(p) for p in glob(f'{src_dir}/GREEN*.tif')]

for name_GREEN in names_GREEN:
    name_QPI = name_GREEN.replace('GREEN', 'QPI')
    tif_GREEN = skio.imread(f'{src_dir}/{name_GREEN}')
    tif_QPI = skio.imread(f'{src_dir}/{name_QPI}')
    for i in range(n_frames):
        img_GREEN = tif_GREEN[-(i+1)]
        img_QPI = tif_QPI[-(i+1)]
        img_GREEN = cv2.normalize(src=img_GREEN, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        p_low = np.percentile(img_QPI, 1, axis=None)
        p_high = np.percentile(img_QPI, 99.9, axis=None)
        img_QPI = np.clip((img_QPI - p_low) / (p_high - p_low + 1e-15), p_low, p_high)
        img_QPI = cv2.normalize(src=img_QPI, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        name_save = name_GREEN.replace('GREEN_','').replace('.tif', f'_{"0"*(2-len(str(i)))+str(i)}.tif')
        skio.imsave(f'{dir_GREEN}/{name_save}', img_GREEN)
        skio.imsave(f'{dir_QPI}/{name_save}', img_QPI)
    


#patt_tif = re.compile(r'(\w+)_(\w+)_(\w+)_(\w+).tif')
#for name_GREEN in names_GREEN:
#    m = patt_tif.match(name_GREEN)
#    if m:
#        (modality, cellline, treatment, fov) = m.groups()
    




# %%
w=600
centre_patch = np.array((w, w)) / 2. - 0.5

def tform_centred(radian, translation, center):
    # first translation, then rotation
    tform1 = skt.SimilarityTransform(translation=center)
    tform2 = skt.SimilarityTransform(rotation=radian)
    tform3 = skt.SimilarityTransform(translation=-center)
    tform4 = skt.SimilarityTransform(translation=translation)
    tform = tform4 + tform3 + tform2 + tform1
    return tform

(rot_min, rot_max) = (10, 15)
(trans_min, trans_max) = (40, 60)

# %%
img_G = skio.imread('./Datasets/GREEN_LNCaP_do_2.tif')
img_QPI = skio.imread('./Datasets/QPI_LNCaP_do_2.tif')

#img_G = cv2.normalize(src=img_G, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#img_QPI = cv2.normalize(src=img_QPI, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# %%
img1 = img_G[-30]
img2 = img_QPI[-30]

# %%
img1 = cv2.normalize(src=img1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
img2 = cv2.normalize(src=img2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# random transformation parameters
rot_degree = random.choice((random.uniform(-rot_max, -rot_min), random.uniform(rot_min, rot_max)))
tx = random.choice((random.uniform(-trans_max, -trans_min), random.uniform(trans_min, trans_max)))
ty = random.choice((random.uniform(-trans_max, -trans_min), random.uniform(trans_min, trans_max)))
rot_radian = np.deg2rad(rot_degree)


tform_img = tform_centred(radian=rot_radian, translation=(tx, ty), center=centre_patch)
img1_trans = np.asarray(skt.warp(img1, tform_img, preserve_range=True), dtype=np.uint8)
img2_trans = np.asarray(skt.warp(img2, tform_img, preserve_range=True), dtype=np.uint8)

#img2_rec, field = register_mi(img2, img1_trans)
img_src = sitk.GetImageFromArray(img2)
img_tar = sitk.GetImageFromArray(img1_trans)
    
parameterMap = sitk.GetDefaultParameterMap('rigid')
parameterMap['ResultImagePixelType'] = ['uint8']
parameterMap['NumberOfResolutions'] = ['7']
parameterMap['MaximumNumberOfIterations'] = ['1024']

#parameterMap['Optimizer'] = ['AdaGrad']

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(img_tar)
elastixImageFilter.SetMovingImage(img_src)
elastixImageFilter.SetParameterMap(parameterMap)
elastixImageFilter.Execute()

resultImage = elastixImageFilter.GetResultImage()
img2_rec = sitk.GetArrayFromImage(resultImage).astype('uint8')
transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0].asdict()
tform_parameter = transformParameterMap['TransformParameters']

skio.imshow(img2_rec)
# %%
skio.imshow(img2_trans)



