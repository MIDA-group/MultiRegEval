# -*- coding: utf-8 -*-
# prepare training data for pix2pix and others
import skimage.io as skio
import skimage.transform as skt
import skimage.util as sku
from tqdm import tqdm
from glob import glob
import os, cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np


# %%
def split_rire_data(fold):
    if fold == 1:
        ids_test = {'109', '106', '003', '006'}
    elif fold == 2:
        ids_test = {'108', '105', '007', '001'}
    elif fold == 3:
        ids_test = {'107', '102', '005', '009'}
    total = {'109', '106', '003', '006', '108', '105', '007', '001', '107', '102', '005', '009'}
    ids_train = total - ids_test
    return list(ids_train), list(ids_test)

def pad_to_shape(img, out_shape):
    # pad the 3D image to a certain shape 
    img = np.asarray(img)
    (w, h) = img.shape[1:]
    if type(out_shape) is not tuple:
        out_shape = (out_shape, out_shape)
    w_pad = out_shape[0] - w
    h_pad = out_shape[1] - h            
    wl = w_pad // 2
    wr = w_pad - wl
    hl = h_pad // 2
    hr = h_pad - hl
    if img.ndim == 2:
        img_pad = np.pad(img, ((wl, wr), (hl, hr)), 'edge')
    else:
        img_pad = np.pad(img, ((0, 0), (wl, wr), (hl, hr)), 'edge')
    return img_pad

def resample_volume(volume, transformation, new_spacing=[1.25, 1.25, 1.25], interpolator=sitk.sitkBSpline):
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, transformation, interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID(), True)

# %%
#src_dir='./Datasets/RIRE'
#subjects = [os.path.basename(p) for p in glob(f'{src_dir}/patient_*')]
## %%
#shapes = {subj: len(sitk.GetArrayFromImage(sitk.ReadImage(f'{src_dir}/{subj}/mr_T1/{subj}_mr_T1.mhd'))) for subj in subjects}
#total = set(['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '101', '102', '104', '105', '106', '107', '108', '109'])
#sub1 = ['109', '106', '003', '006'] # '000', '101', 
#sub2 = ['108', '105', '007', '001'] # '008', '004', '104', 
#sub3 = ['107', '102', '005', '009'] # '002', 
#print('sub1: \t', f'len: {len(sub1)} \t', 'sum:', sum([shapes[f'patient_{i}'] for i in sub1]))
#print('sub2: \t', f'len: {len(sub2)} \t', 'sum:', sum([shapes[f'patient_{i}'] for i in sub2]))
#print('sub3: \t', f'len: {len(sub3)} \t', 'sum:', sum([shapes[f'patient_{i}'] for i in sub3]))
#print('Unused: ', total - set(sub1)-set(sub2)-set(sub3))

# %%
def make_RIREP2P_folds(src_dir, target_dir, fold):
    '''
    Create folder `/path/to/data` with subfolders `A` and `B`. 
    `A` and `B` should each have their own subfolders `train`, `val`, `test`, etc. 
    In `/path/to/data/A/train`, put training images in style A. 
    In `/path/to/data/B/train`, put the corresponding images in style B. 
    '''
#    src_dir='./Datasets/RIRE'
#    target_dir='./Datasets/RIRE_temp'
#    fold=1
    
    ids_train, ids_test = split_rire_data(fold)
    
    for modality in ['A', 'B']:
        for folder in ['train', 'test']:
            if not os.path.exists(f'{target_dir}/fold{fold}/{modality}/{folder}'):
                os.makedirs(f'{target_dir}/fold{fold}/{modality}/{folder}')
    
#    subjects = [os.path.basename(p).split('_')[-1] for p in glob(f'{src_dir}/patient_*')]
    subjects = ids_train + ids_test
    
    for subj in tqdm(subjects):
#        subj = subjects[0]  # for debugging
        folder = 'train' if subj in ids_train else 'test'
        imgA = sitk.ReadImage(f'{src_dir}/patient_{subj}/mr_T1/patient_{subj}_mr_T1.mhd')
        imgB = sitk.ReadImage(f'{src_dir}/patient_{subj}/mr_T2/patient_{subj}_mr_T2.mhd')

        # Rescale only on x, y axes
        tform = sitk.Euler3DTransform()
        imgA = resample_volume(imgA, tform, new_spacing=[1, 1, imgA.GetSpacing()[-1]])
        imgB = resample_volume(imgB, tform, new_spacing=[1, 1, imgB.GetSpacing()[-1]])

        imgA = sku.img_as_float(sitk.GetArrayFromImage(imgA))
        imgB = sku.img_as_float(sitk.GetArrayFromImage(imgB))
        
        if subj in ids_train:
            imgA = pad_to_shape(imgA, 362)
            imgB = pad_to_shape(imgB, 362)
#            imgA = skt.resize(imgA, (len(imgA), 362, 362))    # Bi-linear
#            imgB = skt.resize(imgB, (len(imgB), 362, 362))
        imgA = cv2.normalize(src=imgA, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        imgB = cv2.normalize(src=imgB, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        for z in range(len(imgA)):
            skio.imsave(f'{target_dir}/fold{fold}/A/{folder}/patient{subj}_z{z}.png', 
                        imgA[z])
            skio.imsave(f'{target_dir}/fold{fold}/B/{folder}/patient{subj}_z{z}.png', 
                        imgB[z])
    return

# %%
for i_fold in [1, 2, 3]:
    make_RIREP2P_folds(
            src_dir='./Datasets/RIRE',
            target_dir='./Datasets/RIRE_temp',
            fold=i_fold)
