# -*- coding: utf-8 -*-
import skimage.io as skio
import skimage.util as sku
from tqdm import tqdm
from glob import glob
import os
import SimpleITK as sitk
import numpy as np

# %% Helper functions
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

def unpad_sample(img, wo, ho):
    (wi, hi) = img.shape[:2]
    assert wo <= wi and ho <= hi
    wl = (wi - wo) // 2
    hl = (hi - ho) // 2
    return img[wl:wl+wo, hl:hl+ho]

def pad_to_shape(img, out_shape):
    # pad the 3D image to a certain shape 
    img = np.asarray(img)
    (w, h) = img.shape
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

# %%
def make_RIRE_forFID():
    dataset='RIRE'
    src_realdir=f'./Datasets/{dataset}_temp'
    src_fakedir=f'./Datasets/{dataset}_slices_fake'
    tar_realdir=f'./Datasets/{dataset}_patches_forFID/real'
    tar_fakedir=f'./Datasets/{dataset}_patches_forFID/fake'
    
    for fold in range(1, 4):
        _, f_names = split_rire_data(fold)
        f_names.sort()
        
        gan_names = os.listdir(f'{src_fakedir}/fold{fold}/')
        
        for f_name in f_names:
            imgA_ori = sitk.ReadImage(f"./Datasets/RIRE/patient_{f_name}/mr_T1/patient_{f_name}_mr_T1.mhd")
            sizeA = np.asarray([int(round(osz*ospc)) for osz,ospc in zip(imgA_ori.GetSize(), imgA_ori.GetSpacing())])  
    
            for folder in tqdm(gan_names):
                tardir = f'{tar_fakedir}/fold{fold}/{folder}'
                if not os.path.exists(tardir):
                    os.makedirs(tardir)
                for z in range(len(glob(f'{src_fakedir}/fold{fold}/{folder}/patient{f_name}_z*.png'))):
                    img = skio.imread(f'{src_fakedir}/fold{fold}/{folder}/patient{f_name}_z{z}.png', True)
                    img = sku.img_as_ubyte(img)
                    img = unpad_sample(img, sizeA[0], sizeA[1])
                    img = pad_to_shape(img, 320)
                    skio.imsave(f'{tardir}/patient{f_name}_z{z}.png', img)

            for folder in tqdm(['A', 'B']):
                tardir = f'{tar_realdir}/fold{fold}/{folder}/test'
                if not os.path.exists(tardir):
                    os.makedirs(tardir)
                for z in range(len(glob(f'{src_realdir}/fold{fold}/{folder}/test/patient{f_name}_z*.png'))):
                    img = skio.imread(f'{src_realdir}/fold{fold}/{folder}/test/patient{f_name}_z{z}.png', True)
                    img = sku.img_as_ubyte(img)
                    img = pad_to_shape(img, 320)
                    skio.imsave(f'{tardir}/patient{f_name}_z{z}.png', img)
    return

# %%
if __name__ == '__main__':
    make_RIRE_forFID()

