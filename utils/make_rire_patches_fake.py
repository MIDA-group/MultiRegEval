# -*- coding: utf-8 -*-
# make RIRE data to patches
import pandas as pd
import skimage.io as skio
import skimage.transform as skt
import skimage.util as sku
from tqdm import tqdm
from glob import glob
import os, random, math, cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

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

def get_transform_rigid3D(parameters, center=(0, 0, 0)):
    ''' Get a sitk.Transform instance (rotations in radian)
    '''
    transform = sitk.Euler3DTransform()
#    transform.SetRotation(rotations[0], rotations[1], rotations[2])
#    transform.SetTranslation(translations)
    transform.SetParameters(parameters)
    transform.SetCenter(center)
    return transform

def dist_coords(coords1, coords2):
    ''' Calculate the point-to-point distances between two coordinates, return a list.
    coords1: list, coords2: list
    '''
    from scipy import linalg
    return [linalg.norm(np.array(p_1) -  np.array(p_2)) for p_1,p_2 in zip(coords1, coords2)]

def transform_coords(coords, t):
    ''' Transdform coordinates by t, return a list.
    coords: list
    '''
    return [t.TransformPoint(p) for p in coords]

def warp_volume(img_in, transformParameterMap):
    ''' Transdform volume by transformParameterMap
    '''
#    transformix = sitk.TransformixImageFilter()
#    transformix.SetTransformParameterMap(transformParameterMap)
#    transformix.SetMovingImage(sitk.GetImageFromArray(img_in))
#    transformix.Execute()
#    img_out = sitk.GetArrayFromImage(transformix.GetResultImage())
    
    transformedNewImage = sitk.Transformix(sitk.GetImageFromArray(img_in), transformParameterMap)
    img_out = sitk.GetArrayFromImage(transformedNewImage)
    return img_out

def resample_volume(volume, transformation, new_spacing=[1.25, 1.25, 1.25], interpolator=sitk.sitkBSpline):
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, transformation, interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID(), True)

def unpad_sample(img, wo, ho):
    (wi, hi) = img.shape[:2]
    assert wo <= wi and ho <= hi
    wl = (wi - wo) // 2
    hl = (hi - ho) // 2
    return img[wl:wl+wo, hl:hl+ho]

# %%
def make_patches_fake(src_root, tar_root, img_root, real_root, fold=1, n_samples=10):
#    src_root='./Datasets/RIRE_slices_fake'
#    tar_root='./Datasets/RIRE_patches_fake'
#    img_root='./Datasets/RIRE'
#    real_root='./Datasets/RIRE_patches'
#    fold=1
#    n_samples=10
    
    w=80 # patch width

    _, f_names = split_rire_data(fold)
    f_names.sort()
    
    gan_names = os.listdir(f'{src_root}/fold{fold}/')
    
    for f_name in f_names:
        imgA_ori = sitk.ReadImage(f"{img_root}/patient_{f_name}/mr_T1/patient_{f_name}_mr_T1.mhd")
        sizeA = np.asarray([int(round(osz*ospc)) for osz,ospc in zip(imgA_ori.GetSize(), imgA_ori.GetSpacing())])  
        centre_img = sizeA / 2. - 0.5
#        centre_imgB = sizeB / 2. - 0.5
        osA = np.floor((sizeA - w) / 2).astype(int)   # upper-left corner of patch

        for folder in tqdm(gan_names):
            vlm = [skio.imread(f'{src_root}/fold{fold}/{folder}/patient{f_name}_z{z}.png', as_grey=True) for
                   z in range(len(glob(f'{src_root}/fold{fold}/{folder}/patient{f_name}_z*.png')))]
            vlm = np.stack(vlm, axis=-1)
            vlm = unpad_sample(vlm, sizeA[0], sizeA[1])
            vlm = np.rollaxis(vlm, -1, 0)
            vlm = sku.img_as_uint(vlm)
            vlm = sitk.GetImageFromArray(vlm)
            vlm.SetSpacing((1., 1., imgA_ori.GetSpacing()[-1]))
            
            # resample to spacing (1x1x1)
            vlm_resampled = resample_volume(vlm, sitk.Euler3DTransform(), new_spacing=[1., 1., 1.])
            patchA_ref = vlm_resampled[osA[0]:osA[0]+w, osA[1]:osA[1]+w, osA[2]:osA[2]+w]
            
            for t_level in range(1, 5):
                tardir = f'{tar_root}/fold{fold}/patch_tlevel{t_level}/{folder}'
                if not os.path.exists(tardir):
                    os.makedirs(tardir)
                sitk.WriteImage(patchA_ref, f'{tardir}/patient_{f_name}_R.mhd')
                
                df = pd.read_csv(f'{real_root}/fold{fold}/patch_tlevel{t_level}/info_test.csv', index_col='Filename')
                
                for i in range(n_samples):
                    # get transformation parameters
                    tform_param = df.loc[f'{f_name}_{i}', 
                                         ['AngleRadX', 'AngleRadY', 'AngleRadZ', 'Tx', 'Ty', 'Tz']
                                         ].to_list()
                    # transform original images
                    tform = get_transform_rigid3D(tform_param, centre_img)
                    vlm_trans = resample_volume(vlm, tform, new_spacing=[1., 1., 1.])
                    # crop patches
                    patchA_trans = vlm_trans[osA[0]:osA[0]+w, osA[1]:osA[1]+w, osA[2]:osA[2]+w]
                    sitk.WriteImage(patchA_trans, f'{tardir}/patient_{f_name}_{i}_T.mhd')

# %%
if __name__ == '__main__':
#    trans_mins = list(range(0, 28, 7))
#    trans_maxs = list(range(7, 35, 7))
#    rot_mins = list(range(0, 20, 5))
#    rot_maxs = list(range(5, 25, 5))
    for f in range(1, 4):
        make_patches_fake(
                src_root='./Datasets/RIRE_slices_fake',
                tar_root='./Datasets/RIRE_patches_fake',
                img_root='./Datasets/RIRE',
                real_root='./Datasets/RIRE_patches',
                fold=f,
                n_samples=10)
#        make_patches(
#                img_root='./Datasets/Balvan_1to4tiles', 
#                target_root='./Datasets/RIRE_patches',
#                fold=1,
#                t_level=i,
#                mode='train',
#                display=5)