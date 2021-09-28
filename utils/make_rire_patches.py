# -*- coding: utf-8 -*-
# make RIRE data to patches
import pandas as pd
import skimage.io as skio
import skimage.transform as skt
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

# %%
def make_patches(img_root, target_root, fold=1, t_level=1, n_samples=10,
#                 trans_min=0, trans_max=15, rot_min=0, rot_max=5, 
                 mode='train', display=None):
#    img_root='./Datasets/RIRE'
#    target_root='./Datasets/RIRE_patches'
#    fold=1
#    t_level=2
#    n_samples=10
#    trans_min=0
#    trans_max=15
#    rot_min=0
#    rot_max=5
#    mode='test'
    
    w=80 # patch width
  
    
    #modalities = {'GREEN':'A', 'QPI':'B'}
    tardirA = f'{target_root}/fold{fold}/patch_tlevel{t_level}/A/{mode}'
    tardirB = f'{target_root}/fold{fold}/patch_tlevel{t_level}/B/{mode}'
    if not os.path.exists(tardirA):
        os.makedirs(tardirA)
    if not os.path.exists(tardirB):
        os.makedirs(tardirB)
    
    ids_train, ids_test = split_rire_data(fold)
    f_names = ids_train if mode == 'train' else ids_test
    f_names.sort()
    
    # csv information
    header = [
            'ReferenceImage', 'Method', 
            'X1_Ref', 'Y1_Ref', 'Z1_Ref', 
            'X2_Ref', 'Y2_Ref', 'Z2_Ref', 
            'X3_Ref', 'Y3_Ref', 'Z3_Ref', 
            'X4_Ref', 'Y4_Ref', 'Z4_Ref', 
            'X5_Ref', 'Y5_Ref', 'Z5_Ref', 
            'X6_Ref', 'Y6_Ref', 'Z6_Ref', 
            'X7_Ref', 'Y7_Ref', 'Z7_Ref', 
            'X8_Ref', 'Y8_Ref', 'Z8_Ref', 
            'X1_Trans', 'Y1_Trans', 'Z1_Trans', 
            'X2_Trans', 'Y2_Trans', 'Z2_Trans', 
            'X3_Trans', 'Y3_Trans', 'Z3_Trans', 
            'X4_Trans', 'Y4_Trans', 'Z4_Trans', 
            'X5_Trans', 'Y5_Trans', 'Z5_Trans', 
            'X6_Trans', 'Y6_Trans', 'Z6_Trans', 
            'X7_Trans', 'Y7_Trans', 'Z7_Trans', 
            'X8_Trans', 'Y8_Trans', 'Z8_Trans', 
            'X1_Recover', 'Y1_Recover', 'Z1_Recover', 
            'X2_Recover', 'Y2_Recover', 'Z2_Recover', 
            'X3_Recover', 'Y3_Recover', 'Z3_Recover', 
            'X4_Recover', 'Y4_Recover', 'Z4_Recover', 
            'X5_Recover', 'Y5_Recover', 'Z5_Recover', 
            'X6_Recover', 'Y6_Recover', 'Z6_Recover', 
            'X7_Recover', 'Y7_Recover', 'Z7_Recover', 
            'X8_Recover', 'Y8_Recover', 'Z8_Recover', 
            'Displacement', 'RelativeDisplacement', 
            'Tx', 'Ty', 'Tz', 
            'AngleDegreeX', 'AngleDegreeY', 'AngleDegreeZ', 
            'AngleRadX', 'AngleRadY', 'AngleRadZ', 
            'Error', 'RelativeError']
    df = pd.DataFrame(index=[f'{f_name}_{i}' for f_name in f_names for i in range(n_samples)], columns=header)
    df.index.set_names('Filename', inplace=True)
    
    if display is not None:
        cnt_disp = 0
    for f_name in tqdm(f_names):
        # load original images
        imgA = sitk.ReadImage(f"{img_root}/patient_{f_name}/mr_T1/patient_{f_name}_mr_T1.mhd")
        imgB = sitk.ReadImage(f"{img_root}/patient_{f_name}/mr_T2/patient_{f_name}_mr_T2.mhd")
        
        # resample to spacing (1x1x1)
        imgA_resampled = resample_volume(imgA, sitk.Euler3DTransform(), new_spacing=[1., 1., 1.])
        imgB_resampled = resample_volume(imgB, sitk.Euler3DTransform(), new_spacing=[1., 1., 1.])
        
        sizeA = np.asarray(imgA_resampled.GetSize())
        sizeB = np.asarray(imgB_resampled.GetSize())
        centre_img = sizeA / 2. - 0.5
#        centre_imgB = sizeB / 2. - 0.5
        osA = np.floor((sizeA - w) / 2).astype(int)   # upper-left corner of patch
        osB = np.floor((sizeB - w) / 2).astype(int)
        
        
        # crop patches from resampled volumes
        patchA_ref = imgA_resampled[osA[0]:osA[0]+w, osA[1]:osA[1]+w, osA[2]:osA[2]+w]
        patchB_ref = imgB_resampled[osB[0]:osB[0]+w, osB[1]:osB[1]+w, osB[2]:osB[2]+w]
        sitk.WriteImage(patchA_ref, f'{tardirA}/patient_{f_name}_R.mhd')
        sitk.WriteImage(patchB_ref, f'{tardirB}/patient_{f_name}_R.mhd')

#        sizeA = [int(round(osz*ospc)) for osz,ospc in zip(imgA.GetSize(), imgA.GetSpacing())]
#        sizeB = [int(round(osz*ospc)) for osz,ospc in zip(imgB.GetSize(), imgB.GetSpacing())]

        coords_ref = [
                (osA[0], osA[1], osA[2]), (osA[0]+w, osA[1], osA[2]), 
                (osA[0], osA[1]+w, osA[2]), (osA[0]+w, osA[1]+w, osA[2]), 
                (osA[0], osA[1], osA[2]+w), (osA[0]+w, osA[1], osA[2]+w), 
                (osA[0], osA[1]+w, osA[2]+w), (osA[0]+w, osA[1]+w, osA[2]+w), 
                ]
        
        step_trans = 7 / 300 * w
        step_rot = 5
        trans_min = step_trans * (t_level - 1)
        trans_max = step_trans * t_level
        rot_min = step_rot * (t_level - 1)
        rot_max = step_rot * t_level
        
        for i in range(n_samples):
            # random transformation parameters
            rs_degree = [random.choice((random.uniform(-rot_max, -rot_min), random.uniform(rot_min, rot_max))) for d in range(3)]
            rs_radian = [np.deg2rad(r_degree) for r_degree in rs_degree]
            ts = [random.choice((random.uniform(-trans_max, -trans_min), random.uniform(trans_min, trans_max))) for d in range(3)]
                    
            # transform original images
            tform = get_transform_rigid3D(rs_radian+ts, centre_img)
            imgA_trans = resample_volume(imgA, tform, new_spacing=[1., 1., 1.])
            imgB_trans = resample_volume(imgB, tform, new_spacing=[1., 1., 1.])
                        
            # crop patches
            patchA_trans = imgA_trans[osA[0]:osA[0]+w, osA[1]:osA[1]+w, osA[2]:osA[2]+w]
            patchB_trans = imgB_trans[osB[0]:osB[0]+w, osB[1]:osB[1]+w, osB[2]:osB[2]+w]
    
            # transform patch coordinates
            coords_trans = transform_coords(np.asarray(coords_ref).tolist(), tform)
            
            # calculate distance
            dist_array = dist_coords(coords_trans, coords_ref)
            dist = np.mean(dist_array)
            
            
            # write csv line
            line = {
                'X1_Ref': coords_ref[0][0], 'Y1_Ref': coords_ref[0][1], 'Z1_Ref': coords_ref[0][2], 
                'X2_Ref': coords_ref[1][0], 'Y2_Ref': coords_ref[1][1], 'Z2_Ref': coords_ref[1][2], 
                'X3_Ref': coords_ref[2][0], 'Y3_Ref': coords_ref[2][1], 'Z3_Ref': coords_ref[2][2], 
                'X4_Ref': coords_ref[3][0], 'Y4_Ref': coords_ref[3][1], 'Z4_Ref': coords_ref[3][2], 
                'X5_Ref': coords_ref[4][0], 'Y5_Ref': coords_ref[4][1], 'Z5_Ref': coords_ref[4][2], 
                'X6_Ref': coords_ref[5][0], 'Y6_Ref': coords_ref[5][1], 'Z6_Ref': coords_ref[5][2], 
                'X7_Ref': coords_ref[6][0], 'Y7_Ref': coords_ref[6][1], 'Z7_Ref': coords_ref[6][2], 
                'X8_Ref': coords_ref[7][0], 'Y8_Ref': coords_ref[7][1], 'Z8_Ref': coords_ref[7][2], 
                'X1_Trans': coords_trans[0][0], 'Y1_Trans': coords_trans[0][1], 'Z1_Trans': coords_trans[0][2], 
                'X2_Trans': coords_trans[1][0], 'Y2_Trans': coords_trans[1][1], 'Z2_Trans': coords_trans[1][2], 
                'X3_Trans': coords_trans[2][0], 'Y3_Trans': coords_trans[2][1], 'Z3_Trans': coords_trans[2][2], 
                'X4_Trans': coords_trans[3][0], 'Y4_Trans': coords_trans[3][1], 'Z4_Trans': coords_trans[3][2], 
                'X5_Trans': coords_trans[4][0], 'Y5_Trans': coords_trans[4][1], 'Z5_Trans': coords_trans[4][2], 
                'X6_Trans': coords_trans[5][0], 'Y6_Trans': coords_trans[5][1], 'Z6_Trans': coords_trans[5][2], 
                'X7_Trans': coords_trans[6][0], 'Y7_Trans': coords_trans[6][1], 'Z7_Trans': coords_trans[6][2], 
                'X8_Trans': coords_trans[7][0], 'Y8_Trans': coords_trans[7][1], 'Z8_Trans': coords_trans[7][2], 
                'Displacement': dist, 
                'RelativeDisplacement': dist/w,
                'Tx': ts[0], 'Ty': ts[1], 'Tz': ts[2], 
                'AngleDegreeX': rs_degree[0], 'AngleDegreeY': rs_degree[1], 'AngleDegreeZ': rs_degree[2], 
                'AngleRadX': rs_radian[0], 'AngleRadY': rs_radian[1], 'AngleRadZ': rs_radian[2], 
                }
            df.loc[f'{f_name}_{i}'] = line
            
            # save volumes
            sitk.WriteImage(patchA_trans, f'{tardirA}/patient_{f_name}_{i}_T.mhd')
            sitk.WriteImage(patchB_trans, f'{tardirB}/patient_{f_name}_{i}_T.mhd')
            
            # display patch outline in original image
            if display is not None and cnt_disp < display:
                pass
                cnt_disp += 1
    
    df.to_csv(f'{target_root}/fold{fold}/patch_tlevel{t_level}/info_{mode}.csv')

# %%
if __name__ == '__main__':
#    trans_mins = list(range(0, 28, 7))
#    trans_maxs = list(range(7, 35, 7))
#    rot_mins = list(range(0, 20, 5))
#    rot_maxs = list(range(5, 25, 5))
    for f in range(1, 4):
        for i in range(1, 5):
            make_patches(
                    img_root='./Datasets/RIRE', 
                    target_root='./Datasets/RIRE_patches',
                    fold=f,
                    t_level=i,
                    n_samples=10,
                    mode='test',
                    display=None)
#        make_patches(
#                img_root='./Datasets/Balvan_1to4tiles', 
#                target_root='./Datasets/RIRE_patches',
#                fold=1,
#                t_level=i,
#                mode='train',
#                display=5)