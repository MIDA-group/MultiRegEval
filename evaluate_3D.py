# -*- coding: utf-8 -*-
# evaluate registration error and write into csv
import pandas as pd
import skimage.io as skio
import skimage.transform as skt
import scipy.io as sio
from tqdm import tqdm
import os, cv2, argparse
import numpy as np
from glob import glob
from sklearn.decomposition import PCA
from skimage import exposure
import SimpleITK as sitk


# self-defined functions
from utils.make_eliceiri_patches import tform_centred
from utils.make_rire_patches import get_transform_rigid3D, dist_coords, transform_coords, resample_volume
from mi import register_mi_3D
from sift import register_sift
import sys
sys.path.append(os.path.abspath("./alpha_amd"))
import aamd

# %%
def tform_centred_rec(radian, translation, center):
    tform1 = skt.SimilarityTransform(translation=translation)
    tform2 = skt.SimilarityTransform(translation=center)
    tform3 = skt.SimilarityTransform(rotation=radian)
    tform4 = skt.SimilarityTransform(translation=-center)
    tform = tform4 + tform3 + tform2 + tform1
    return tform

# %%
def evaluate_methods(data_root, method, gan_name='', preprocess='nopre', mode='b2a', display=None):
    '''
    data_root: 
        should contains "{data_root}/A/test/" and "{data_root}/B/test/". 
        Corresponding images should have the same name.
    '''
#    data_root='./Datasets/RIRE_patches/fold1/patch_tlevel3/'
#    method='MI'
#    gan_name=''
#    data_root_fake='./Datasets/RIRE_patches_fake/fold1'
#    preprocess='nopre'
#    mode='b2a'
#    display=None

    
    # dataset-specific variables
    if 'RIRE' in data_root:
        img_root='../Datasets/RIRE'
        fold = data_root[data_root.rfind('fold') + len('fold')]
        data_root_fake=f'./Datasets/RIRE_patches_fake/fold{fold}'
        if 'MI' in method and method.replace('MI', '') != '':
            n_mi_res=int(method.replace('MI', ''))          # number of resolution level for MI
        else:
            n_mi_res=4
        n_aAMD_iters=0.3    # factor of number of iterations for aAMD
    
    
    dir_A = data_root + 'A/test/'
    dir_B = data_root + 'B/test/'
    
    if gan_name != '':
        assert data_root_fake, "data_root_fake must not be None when given gan_name."
        assert gan_name in ['cyc_A', 'cyc_B', 'p2p_A', 'p2p_B', 'drit_A', 'drit_B', 'star_A', 'star_B', 'comir'], (
                "gan_name must be in 'cyc_A', 'cyc_B', 'p2p_A', 'p2p_B', 'drit_A', 'drit_B', 'star_A', 'star_B', 'comir'")
        if 'comir' in gan_name:
            dir_A = f'{data_root_fake}/{os.path.split(data_root[:-1])[-1]}/{gan_name}_A/'
            dir_B = f'{data_root_fake}/{os.path.split(data_root[:-1])[-1]}/{gan_name}_B/'
        elif '_A' in gan_name:
            dir_B = f'{data_root_fake}/{os.path.split(data_root[:-1])[-1]}/{gan_name}/'
        elif '_B' in gan_name:
            dir_A = f'{data_root_fake}/{os.path.split(data_root[:-1])[-1]}/{gan_name}/'
    
    assert mode in ['a2b', 'b2a', 'a2a', 'b2b'], "mode must be in ['a2b', 'b2a', 'a2a', 'b2b']"
    if mode=='a2b':
        dir_src = dir_A
        dir_tar = dir_B
    elif mode=='b2a':
        dir_src = dir_B
        dir_tar = dir_A
    elif mode=='a2a':
        dir_src = dir_A
        dir_tar = dir_A
    elif mode=='b2b':
        dir_src = dir_B
        dir_tar = dir_B
        
    assert preprocess in ['', 'nopre', 'PCA', 'hiseq'], "preprocess must be in ['', 'nopre', 'PCA', 'hiseq']"
    

    suffix_src = '_' + os.listdir(dir_src)[0].split('_')[-1]
    name_srcs = set([name[:-len(suffix_src)] for name in os.listdir(dir_src)])
    suffix_tar = '_' + os.listdir(dir_tar)[0].split('_')[-1]
    name_tars = set([name[:-len(suffix_tar)] for name in os.listdir(dir_tar)])
    f_names = name_srcs & name_tars
    f_names = list(f_names)
    f_names.sort()
    f_names = [f_name for f_name in f_names if len(f_name) > len('patient_003')]

    df = pd.read_csv(data_root + 'info_test.csv', index_col='Filename')
    
    for f_name in tqdm(f_names):
        _, f_name, i = f_name.split('_')
        # extract reference and transformed patch coordinates
        coords_ref = df.loc[
                f'{f_name}_{i}', 
                ['X1_Ref', 'Y1_Ref', 'Z1_Ref', 
            'X2_Ref', 'Y2_Ref', 'Z2_Ref', 
            'X3_Ref', 'Y3_Ref', 'Z3_Ref', 
            'X4_Ref', 'Y4_Ref', 'Z4_Ref', 
            'X5_Ref', 'Y5_Ref', 'Z5_Ref', 
            'X6_Ref', 'Y6_Ref', 'Z6_Ref', 
            'X7_Ref', 'Y7_Ref', 'Z7_Ref', 
            'X8_Ref', 'Y8_Ref', 'Z8_Ref']
                ].to_numpy().reshape((8, 3))
        coords_trans = df.loc[
                f'{f_name}_{i}', 
                ['X1_Trans', 'Y1_Trans', 'Z1_Trans', 
            'X2_Trans', 'Y2_Trans', 'Z2_Trans', 
            'X3_Trans', 'Y3_Trans', 'Z3_Trans', 
            'X4_Trans', 'Y4_Trans', 'Z4_Trans', 
            'X5_Trans', 'Y5_Trans', 'Z5_Trans', 
            'X6_Trans', 'Y6_Trans', 'Z6_Trans', 
            'X7_Trans', 'Y7_Trans', 'Z7_Trans', 
            'X8_Trans', 'Y8_Trans', 'Z8_Trans']
                ].to_numpy().reshape((8, 3))
                    
            # load image (w, h)
#                img_grey = np.asarray((img_rgb[...,0] * 0.299 + img_rgb[...,1] * 0.587 + img_rgb[...,2] * 0.114), dtype=np.uint8)
        img_src = sitk.ReadImage(dir_src + f"patient_{f_name}_{i}_T.{suffix_src.split('.')[-1]}")
        img_tar = sitk.ReadImage(dir_tar + f"patient_{f_name}_R.{suffix_tar.split('.')[-1]}")

        if 'MI' in method:
            # register
            try:
                transformParameterMap = register_mi_3D(img_src, img_tar, n_res=n_mi_res)
            except:
                continue
            # transform the transformed patch coordinates back
            tform = get_transform_rigid3D(
                    parameters=[float(c) for c in transformParameterMap['TransformParameters']],
                    center=[float(c) for c in transformParameterMap['CenterOfRotationPoint']])
            coords_rec = transform_coords(coords_trans, tform)
        elif 'aAMD' in method:
        # TODO: not working
            if img_src.ndim == 2:
                img_src = np.expand_dims(img_src, axis=-1)
            if img_tar.ndim == 2:
                img_tar = np.expand_dims(img_tar, axis=-1)
            # register
            try:
                img_rec, t = aamd.register_aamd(ref_im=img_tar, flo_im=img_src, iterations=n_aAMD_iters)
            except:
                continue
            coords_rec = aamd.transform_coords(t, coords_in=coords_trans, centre_patch=centre_patch)
        elif 'SIFT' in method:
#                img_src = cv2.imread(dir_src + f"{f_name}_T.{suffix_src.split('.')[-1]}", 0)
#                img_tar = cv2.imread(dir_tar + f"{f_name}_R.{suffix_tar.split('.')[-1]}", 0)
            # register
            try:
                img_match, img_rec, tform = register_sift(img_src, img_tar)
            except:
                continue
            coords_rec = skt.matrix_transform(coords_trans, tform.params)
        
        # calculate error
        disp_error = dist_coords(coords_rec, coords_ref)
        
        result = {
                'X1_Recover': coords_rec[0][0], 'Y1_Recover': coords_rec[0][1], 'Z1_Recover': coords_rec[0][2], 
                'X2_Recover': coords_rec[1][0], 'Y2_Recover': coords_rec[1][1], 'Z2_Recover': coords_rec[1][2], 
                'X3_Recover': coords_rec[2][0], 'Y3_Recover': coords_rec[2][1], 'Z3_Recover': coords_rec[2][2], 
                'X4_Recover': coords_rec[3][0], 'Y4_Recover': coords_rec[3][1], 'Z4_Recover': coords_rec[3][2], 
                'X5_Recover': coords_rec[4][0], 'Y5_Recover': coords_rec[4][1], 'Z5_Recover': coords_rec[4][2], 
                'X6_Recover': coords_rec[5][0], 'Y6_Recover': coords_rec[5][1], 'Z6_Recover': coords_rec[5][2], 
                'X7_Recover': coords_rec[6][0], 'Y7_Recover': coords_rec[6][1], 'Z7_Recover': coords_rec[6][2], 
                'X8_Recover': coords_rec[7][0], 'Y8_Recover': coords_rec[7][1], 'Z8_Recover': coords_rec[7][2], 
            'Error': np.mean(disp_error)}
        # update result
        df.loc[f'{f_name}_{i}', 
               ['X1_Recover', 'Y1_Recover', 'X2_Recover', 'Y2_Recover', 
                'X3_Recover', 'Y3_Recover', 'X4_Recover', 'Y4_Recover', 
                'Error']
               ] = result

    result_dir = os.path.join(data_root, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    df.to_csv(f'{result_dir}/{method+gan_name}_{mode}_{preprocess}.csv')
    
    return

# %%
if __name__ == '__main__':
    # for running from terminal
    parser = argparse.ArgumentParser(description='Evaluate methods.')
    parser.add_argument(
            '--data_dir', '-d', 
            help="data_dir", 
            default='./Datasets/RIRE_patches/fold1/patch_tlevel3/')
    parser.add_argument(
            '--method', '-m', 
            help="registration method", 
            choices=['MI', 'SIFT', 'aAMD', 'CA'], 
            default='aAMD')
    parser.add_argument(
            '--gan', '-g', 
            help="gan method", 
            choices=['p2p_A', 'p2p_B', 'cyc_A', 'cyc_B', 'drit_A', 'drit_B', 'star_A', 'star_B', 'comir', ''], 
            default='')
    parser.add_argument(
            '--mode', 
            help="registration direction", 
            choices=['a2b', 'b2a', 'a2a', 'b2b'], 
            default='b2a')
    parser.add_argument(
            '--pre', 
            help="pre-processing method", 
            choices=['', 'nopre', 'PCA', 'hiseq'], 
            default='nopre')
    args = parser.parse_args()

    evaluate_methods(
            data_root=args.data_dir, 
            method=args.method, 
            gan_name=args.gan, 
            preprocess=args.pre,
            mode=args.mode, 
            display=None)

#    # local serial run
#    for data_dir in glob('./Datasets/RIRE_patches/fold1/patch_tlevel*/'):
#        for gan in ['p2p_A', 'p2p_B', 'cyc_A', 'cyc_B', 'drit_A', 'drit_B']:
#            evaluate_methods(
#                    data_root=data_dir, 
#                    method='SIFT', 
#                    gan_name=gan, 
#                    preprocess='nopre',
#                    mode='b2a', 
#                    display=None)

#    for dataset in ['Zurich', 'Balvan']:
#        for fold in [2, 3]:
#            for data_dir in glob(f'./Datasets/{dataset}_patches/fold{fold}/patch_tlevel*/'):
#                evaluate_methods(
#                        data_root=data_dir, 
#                        method='MI', 
#                        preprocess='nopre',
#                        mode='b2a', 
#                        display=None)
    

# %% Print out success rate
# =============================================================================
# ROOT_DIR='./Datasets/Balvan_patches/fold1'
# 
# def success_rate(method, gan_name='', preprocess='nopre', mode='b2a'):
#     # read results
#     dfs = [pd.read_csv(csv_path) for csv_path 
#            in glob(f'{ROOT_DIR}/patch_tlevel*/results/{method+gan_name}_{mode}_{preprocess}.csv')]
#     
#     whole_df = pd.concat(dfs)
#     n_success = whole_df['Error'][whole_df['Error'] <= 300*0.02].count()
#     rate_success = n_success / len(whole_df)
#     print(f'{method+gan_name}_{preprocess}', rate_success)
#     
# success_rate(method='MI')
# success_rate(method='CA')
# for method in ['SIFT', 'aAMD']:
#     for gan in ['', 'p2p_A', 'p2p_B', 'cyc_A', 'cyc_B', 'drit_A', 'drit_B']:
#         success_rate(method=method, gan_name=gan)
# success_rate(method='VXM', preprocess='su')
# success_rate(method='VXM', preprocess='us')
# 
# =============================================================================
