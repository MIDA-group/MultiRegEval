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


# self-defined functions
from utils.make_eliceiri_patches import dist_coords, tform_centred
#from mi import register_mi
from mi_ms import register_mi_ms
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
#    data_root='./Datasets/Balvan_patches/fold1/patch_tlevel3/'
#    method='MI'
#    gan_name=''
#    data_root_fake='./Datasets/Balvan_patches_fake/fold1'
#    preprocess='nopre'
#    mode='b2a'
#    display=None

    
    # dataset-specific variables
    if 'Eliceiri' in data_root:
        img_root='../Datasets/HighRes_Splits/WSI'
        w=834
        o=608
        data_root_fake='./Datasets/Eliceiri_patches_fake'
        n_mi_res=6          # number of resolution level for MI
        n_aAMD_iters=1.0    # factor of number of iterations for aAMD
    elif 'Balvan' in data_root:
        img_root='../Datasets/Balvan_1to4tiles'
        w=300 # patch width
        o=w//2 # offset: upper-left corner of patch
        fold = data_root[data_root.rfind('fold') + len('fold')]
        data_root_fake=f'./Datasets/Balvan_patches_fake/fold{fold}'
        if 'MI' in method and method.replace('MI', '') != '':
            n_mi_res=int(method.replace('MI', ''))          # number of resolution level for MI
        else:
            n_mi_res=4
        n_aAMD_iters=0.3    # factor of number of iterations for aAMD
    elif 'Zurich' in data_root:
        img_root='../Datasets/Zurich_tiles'
        w=300 # patch width
        o=w//2 # offset: upper-left corner of patch
        fold = data_root[data_root.rfind('fold') + len('fold')]
        data_root_fake=f'./Datasets/Zurich_patches_fake/fold{fold}'
        if 'MI' in method and method.replace('MI', '') != '':
            n_mi_res=int(method.replace('MI', ''))          # number of resolution level for MI
        else:
            n_mi_res=4
        n_aAMD_iters=0.3    # factor of number of iterations for aAMD
    
    coords_ref = np.array(([0,0], [0,w], [w,w], [w,0]))
    centre_patch = np.array((w, w)) / 2. - 0.5
    
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

    df = pd.read_csv(data_root + 'info_test.csv', index_col='Filename')
    
    cnt_disp = 0
    for f_name in tqdm(f_names):    
        # extract transformed patch coordinates
        coords_trans = df.loc[
                f_name, 
                ['X1_Trans', 'Y1_Trans', 'X2_Trans', 'Y2_Trans', 'X3_Trans', 'Y3_Trans', 'X4_Trans', 'Y4_Trans']
                ].to_numpy().reshape((4, 2))
                    
        if 'Eliceiri' in data_root and method == 'CA':  # CurveAlign
            tlevel = os.path.split(data_root[:-1])[-1][-1]
            try:
                t = sio.loadmat(f'./CurveAlign/eliceiri_4levels/HE/tform_{f_name}_l{tlevel}.mat')
            except:
                continue
            
            rot_matrix = np.asarray(t['t'])[:2, :2]
            translation_rec = np.asarray(t['t'])[-1, :-1]

            tform = np.eye(3)
            tform[:2, :2] = rot_matrix # rotation
            tform[:-1, -1] = translation_rec # translation
            tform = skt.SimilarityTransform(matrix=tform)
            
            tform_patch_rec = tform_centred_rec(radian=tform.rotation, translation=translation_rec, center=centre_patch)
            tform_patch_rec2 = tform_centred(radian=tform.rotation, translation=tform_patch_rec.translation, center=centre_patch)
            coords_rec = skt.matrix_transform(coords_trans, tform_patch_rec2.params)
            
        
        else:
            # PCA
            if preprocess == 'PCA':
                # load image with 3 channels
                img_src = cv2.imread(dir_src + f"{f_name}_T.{suffix_src.split('.')[-1]}")
                img_tar = cv2.imread(dir_tar + f"{f_name}_R.{suffix_tar.split('.')[-1]}")
                
                pixels_src = img_src.reshape(-1, img_src.shape[-1])
                result_src = PCA(n_components=1).fit_transform(pixels_src)
                img_src = result_src.reshape(img_src.shape[0], img_src.shape[1])
                pixels_tar = img_tar.reshape(-1, img_tar.shape[-1])
                result_tar = PCA(n_components=1).fit_transform(pixels_tar)
                img_tar = result_tar.reshape(img_tar.shape[0], img_tar.shape[1])
            else:
                # load image (w, h)
#                img_grey = np.asarray((img_rgb[...,0] * 0.299 + img_rgb[...,1] * 0.587 + img_rgb[...,2] * 0.114), dtype=np.uint8)
                img_src = cv2.imread(dir_src + f"{f_name}_T.{suffix_src.split('.')[-1]}", 0)
                img_tar = cv2.imread(dir_tar + f"{f_name}_R.{suffix_tar.split('.')[-1]}", 0)
                if 'Zurich' in data_root and 'comir' not in gan_name:   # because Zurich images are stored in BGR
                    if 'B' in dir_src:
                        img_src_rgb = cv2.imread(dir_src + f"{f_name}_T.{suffix_src.split('.')[-1]}")
                        img_src = np.asarray((img_src_rgb[...,0] * 0.299 + img_src_rgb[...,1] * 0.587 + img_src_rgb[...,2] * 0.114), dtype=np.uint8)
                    if 'B' in dir_tar:
                        img_tar_rgb = cv2.imread(dir_tar + f"{f_name}_R.{suffix_tar.split('.')[-1]}")
                        img_tar = np.asarray((img_tar_rgb[...,0] * 0.299 + img_tar_rgb[...,1] * 0.587 + img_tar_rgb[...,2] * 0.114), dtype=np.uint8)
                        
            # histogram equlisation
            if preprocess == 'hiseq':
    #            print(img_src.shape, img_tar.shape)
                img_src = exposure.equalize_adapthist(img_src, clip_limit=0.03)
                img_tar = exposure.equalize_adapthist(img_tar, clip_limit=0.03)
    
            if 'MI' in method:
                # register
                try:
#                    img_rec, field = register_mi(img_src, img_tar, n_res=n_mi_res)
                    field, _ = register_mi_ms(img_src, img_tar, data_root, n_res=n_mi_res)
                except:
                    continue
                rot_radian = float(field[0])
                tx = float(field[1])
                ty = float(field[2])
                # transform the transformed patch coordinates back
                tform_patch_rec = tform_centred_rec(radian=rot_radian, translation=(tx, ty), center=centre_patch)
                coords_rec = skt.matrix_transform(coords_trans, tform_patch_rec.params)
            elif 'aAMD' in method:
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
            'X1_Recover': coords_rec[0][0], 'Y1_Recover': coords_rec[0][1], 
            'X2_Recover': coords_rec[1][0], 'Y2_Recover': coords_rec[1][1], 
            'X3_Recover': coords_rec[2][0], 'Y3_Recover': coords_rec[2][1], 
            'X4_Recover': coords_rec[3][0], 'Y4_Recover': coords_rec[3][1], 
            'Error': np.mean(disp_error)}
        # update result
        df.loc[f_name, 
               ['X1_Recover', 'Y1_Recover', 'X2_Recover', 'Y2_Recover', 
                'X3_Recover', 'Y3_Recover', 'X4_Recover', 'Y4_Recover', 
                'Error']
               ] = result


        # display patch outline in original image
        # TODO: not working
        if display:
            if cnt_disp < display:               
                suffix = os.path.basename(os.listdir(f'{img_root}/A/')[0]).split('.')[-1]
                imgB = skio.imread(f"{img_root}/B/{f_name}.{suffix}")
                dispdirB = f'{data_root}/display/B/test'
                if not os.path.exists(dispdirB):
                    os.makedirs(dispdirB)
                if 'Eliceiri' in data_root:
                    imgB_disp = imgB
                elif 'Balvan' in data_root:
                    imgB_disp = np.pad(imgB, w//2, mode='reflect')
                if len(imgB_disp.shape) == 2:
                    imgB_disp = np.repeat(imgB_disp.reshape(imgB_disp.shape[0], imgB_disp.shape[1], 1), 3, axis=-1)
                imgB_disp = cv2.polylines(imgB_disp, pts=[(o+coords_ref).reshape((-1,1,2))], isClosed=True, color=(0,255,0), thickness=2)
                imgB_disp = cv2.polylines(imgB_disp, pts=[np.int32(o+coords_trans).reshape((-1,1,2))], isClosed=True, color=(0,0,255), thickness=2)
                imgB_disp = cv2.polylines(imgB_disp, pts=[np.int32(o+coords_rec).reshape((-1,1,2))], isClosed=True, color=(255,0,0), thickness=2)
#                skio.imshow(imgB_disp)
                skio.imsave(f'{dispdirB}/{f_name}_{method+gan_name}_{mode}_{preprocess}.{suffix}', imgB_disp)
                cnt_disp += 1
            else:
                return

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
            default='./Datasets/Balvan_patches/fold1/patch_tlevel3/')
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
#    for data_dir in glob('./Datasets/Zurich_patches/fold1/patch_tlevel*/'):
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
