# -*- coding: utf-8 -*-
# make Eliceiri data from WSIs to patches
import pandas as pd
import skimage.io as skio
import skimage.transform as skt
from tqdm import tqdm
from glob import glob
import os, random, math, cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
def tform_centred(radian, translation, center):
    # first translation, then rotation
    tform1 = skt.SimilarityTransform(translation=center)
    tform2 = skt.SimilarityTransform(rotation=radian)
    tform3 = skt.SimilarityTransform(translation=-center)
    tform4 = skt.SimilarityTransform(translation=translation)
    tform = tform4 + tform3 + tform2 + tform1
    return tform

def dist_coords(coords1, coords2):
    ''' Calculate the point-to-point distances between two coordinates, return a list.
    '''
    return [sum((coords1[i] - coords2[i]) ** 2) ** 0.5 for i in range(len(coords1))]
    
# %%
def make_patches(img_root, target_root, fold=None, t_level=1, 
#                 trans_min=0, trans_max=20, rot_min=0, rot_max=5, 
                 mode='train', display=None):
#    img_root='../Datasets/HighRes_Splits/WSI'
#    target_root='../Datasets/Eliceiri_patches'
#    trans_min=0
#    trans_max=20
#    rot_min=0
#    rot_max=0
#    mode='train'
    
    w=834
    o=608
    
    coords_ref = np.array(([0,0], [0,w], [w,w], [w,0]))
    centre_patch = np.array((w, w)) / 2. - 0.5

    step_trans = 20
    step_rot = 5
    trans_min = step_trans * (t_level - 1)
    trans_max = step_trans * t_level
    rot_min = step_rot * (t_level - 1)
    rot_max = step_rot * t_level
    
    #modalities = {'MI':'SHG', 'WB':'BF'}
    tardirA = f'{target_root}/patch_tlevel{t_level}/A/{mode}'
    tardirB = f'{target_root}/patch_tlevel{t_level}/B/{mode}'
    if not os.path.exists(tardirA):
        os.makedirs(tardirA)
    if not os.path.exists(tardirB):
        os.makedirs(tardirB)
    
    f_names = set(['_'.join(name.split('_')[:-1]) for name in os.listdir(f'{img_root}/{mode}')])
    f_names = list(f_names)
    f_names.sort()
    
    # csv information
    header = [
            'ReferenceImage', 'Method', 
            'X1_Ref', 'Y1_Ref', 'X2_Ref', 'Y2_Ref', 'X3_Ref', 'Y3_Ref', 'X4_Ref', 'Y4_Ref', 
            'X1_Trans', 'Y1_Trans', 'X2_Trans', 'Y2_Trans', 'X3_Trans', 'Y3_Trans', 'X4_Trans', 'Y4_Trans', 
            'X1_Recover', 'Y1_Recover', 'X2_Recover', 'Y2_Recover', 'X3_Recover', 'Y3_Recover', 'X4_Recover', 'Y4_Recover', 
            'Displacement', 'Tx', 'Ty', 'AngleDegree', 'AngleRad', 'Error', 'DisplacementCategory']
    df = pd.DataFrame(index=f_names, columns=header)
    df.index.set_names('Filename', inplace=True)
    
    if display is not None:
        cnt_disp = 0
    for f_name in tqdm(f_names):
        # load WSI
        if mode == 'train':
            f_nameA = f_name + '_MI.tif'
            f_nameB = f_name + '_WB.tif'
        else:
            f_nameA = f_name + '_SHG.tif'
            f_nameB = f_name + '_BF.tif'
        imgA = skio.imread(f'{img_root}/{mode}/{f_nameA}')
        imgB = skio.imread(f'{img_root}/{mode}/{f_nameB}')
        
        # random transformation parameters
        rot_degree = random.choice((random.uniform(-rot_max, -rot_min), random.uniform(rot_min, rot_max)))
        tx = random.choice((random.uniform(-trans_max, -trans_min), random.uniform(trans_min, trans_max)))
        ty = random.choice((random.uniform(-trans_max, -trans_min), random.uniform(trans_min, trans_max)))
        rot_radian = np.deg2rad(rot_degree)
                
        # transform WSI
        centre_img = np.array((imgA.shape[0], imgA.shape[1])) / 2. - 0.5
        tform_img = tform_centred(radian=rot_radian, translation=(tx, ty), center=centre_img)
        imgA_trans = np.asarray(skt.warp(imgA, tform_img, preserve_range=True), dtype=np.uint8)
        imgB_trans = np.asarray(skt.warp(imgB, tform_img, preserve_range=True), dtype=np.uint8)
        
        # crop patches
        patchA_ref = imgA[o:o+w, o:o+w]
        patchB_ref = imgB[o:o+w, o:o+w]
        patchA_trans = imgA_trans[o:o+w, o:o+w]
        patchB_trans = imgB_trans[o:o+w, o:o+w]
        
        # transform patch coordinates
        tform_patch = tform_centred(radian=rot_radian, translation=(tx, ty), center=centre_patch)
        coords_trans = skt.matrix_transform(coords_ref, tform_patch.params)
        
        # calculate distance
        dist = dist_coords(coords_trans, coords_ref)
        
        # write csv line
        line = {
            'X1_Ref': coords_ref[0][0], 'Y1_Ref': coords_ref[0][1], 
            'X2_Ref': coords_ref[1][0], 'Y2_Ref': coords_ref[1][1], 
            'X3_Ref': coords_ref[2][0], 'Y3_Ref': coords_ref[2][1], 
            'X4_Ref': coords_ref[3][0], 'Y4_Ref': coords_ref[3][1], 
            'X1_Trans': coords_trans[0][0], 'Y1_Trans': coords_trans[0][1], 
            'X2_Trans': coords_trans[1][0], 'Y2_Trans': coords_trans[1][1], 
            'X3_Trans': coords_trans[2][0], 'Y3_Trans': coords_trans[2][1], 
            'X4_Trans': coords_trans[3][0], 'Y4_Trans': coords_trans[3][1], 
            'Displacement': np.mean(dist), 
            'Tx': tx, 'Ty': ty, 
            'AngleDegree': rot_degree, 'AngleRad': rot_radian}
        df.loc[f_name] = line
        
        # save patches
        skio.imsave(f'{tardirA}/{f_name}_R.tif', patchA_ref)
        skio.imsave(f'{tardirA}/{f_name}_T.tif', patchA_trans)
        skio.imsave(f'{tardirB}/{f_name}_R.tif', patchB_ref)
        skio.imsave(f'{tardirB}/{f_name}_T.tif', patchB_trans)
        
        # display patch outline in original image
        if display is not None and cnt_disp < display:
            dispdirA = f'{target_root}/patch_tlevel{t_level}/display/A/{mode}'
            dispdirB = f'{target_root}/patch_tlevel{t_level}/display/B/{mode}'
            if not os.path.exists(dispdirA):
                os.makedirs(dispdirA)
            if not os.path.exists(dispdirB):
                os.makedirs(dispdirB)
            imgA = np.repeat(imgA.reshape(imgA.shape[0], imgA.shape[1], 1), 3, axis=-1)
            imgA = cv2.polylines(imgA, pts=[(o+coords_ref).reshape((-1,1,2))], isClosed=True, color=(0,255,0), thickness=3)
            imgA = cv2.polylines(imgA, pts=[np.int32(o+coords_trans).reshape((-1,1,2))], isClosed=True, color=(0,0,255), thickness=3)
            skio.imsave(f'{dispdirA}/{f_name}_display.tif', imgA)
            imgB = cv2.polylines(imgB, pts=[(o+coords_ref).reshape((-1,1,2))], isClosed=True, color=(0,255,0), thickness=3)
            imgB = cv2.polylines(imgB, pts=[np.int32(o+coords_trans).reshape((-1,1,2))], isClosed=True, color=(0,0,255), thickness=3)
#            imgB = cv2.polylines(imgB, pts=[np.int32(o+coords_rec).reshape((-1,1,2))], isClosed=True, color=(255,0,0), thickness=3)
            skio.imsave(f'{dispdirB}/{f_name}_display.tif', imgB)
            cnt_disp += 1
    
    df.to_csv(f'{target_root}/patch_tlevel{t_level}/info_{mode}.csv')

# %%
if __name__ == '__main__':
#    trans_mins = list(range(0, 80, 20))
#    trans_maxs = list(range(20, 100, 20))
#    rot_mins = list(range(0, 20, 5))
#    rot_maxs = list(range(5, 25, 5))
    for i in range(1, 5):
        make_patches(
                img_root='./Datasets/HighRes_Splits/WSI', 
                target_root='./Datasets/Eliceiri_patches',
#                fold=None,
                t_level=i,
                mode='test',
                display=5)
#        make_patches(
#                img_root='./Datasets/HighRes_Splits/WSI', 
#                target_root='./Datasets/Eliceiri_patches',
#                fold=None,
#                t_level=i,
#                mode='train',
#                display=5)