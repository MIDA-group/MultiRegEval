# -*- coding: utf-8 -*-
# make Zurich data to patches
import pandas as pd
import skimage.io as skio
import skimage.transform as skt
from tqdm import tqdm
from glob import glob
import os, random, math, cv2
import numpy as np
import matplotlib.pyplot as plt


# %% Split data
def split_zurich_data(fold):
    if fold == 1:
        ids_test = {7, 9, 20, 3, 15, 18}
    elif fold == 2:
        ids_test = {10, 1, 13, 4, 11, 6, 16}
    elif fold == 3:
        ids_test = {14, 8, 17, 5, 19, 12, 2}
    ids_train = set(range(1, 21)) - ids_test
    return list(ids_train), list(ids_test)

# %% Helper functions
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
    
# =============================================================================
# # %% Cut 1 image to 4 patches (ONLY NEED TO RUN ONCE)
# import image_slicer
# 
# def slice_to_tiles(src_dir, tar_dir, w=300):
#     ''' Slice each image in src_dir to tiles and save to tar_dir
#     w: tile width
#     '''
# #    src_dir='./Datasets/Zurich'
# #    tar_dir='./Datasets/Zurich_tiles'
# #    w=300
#     
# #    ns_tiles = {}
#     
#     modalities = {'IR':'A', 'RGB':'B'}
#     for modality in os.listdir(src_dir):
#         if not os.path.exists(f'{tar_dir}/{modalities[modality]}'):
#             os.makedirs(f'{tar_dir}/{modalities[modality]}')
#         img_paths = glob(f'{src_dir}/{modality}/*.png')
#         for img_path in tqdm(img_paths):
#             img = skio.imread(img_path)
#             (rows, cols) = (img.shape[0] // w, img.shape[1] // w)
# #            ns_tiles[int(os.path.basename(img_path).split('.')[0].split('zh')[-1])] = rows*cols
#             tiles = image_slicer.slice(img_path, row=rows, col=cols, save=False)
#             for tile in tiles:
#                 (x_tile, y_tile) = tile.image.size
#                 tile.image = tile.image.crop(((x_tile-w)//2, (y_tile-w)//2, (x_tile+w)//2, (y_tile+w)//2))
#             image_slicer.save_tiles(tiles, 
#                                     directory=f'{tar_dir}/{modalities[modality]}', 
#                                     prefix=os.path.basename(img_path).split('.')[0], 
#                                     format='png')
#     return
# 
# slice_to_tiles(src_dir='./Datasets/Zurich', tar_dir='./Datasets/Zurich_tiles') 
# 
# =============================================================================
# %%
def make_patches(img_root, target_root, fold=1, t_level=1, 
                 mode='train', display=None):
#    img_root='./Datasets/Zurich_tiles'
#    target_root='./Datasets/Zurich_patches'
#    fold=1
#    t_level=2
#    mode='test'
    
    w=300 # patch width
    o=0 # upper-left corner of patch
    
    coords_ref = np.array(([0,0], [0,w], [w,w], [w,0]))
    centre_patch = np.array((w, w)) / 2. - 0.5
    
    step_trans = 7
    step_rot = 5
    trans_min = step_trans * (t_level - 1)
    trans_max = step_trans * t_level
    rot_min = step_rot * (t_level - 1)
    rot_max = step_rot * t_level
    
    
    #modalities = {'IR':'A', 'RGB':'B'}
    tardirA = f'{target_root}/fold{fold}/patch_tlevel{t_level}/A/{mode}'
    tardirB = f'{target_root}/fold{fold}/patch_tlevel{t_level}/B/{mode}'
    if not os.path.exists(tardirA):
        os.makedirs(tardirA)
    if not os.path.exists(tardirB):
        os.makedirs(tardirB)
    
    ids_train, ids_test = split_zurich_data(fold)
    
    if mode=='train':
        f_names = [os.path.basename(f_path).split('.')[0] for 
                   id_img in ids_train for 
                   f_path in glob(f'{img_root}/A/zh{id_img}_*')]
    elif mode=='test':
        f_names = [os.path.basename(f_path).split('.')[0] for 
                   id_img in ids_test for 
                   f_path in glob(f'{img_root}/A/zh{id_img}_*')]
    
    f_names = list(f_names)
    f_names.sort()
    
    # csv information
    header = [
            'ReferenceImage', 'Method', 
            'X1_Ref', 'Y1_Ref', 'X2_Ref', 'Y2_Ref', 'X3_Ref', 'Y3_Ref', 'X4_Ref', 'Y4_Ref', 
            'X1_Trans', 'Y1_Trans', 'X2_Trans', 'Y2_Trans', 'X3_Trans', 'Y3_Trans', 'X4_Trans', 'Y4_Trans', 
            'X1_Recover', 'Y1_Recover', 'X2_Recover', 'Y2_Recover', 'X3_Recover', 'Y3_Recover', 'X4_Recover', 'Y4_Recover', 
            'Displacement', 'RelativeDisplacement', 'Tx', 'Ty', 'AngleDegree', 'AngleRad', 'Error', 'DisplacementCategory']
    df = pd.DataFrame(index=f_names, columns=header)
    df.index.set_names('Filename', inplace=True)
    
    if display is not None:
        cnt_disp = 0
    for f_name in tqdm(f_names):
        # load original images
        suffix = os.path.basename(os.listdir(f'{img_root}/A/')[0]).split('.')[-1]
        imgA = skio.imread(f"{img_root}/A/{f_name}.{suffix}")
        imgB = skio.imread(f"{img_root}/B/{f_name}.{suffix}")
        
        # random transformation parameters
        rot_degree = random.choice((random.uniform(-rot_max, -rot_min), random.uniform(rot_min, rot_max)))
        tx = random.choice((random.uniform(-trans_max, -trans_min), random.uniform(trans_min, trans_max)))
        ty = random.choice((random.uniform(-trans_max, -trans_min), random.uniform(trans_min, trans_max)))
        rot_radian = np.deg2rad(rot_degree)
                
        # transform original images
        centre_img = np.array((imgA.shape[0], imgA.shape[1])) / 2. - 0.5
        tform_img = tform_centred(radian=rot_radian, translation=(tx, ty), center=centre_img)
        imgA_trans = np.asarray(skt.warp(imgA, tform_img, mode='reflect', preserve_range=True), dtype=np.uint8)
        imgB_trans = np.asarray(skt.warp(imgB, tform_img, mode='reflect', preserve_range=True), dtype=np.uint8)
        
        # crop patches
        patchA_ref = imgA[o:o+w, o:o+w]
        patchB_ref = imgB[o:o+w, o:o+w]
        patchA_trans = imgA_trans[o:o+w, o:o+w]
        patchB_trans = imgB_trans[o:o+w, o:o+w]
        
        # transform patch coordinates
        tform_patch = tform_centred(radian=rot_radian, translation=(tx, ty), center=centre_patch)
        coords_trans = skt.matrix_transform(coords_ref, tform_patch.params)
        
        # calculate distance
        dist_array = dist_coords(coords_trans, coords_ref)
        dist = np.mean(dist_array)
        
        
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
            'Displacement': dist, 
            'RelativeDisplacement': dist/w,
            'Tx': tx, 'Ty': ty, 
            'AngleDegree': rot_degree, 'AngleRad': rot_radian}
        df.loc[f_name] = line
        
        # save patches
        skio.imsave(f'{tardirA}/{f_name}_R.{suffix}', patchA_ref)
        skio.imsave(f'{tardirA}/{f_name}_T.{suffix}', patchA_trans)
        skio.imsave(f'{tardirB}/{f_name}_R.{suffix}', patchB_ref)
        skio.imsave(f'{tardirB}/{f_name}_T.{suffix}', patchB_trans)
        
        # display patch outline in original image
        if display is not None and cnt_disp < display:
            dispdirA = f'{target_root}/fold{fold}/patch_tlevel{t_level}/display/A/{mode}'
            dispdirB = f'{target_root}/fold{fold}/patch_tlevel{t_level}/display/B/{mode}'
            if not os.path.exists(dispdirA):
                os.makedirs(dispdirA)
            if not os.path.exists(dispdirB):
                os.makedirs(dispdirB)
            if len(imgA.shape) == 2:
                imgA_disp = np.pad(imgA, w//2, mode='reflect')
                imgA_disp = np.repeat(imgA_disp.reshape(imgA_disp.shape[0], imgA_disp.shape[1], 1), 3, axis=-1)
            else:
                imgA_disp = np.pad(imgA, ((w//2, w//2), (w//2, w//2), (0, 0)), mode='reflect')
            imgA_disp = cv2.polylines(imgA_disp, pts=[(w//2+coords_ref).reshape((-1,1,2))], isClosed=True, color=(0,255,0), thickness=2)
            imgA_disp = cv2.polylines(imgA_disp, pts=[np.int32(w//2+coords_trans).reshape((-1,1,2))], isClosed=True, color=(0,0,255), thickness=2)
            skio.imsave(f'{dispdirA}/{f_name}_display.{suffix}', imgA_disp)
            if len(imgB.shape) == 2:
                imgB_disp = np.pad(imgB, w//2, mode='reflect')
                imgB_disp = np.repeat(imgB_disp.reshape(imgB_disp.shape[0], imgB_disp.shape[1], 1), 3, axis=-1)
            else:
                imgB_disp = np.pad(imgB, ((w//2, w//2), (w//2, w//2), (0, 0)), mode='reflect')
            imgB_disp = cv2.polylines(imgB_disp, pts=[(w//2+coords_ref).reshape((-1,1,2))], isClosed=True, color=(0,255,0), thickness=2)
            imgB_disp = cv2.polylines(imgB_disp, pts=[np.int32(w//2+coords_trans).reshape((-1,1,2))], isClosed=True, color=(0,0,255), thickness=2)
#            imgB_disp = cv2.polylines(imgB_disp, pts=[np.int32(w//2+coords_rec).reshape((-1,1,2))], isClosed=True, color=(255,0,0), thickness=2)
            skio.imsave(f'{dispdirB}/{f_name}_display.{suffix}', imgB_disp)
            cnt_disp += 1
    
    df.to_csv(f'{target_root}/fold{fold}/patch_tlevel{t_level}/info_{mode}.csv')

# %%
if __name__ == '__main__':
#    trans_mins = list(range(0, 28, 7))
#    trans_maxs = list(range(7, 35, 7))
#    rot_mins = list(range(0, 20, 5))
#    rot_maxs = list(range(5, 25, 5))
    for i in range(1, 5):
        make_patches(
                img_root='./Datasets/Zurich_tiles', 
                target_root='./Datasets/Zurich_patches',
                fold=1,
                t_level=i,
                mode='test',
                display=5)
#        make_patches(
#                img_root='./Datasets/Zurich_tiles', 
#                target_root='./Datasets/Zurich_patches',
#                fold=1,
#                t_level=i,
#                mode='train',
#                display=5)