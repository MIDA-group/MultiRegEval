# -*- coding: utf-8 -*-

import os, cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import skimage.io as skio
import SimpleITK as sitk


# %% Stefan testing
#DATA_ROOT = './Datasets/Stefan/'
#img1 = cv2.imread(DATA_ROOT+'HE/146558_2_HE.tif', 0)
##img_MPM = cv2.imread(DATA_ROOT+'MPM/146558_2_MPM.tif',0)
#img2 = cv2.imread(DATA_ROOT+'SHG/146558_2_SHG.tif', 0)
#img2 = np.rot90(img2, k=3)
#img2 = cv2.normalize(src=img2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# %%
# SimpleElastix
def register_mi(img1, img2, n_res=7):
#    img1 = cv2.imread('./Datasets/Eliceiri_patches/patch_trans10-20_rot10-15/B/test/1B_A7_T.tif', 0)
#    img2 = cv2.imread('./Datasets/Eliceiri_patches/patch_trans10-20_rot10-15/A/test/1B_A7_R.tif', 0)
    
    img1 = sitk.GetImageFromArray(img1)
    img2 = sitk.GetImageFromArray(img2)
        
    parameterMap = sitk.GetDefaultParameterMap('rigid')
    parameterMap['ResultImagePixelType'] = ['uint8']
    parameterMap['NumberOfResolutions'] = [str(n_res)]
    parameterMap['MaximumNumberOfIterations'] = ['1024']
    
    #parameterMap['Optimizer'] = ['AdaGrad']
    
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetLogToConsole(False)
    elastixImageFilter.SetFixedImage(img2)
    elastixImageFilter.SetMovingImage(img1)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute()
    
    resultImage = elastixImageFilter.GetResultImage()
    img1Reg = sitk.GetArrayFromImage(resultImage).astype('uint8')
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0].asdict()
    tform_parameter = transformParameterMap['TransformParameters']
#    radian = float(tform_parameter[0])
#    tx = float(tform_parameter[1])
#    ty = float(tform_parameter[2])
#    skio.imshow(img1Reg)
    return img1Reg, tform_parameter

# %%

def overlay(img1, img2):
    ''' Overlay img1 to R, img2 to G
    '''
    h, w = img1.shape[:2]
    _img = np.zeros((h, w, 3), dtype='uint8')
    _img[:,:,0]=img1 # R
    img1_r = _img
    _img = np.zeros((h, w, 3), dtype='uint8')
    _img[:,:,1]=img2 # g
    img2_g = _img
    overlay = cv2.addWeighted(img1_r,0.5, img2_g,0.5, 0)
    return overlay

def register_mi_batch_stefan(data_root, target_dir):
#    target_dir='./outputs/MI/Stefan/'
    dirA = data_root + 'HE/'
    dirB = data_root + 'SHG/'
    
    dir_matches = target_dir + 'matches'
    dir_results = target_dir + 'results'
    if not os.path.exists(dir_matches):
        os.makedirs(dir_matches)
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    
    suffixA = '_' + os.listdir(dirA)[0].split('_')[-1]
    nameAs = set([name[:-len(suffixA)] for name in os.listdir(dirA)])
    suffixB = '_' + os.listdir(dirB)[0].split('_')[-1]
    nameBs = set([name[:-len(suffixB)] for name in os.listdir(dirB)])
    f_names = nameAs & nameBs

    for f_name in tqdm(f_names):
        f_nameA = f_name + suffixA
        f_nameB = f_name + suffixB
        imgA = cv2.imread(dirA + f_nameA, 0)
        imgB = cv2.imread(dirB + f_nameB, 0)
        imgB = np.rot90(imgB, k=3)
        imgB = cv2.normalize(src=imgB, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        try:
            imgA2, field = register_mi(imgA, imgB)
            imgA2 = imgA2.astype('uint8')
            img_match = np.concatenate([overlay(imgA, imgB), overlay(imgA2, imgB)], axis=1)
        except:
            heightA, widthA = imgA.shape[:2]
            imgA2 = np.zeros((heightA, widthA))
            img_match = np.zeros((heightA, widthA))
                    
        skio.imsave(f'{dir_matches}/{f_name}.tif', img_match)
        skio.imsave(f'{dir_results}/{f_name}_HEreg.tif', imgA2)
    return

def register_mi_batch_eliceiri(data_root, target_dir, mode='a2b'):
#    data_root='./Datasets/Eliceiri_test/processed/'
#    target_dir = './outputs/SIFT/Eliceiri/'
    assert mode in ['a2b', 'b2a'], "mode must be in ['a2b', 'b2a']"
    dirA = data_root + 'A/test/'
    dirB = data_root + 'B/test/'
    
    dir_matches = target_dir + 'matches'
    dir_results = target_dir + 'results'
    if not os.path.exists(dir_matches):
        os.makedirs(dir_matches)
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    
    suffixA = '_' + os.listdir(dirA)[0].split('_')[-1]
    nameAs = set([name[:-len(suffixA)] for name in os.listdir(dirA)])
    suffixB = '_' + os.listdir(dirB)[0].split('_')[-1]
    nameBs = set([name[:-len(suffixB)] for name in os.listdir(dirB)])
    f_names = nameAs & nameBs

    if mode=='a2b':
        for f_name in tqdm(f_names):
            f_nameA = f_name + '_T.tif'
            f_nameB = f_name + '_R.tif'
            imgA = cv2.imread(dirA + f_nameA, 0)
            imgB = cv2.imread(dirB + f_nameB, 0)
            
            try:
                imgA2, field = register_mi(imgA, imgB)
                imgA2 = imgA2.astype('uint8')
                img_match = np.concatenate([overlay(imgB, imgA), overlay(imgB, imgA2)], axis=1)
            except:
                heightA, widthA = imgA.shape[:2]
                imgA2 = np.zeros((heightA, widthA))
                img_match = np.zeros((heightA, widthA))
            
            skio.imsave(f'{dir_matches}/{f_name}.tif', img_match)
            skio.imsave(f'{dir_results}/{f_name}_Areg.tif', imgA2)
    elif mode=='b2a':
        for f_name in tqdm(f_names):
            f_nameA = f_name + '_R.tif'
            f_nameB = f_name + '_T.tif'
            imgA = cv2.imread(dirA + f_nameA, 0)
            imgB = cv2.imread(dirB + f_nameB, 0)
            
            try:
                imgB2, field = register_mi(imgB, imgA)
                imgB2 = imgB2.astype('uint8')
                img_match = np.concatenate([overlay(imgB, imgA), overlay(imgB2, imgA)], axis=1)
            except:
                heightB, widthB = imgB.shape[:2]
                imgB2 = np.zeros((heightB, widthB))
                img_match = np.zeros((heightB, widthB))
            
            skio.imsave(f'{dir_matches}/{f_name}.tif', img_match)
            skio.imsave(f'{dir_results}/{f_name}_Breg.tif', imgB2)
    return

# %%
if __name__ == '__main__':

    register_mi_batch_stefan(data_root='./Datasets/Stefan/', target_dir='./outputs/MI/')
    
    register_mi_batch_eliceiri(
            data_root='./Datasets/Eliceiri_patches/patch_trans10_rot5/', 
            target_dir='./outputs/MI/Eliceiri_trans10_rot5_a2b/',
            mode='a2b')
    register_mi_batch_eliceiri(
            data_root='./Datasets/Eliceiri_patches/patch_trans10-20_rot10-15/', 
            target_dir='./outputs/MI/Eliceiri_trans10-20_rot10-15_b2a/',
            mode='b2a')
