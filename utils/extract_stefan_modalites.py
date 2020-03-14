# -*- coding: utf-8 -*-

import os, re, shutil
from tqdm import tqdm
#import image_slicer
# %%
DIR_TAR = './Datasets/Stefan/'

# %%
def extract_MPM_HE(root_path, dir_tar):
    ''' classify the files into two groups
    paths: list of paths
    groupA, groupB: str of group names
    patt_A, patt_B: str of re patterns
    '''
#    root_path = './MPMSkin/aligned_190508/'
    patt_MPM = root_path + r'.*[\\|/](\w+)[\\|/](\d+).tif'
    patt_HE = root_path + r'.*[\\|/](\w+)[\\|/].+(\b\d+).tif'
    pMPM = re.compile(patt_MPM)
    pHE = re.compile(patt_HE)
    
    dirMPM = dir_tar + 'MPM/'
    dirHE = dir_tar + 'HE/'
    if not os.path.exists(dirMPM):
        os.makedirs(dirMPM)
    if not os.path.exists(dirHE):
        os.makedirs(dirHE)
        
    paths = [os.path.join(root, file) for (root, dirs, files) in os.walk(root_path) for file in files]
    for f_path in tqdm(paths):
        matched = pMPM.match(f_path)
        if matched:
            (slide, region) = matched.groups()
            shutil.copyfile(f_path, dirMPM + f'{slide}_{region}_MPM.tif')
        else:
            matched = pHE.match(f_path)
            if matched:
                (slide, region) = matched.groups()
                shutil.copyfile(f_path, dirHE + f'{slide}_{region}_HE.tif')
    return

def extract_SHG(root_path, dir_tar):
#    root_path = './MPMSkin/'
    dirSHG = dir_tar + 'SHG/'
    patt_SHG = root_path + r'.*[\\|/](\w+)[\\|/]med[\\|/](\d+)-shg.tif'
    
    pSHG = re.compile(patt_SHG)
    if not os.path.exists(dirSHG):
        os.makedirs(dirSHG)
    
    dir_paths = ['DysplasticTissue', 'HealthyTissue', 'MalignantTissue']
    paths = []
    for dir_path in dir_paths:
        paths += [os.path.join(root, file) for (root, dirs, files) in os.walk(root_path + f'{dir_path}/') for file in files]

    for f_path in tqdm(paths):
        matched = pSHG.match(f_path)
        if matched:
            (slide, region) = matched.groups()
            shutil.copyfile(f_path, dirSHG + f'{slide}_{region}_SHG.tif')
    return

# %%
if __name__ == '__main__':
    extract_MPM_HE(root_path='./MPMSkin/aligned_190508/', dir_tar=DIR_TAR)
    extract_SHG(root_path='./MPMSkin/', dir_tar=DIR_TAR)
        
        