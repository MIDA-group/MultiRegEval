# -*- coding: utf-8 -*-
# make plots from csv data
import pandas as pd
import os, cv2, random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import skimage.io as skio
import skimage.util as sku
from skimage import exposure

# %%
def channelwise_rescale(img_src, percentiles):
    img_tar = np.zeros(img_src.shape, dtype=np.uint8)
    assert len(img_src.shape) == 3
    channels = img_src.shape[-1]
    assert channels == len(percentiles), "len(percentiles) must match number of channels."
    for c in range(channels):
        img_tar[..., c] = exposure.rescale_intensity(img_src[..., c], in_range=tuple(percentiles[c]))
    return img_tar

    
# %%
src_dir = './Datasets/Zurich_dataset_v1.0/images_tif'
tar_dir = './Datasets/Zurich'


dir_IR = f'{tar_dir}/IR'
dir_RGB = f'{tar_dir}/RGB'
if not os.path.exists(dir_IR):
    os.makedirs(dir_IR)
if not os.path.exists(dir_RGB):
    os.makedirs(dir_RGB)

id_tif = 11

img_name = f'zh{id_tif}.png'    
img_BGR = skio.imread(f'{dir_RGB}/{img_name}')
img_RGB = img_BGR[..., ::-1]


p1, p99 = np.percentile(img_RGB[..., 0], (1, 99))
img_en = np.zeros(img_RGB.shape, dtype=np.uint8)
for c in range(img_RGB.shape[-1]):
    p1, p99 = np.percentile(img_RGB[..., c], (1, 99))
    img_en[..., c] = exposure.rescale_intensity(img_RGB[..., c], in_range=(p1, p99))


# %%
def result_montage(dataset, n=3):
#    dataset='Zurich'
#    modality='A'
#    n=3
    assert dataset in ['Balvan', 'Eliceiri', 'Zurich'], "dataset must be in ['Balvan', 'Eliceiri', 'Zurich']"
    if dataset == 'Eliceiri':
        dataroot_real = f'./Datasets/{dataset}_patches/patch_tlevel1'
        dataroot_fake = f'./Datasets/{dataset}_patches_fake/patch_tlevel1'
        sample_name = '1B_A2'
        sample_path = {
                'A':f'./Datasets/HighRes_Splits/WSI/train/{sample_name}_MI.tif', 
                'B':f'./Datasets/HighRes_Splits/WSI/train/{sample_name}_WB.tif'}
        intro_examples = {
                'realA':'./Datasets/Eliceiri_patches/patch_tlevel4/A/test/1B_E2_R.tif',
                'realB':'./Datasets/Eliceiri_patches/patch_tlevel4/B/test/1B_E2_T.tif',
                'fakeA':'./Datasets/Eliceiri_patches_fake/patch_tlevel4/p2p_A/1B_E2_T.png',
                'fakeB':'./Datasets/Eliceiri_patches_fake/patch_tlevel4/star_B/1B_E2_R.png'}
        
    else:
        dataroot_real = f'./Datasets/{dataset}_patches/fold{{fold}}/patch_tlevel1'
        dataroot_fake = f'./Datasets/{dataset}_patches_fake/fold{{fold}}/patch_tlevel1'
        if dataset == 'Zurich':
            sample_name = 'zh11'
            sample_path = {
                    'A':f'./Datasets/Zurich/IR/{sample_name}.png', 
                    'B':f'./Datasets/Zurich/RGB/{sample_name}.png'}
        elif dataset == 'Balvan':
            sample_name = 'DU145_do_2_f60'
            sample_path = {
                    'A':f'./Datasets/Balvan/GREEN/{sample_name}.tif', 
                    'B':f'./Datasets/Balvan/QPI/{sample_name}.tif'}
            intro_examples = {
                    'realA':'./Datasets/Balvan_patches/fold2/patch_tlevel4/A/test/PNT1A_st_3_f00_01_02_R.png',
                    'realB':'./Datasets/Balvan_patches/fold2/patch_tlevel4/B/test/PNT1A_st_3_f00_01_02_T.png',
                    'fakeA':'./Datasets/Balvan_patches_fake/fold2/patch_tlevel4/cyc_A/PNT1A_st_3_f00_01_02_T.png',
                    'fakeB':'./Datasets/Balvan_patches_fake/fold2/patch_tlevel4/cyc_B/PNT1A_st_3_f00_01_02_R.png'}

    def imread(img_path, dataset):
        assert dataset in ['Balvan', 'Eliceiri', 'Zurich'], "dataset must be in ['Balvan', 'Eliceiri', 'Zurich']"
        img = skio.imread(img_path) if dataset == 'Eliceiri' else cv2.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        return img
    def enhance(img, percentiles, dataset):
        assert dataset in ['Balvan', 'Eliceiri', 'Zurich'], "dataset must be in ['Balvan', 'Eliceiri', 'Zurich']"
        if dataset == 'Eliceiri':
            img_en = exposure.rescale_intensity(img, tuple(percentiles))
        else:
            img_en = channelwise_rescale(img, percentiles)
        return img_en

    # dataroot_real.format(fold=fold) for fold in folds
    save_dir = f'./Datasets/{dataset}_patches/result_imgs/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    direction = {'A': 'R', 'B': 'T'}
    title_dict = {
            'A':{'ori':'Modality A', 'cyc':'CycleGAN', 'drit':'DRIT++', 'p2p':'Pix2pix', 'star':'StarGANv2', 'comir':'CoMIR'},
            'B':{'ori':'Modality B', 'cyc':'CycleGAN', 'drit':'DRIT++', 'p2p':'Pix2pix', 'star':'StarGANv2', 'comir':'CoMIR'},
            }
    gan_names = ['cyc_A', 'cyc_B', 'drit_A', 'drit_B', 'p2p_A', 'p2p_B', 'star_A', 'star_B', 'comir_A', 'comir_B']
    modalities = ['A', 'B']

    # Pick samples
    f_names = {}
    for i_sample in range(n):
        fold = None if dataset == 'Eliceiri' else i_sample % 3 + 1
        f_name = os.path.basename(random.choice(glob(f'{dataroot_real}/A/test/*_R.*'.format(fold=fold)))).split('.')[0][:-2]
        while f_name in f_names:
            f_name = os.path.basename(random.choice(glob(f'{dataroot_real}/A/test/*_R.*'.format(fold=fold)))).split('.')[0][:-2]
        f_names[f_name] = fold
    
    
    # Contrast Enhance
    ## load all images
    imgs = {
            'A': {'sample': imread(sample_path['A'], dataset)},
            'B': {'sample': imread(sample_path['B'], dataset)},
            }
    imgs_flat = {'A':[], 'B':[]}
    pers = {}
    for modality in modalities:
        gan_types = [folder for folder in gan_names if modality not in folder]
        img = imgs[modality]['sample']
        h, w, channels = img.shape
        imgs_flat[modality].append(img.reshape((h*w, channels)))
        
        # include intro examples into percentile computing
        if dataset != 'Zurich':
            for rof in ['real', 'fake']:
                img = imread(intro_examples[f'{rof}{modality}'], dataset)
                h, w, channels = img.shape
                imgs_flat[modality].append(img.reshape((h*w, channels)))
                imgs[modality][f'intro_{rof}'] = img
        
        for f_name, fold in f_names.items():
            for i_gan in range(len(gan_types)+1):
                if i_gan == 0:
                    title = 'ori'
                    suffix = os.path.basename(glob(f'{dataroot_real}/{modality}/test/*_{direction[modality]}.*'.format(fold=fold))[0]).split('.')[-1]
                    img = imread(f'{dataroot_real}/{modality}/test/{f_name}_{direction[modality]}.{suffix}'.format(fold=fold), dataset)
                    imgs[modality][f'{title}_{f_name}'] = img
                else:
                    title, modality_gan = gan_types[i_gan-1].split('_')
                    suffix = os.path.basename(glob(f'{dataroot_fake}/{title}_{modality}/*_{direction[modality]}.*'.format(fold=fold))[0]).split('.')[-1]
                    img = imread(f'{dataroot_fake}/{title}_{modality}/{f_name}_{direction[modality]}.{suffix}'.format(fold=fold), dataset)
                    imgs[modality][f'{title}_{f_name}'] = img
                if title != 'comir':
                    h, w, channels = img.shape
                    imgs_flat[modality].append(img.reshape((h*w, channels)))
        ## compute per-channel percentiles
        imgs_flat[modality] = np.concatenate(imgs_flat[modality], axis=0)
        if dataset == 'Eliceiri':
            pers[modality] = np.percentile(imgs_flat[modality], (1, 99))
        else:
            pers[modality] = [np.percentile(imgs_flat[modality][..., c], (1, 99)) for c in range(channels)]

    
    # Save enhanced sample original images
    for modality in modalities:
        sample_en = enhance(imgs[modality]['sample'], pers[modality], dataset)
        skio.imsave(save_dir + f'{dataset}_{sample_name}_{modality}_en.png', sample_en)
        # save enhanced intro examples
        if dataset != 'Zurich':
            for rof in ['real', 'fake']:
                sample_en = enhance(imgs[modality][f'intro_{rof}'], pers[modality], dataset)
                skio.imsave(save_dir + f'{dataset}_intro_{rof}{modality}_en.png', sample_en)

    
    # Draw sample GAN imgs
    for modality in modalities:
        gan_types = [folder for folder in gan_names if modality not in folder]
        ncol, nrow = len(gan_types)+1, n
        gap = 0.01
        fig, axs = plt.subplots(
                nrows=n, ncols=len(gan_types)+1, 
                gridspec_kw=dict(wspace=gap, hspace=gap,
                                 top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                 left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1)),
                figsize=(ncol + 1 + (ncol-1)*gap, nrow + 1 + (nrow-1)*gap), dpi=200,
                sharex='col', sharey='row')
        i_sample = 0
        for f_name, fold in f_names.items():
            for i_gan in range(len(gan_types)+1):
                if i_gan == 0:
                    title = 'ori'
                    suffix = os.path.basename(glob(f'{dataroot_real}/{modality}/test/*_{direction[modality]}.*'.format(fold=fold))[0]).split('.')[-1]
                    img = enhance(imgs[modality][f'{title}_{f_name}'], pers[modality], dataset)
                else:
                    title, modality_gan = gan_types[i_gan-1].split('_')
                    suffix = os.path.basename(glob(f'{dataroot_fake}/{title}_{modality_gan}/*_{direction[modality]}.*'.format(fold=fold))[0]).split('.')[-1]
                    img = imgs[modality_gan][f'{title}_{f_name}']
                    if title != 'comir':
                        img = enhance(img, pers[modality_gan], dataset)

                axs[i_sample, i_gan].imshow(img)
                axs[i_sample, i_gan].label_outer()
                axs[i_sample, i_gan].set_axis_off()
                if i_sample == n - 1:
                    axs[i_sample, i_gan].set_title(title_dict[modality][title], y=-0.25, fontsize=12, color='black')
            i_sample += 1

        plt.savefig(save_dir + f'{dataset}_samples_{modality}_en.png', format='png', dpi=300, bbox_inches='tight')
    return
    
# %%
for dataset in ['Balvan', 'Zurich']:
    result_montage(dataset, n=3)
