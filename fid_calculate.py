# -*- coding: utf-8 -*-

import torch
from pytorch_fid import fid_score
import pandas as pd
from glob import glob
import os, argparse

# %%

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
batch_size = 50
dim = 2048
#path1 = './Datasets/Zurich_patches/fold2/patch_tlevel1/A/test'
#path2 = './Datasets/Zurich_patches_fake/fold2/patch_tlevel1/cyc_A'

# %%

#fid_value = fid_score.calculate_fid_given_paths(
#        [path1, path2], 
#        batch_size, device, dim)

# %%

def calculate_FIDs(dataset, fold=1):
#    dataset='Zurich'
#    fold=1
    
    assert dataset in ['Balvan', 'Eliceiri', 'Zurich'], "dataset must be in [['Balvan', 'Eliceiri', 'Zurich']"
    if dataset == 'Eliceiri':
        dataroot_real = f'./Datasets/{dataset}_patches'
        dataroot_fake = f'./Datasets/{dataset}_patches_fake'
        dataroot_train = f'./Datasets/{dataset}_temp'
    else:
        dataroot_real = f'./Datasets/{dataset}_patches/fold{fold}'
        dataroot_fake = f'./Datasets/{dataset}_patches_fake/fold{fold}'
        dataroot_train = f'./Datasets/{dataset}_temp/fold{fold}'
    
    
    gan_names = ['train2testA', 'train2testB', 'testA', 'testB', 
                 'cyc_A', 'cyc_B', 'drit_A', 'drit_B', 'p2p_A', 'p2p_B', 'star_A', 'star_B', 'comir']
      
    
    # csv information
    header = [
            'Dataset', 'Fold', 'Tlevel', 'GAN_name', 'Path_fake', 'Path_real',
            'FID', 
            ]
    df = pd.DataFrame(columns=header)
    
    row_dict = {'Dataset': dataset, 'Fold': fold}
    
    for tlevel in [int(tl[-1]) for tl in glob(f'{dataroot_fake}/patch_tlevel*')]:
        row_dict['Tlevel'] = tlevel
        for gan_name in gan_names:
            row_dict['GAN_name'] = gan_name
            if gan_name in ['train2testA', 'train2testB']:
                row_dict['Path_fake'] = f'{dataroot_train}/{gan_name[-1]}/train/'
                row_dict['Path_real'] = f'{dataroot_real}/patch_tlevel{tlevel}/{gan_name[-1]}/test/'
            elif gan_name in ['testA', 'testB']:
                row_dict['Path_fake'] = f'{dataroot_real}/patch_tlevel{tlevel}/{gan_name[-1]}/test/'
                row_dict['Path_real'] = f'{dataroot_real}/patch_tlevel{tlevel}/{gan_name[-1]}/test/'
            elif gan_name == 'comir':
                row_dict['Path_fake'] = f'{dataroot_fake}/patch_tlevel{tlevel}/{gan_name}_A/'
                row_dict['Path_real'] = f'{dataroot_fake}/patch_tlevel{tlevel}/{gan_name}_B/'
            else:
                row_dict['Path_fake'] = f'{dataroot_fake}/patch_tlevel{tlevel}/{gan_name}/'
                row_dict['Path_real'] = f'{dataroot_real}/patch_tlevel{tlevel}/{gan_name[-1]}/test/'
            row_dict['FID'] = fid_score.calculate_fid_given_paths(
                    [ row_dict['Path_fake'], row_dict['Path_real'] ], 
                    batch_size, device, dim)
            
            df = df.append(row_dict, ignore_index=True)
    
    result_dir = dataroot_fake
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    df.to_csv(f'{result_dir}/FIDs.csv')

    return

# %%
if __name__ == '__main__':
    # for running from terminal
    parser = argparse.ArgumentParser(description='Calculate FIDs for generated patches.')
    parser.add_argument(
            '--dataset', '-d', 
            help="dataset", 
            choices=['Balvan', 'Eliceiri', 'Zurich'], 
            default='Zurich')
    parser.add_argument(
            '--fold', '-f', 
            help="fold", 
            type=int, 
            choices=[1, 2, 3], 
            default=1)
    args = parser.parse_args()
    
    calculate_FIDs(dataset=args.dataset, fold=args.fold)
