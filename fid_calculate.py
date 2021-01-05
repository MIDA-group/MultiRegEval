# -*- coding: utf-8 -*-

import torch
from pytorch_fid import fid_score
import pandas as pd
from glob import glob
import os, argparse
import numpy as np

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
    
    assert dataset in ['Balvan', 'Eliceiri', 'Zurich'], "dataset must be in ['Balvan', 'Eliceiri', 'Zurich']"
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
def make_FID_success_table(dataset, preprocess='nopre'):
#    dataset='Zurich'
#    fold=1
    
    assert dataset in ['Balvan', 'Eliceiri', 'Zurich'], "dataset must be in ['Balvan', 'Eliceiri', 'Zurich']"
    if dataset == 'Eliceiri':
        dataroot_real = f'./Datasets/{dataset}_patches'
        path_FIDcsv = f'./Datasets/{dataset}_patches_fake'
        w = 834
        folds = ['']
    else:
        dataroot_real = f'./Datasets/{dataset}_patches/fold{{fold}}'
        path_FIDcsv = f'./Datasets/{dataset}_patches_fake/fold*'
        w = 300
        folds = [1, 2, 3]
    
    
    def success_rate(patches_dir, method, gan_name='', preprocess='nopre', mode='b2a'):
        if gan_name in ['A2A', 'B2B']:
            gan_name = ''
        # read results
        dfs = [pd.read_csv(csv_path) for csv_path 
               in glob(f'{patches_dir}/patch_tlevel*/results/{method+gan_name}_{mode}_{preprocess}.csv')]
        
        whole_df = pd.concat(dfs)
        n_success = whole_df['Error'][whole_df['Error'] <= w*0.02].count()
        rate_success = n_success / len(whole_df)
#        print(f'{method+gan_name}_{preprocess}', rate_success)
        return rate_success
    
    
    gan_names = ['testA', 'testB', 
                 'cyc_A', 'cyc_B', 'drit_A', 'drit_B', 'p2p_A', 'p2p_B', 'star_A', 'star_B', 'comir']
    
    
    # csv information
    header = [
            'Method', 'Dataset', 
            'FID_mean', 'FID_STD', 
            'Success_aAMD_mean', 'Success_aAMD_STD', 
            'Success_SIFT_mean', 'Success_SIFT_STD', 
            ]
    df = pd.DataFrame(columns=header)
    
    # calculate overall FID
    dfs_FID = [pd.read_csv(csv_path) for csv_path 
           in glob(f'{path_FIDcsv}/FIDs.csv')]        
    _whole_df_FID = pd.concat(dfs_FID)
    _whole_df_FID = _whole_df_FID.drop(columns=[_whole_df_FID.columns[0], 'Tlevel'])
    df_FID_grouped = _whole_df_FID.groupby(['GAN_name', 'Fold']).mean().groupby(['GAN_name'])
    
    row_dict = {'Dataset': dataset}
    
    for gan_name in gan_names:
        # calculate overall success rate
        if gan_name == 'testA':
            gan = 'A2A'
            direction = 'a2a'
        elif gan_name == 'testB':
            gan = 'B2B'
            direction = 'b2b'
        else:
            gan = gan_name
            direction = 'b2a'
        row_dict['Method'] = gan
        for reg_method in ['SIFT', 'aAMD']:
            l_success = [success_rate(patches_dir=dataroot_real.format(fold=fold), 
                                      method=reg_method, 
                                      gan_name=gan, 
                                      preprocess=preprocess, 
                                      mode=direction
                                      ) for fold in folds]
            row_dict[f'Success_{reg_method}_mean'] = np.mean(l_success)
            row_dict[f'Success_{reg_method}_STD'] = np.std(l_success)
        row_dict['FID_mean'] = df_FID_grouped.mean().loc[gan_name, 'FID']
        row_dict['FID_STD'] = df_FID_grouped.std().loc[gan_name, 'FID']
    
        df = df.append(row_dict, ignore_index=True)  
        
    row_dict = {'Dataset': dataset}
    for baseline in ['train2testA', 'train2testB']:
        row_dict['Method'] = baseline
        row_dict['FID_mean'] = df_FID_grouped.mean().loc[baseline, 'FID']
        row_dict['FID_STD'] = df_FID_grouped.std().loc[baseline, 'FID']
        df = df.append(row_dict, ignore_index=True)  
    
    result_dir = f'./Datasets/{dataset}_patches_fake'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    df.to_csv(f'{result_dir}/FID_success_{preprocess}.csv')

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
    
    for pre in ['nopre', 'hiseq']:
        for dataset in ['Balvan', 'Eliceiri', 'Zurich']:
            make_FID_success_table(dataset=dataset, preprocess=pre)
