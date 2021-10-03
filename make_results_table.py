# -*- coding: utf-8 -*-

import pandas as pd
from glob import glob
import os
import numpy as np

# %%
def make_success_table(preprocess='nopre', ds=2):
#    dataset='Zurich'
#    fold=1
    
    def success_rate(patches_dir, method, gan_name='', preprocess='nopre', mode='b2a'):
        if gan_name in ['A2A', 'B2B', 'B2A']:
            gan_name = ''
        # read results
        dfs = [pd.read_csv(csv_path) for csv_path 
               in glob(f'{patches_dir}/patch_tlevel*/results/{method+gan_name}_{mode}_{preprocess}.csv')]
        
        whole_df = pd.concat(dfs)
        n_success = whole_df['Error'][whole_df['Error'] <= w*0.02].count()
        rate_success = n_success / len(whole_df)
#        print(f'{method+gan_name}_{preprocess}', rate_success)
        return rate_success
    
    assert ds in [2, 3], ("Number of dimensions must be in [2, 3]")    
    row_names = ['cyc_A', 'cyc_B', 'drit_A', 'drit_B', 'p2p_A', 'p2p_B', 'star_A', 'star_B', 'comir', 
#                 'A2A', 'B2B', 
                 'B2A',
#                 'MI', 'CA',
                 ]
    
    # csv information
    if ds == 3:
        header = ['Method', 'RIRE_aAMD']
        datasets = ['RIRE']
        reg_methods = ['aAMD']
    else:
        header = [
                'Method', 
                'Zurich_aAMD', 'Zurich_SIFT', 
                'Balvan_aAMD', 'Balvan_SIFT', 
                'Eliceiri_aAMD', 'Eliceiri_SIFT', 
                ]
        datasets = ['Balvan', 'Eliceiri', 'Zurich']
        reg_methods = ['SIFT', 'aAMD']
    df = pd.DataFrame(columns=header)
        
    
    for row_name in row_names:
        row_dict = {}
        # calculate overall success rate
        if row_name == 'A2A':
            direction = 'a2a'
        elif row_name == 'B2B':
            direction = 'b2b'
        else:
            direction = 'b2a'
        row_dict['Method'] = row_name
        
        for dataset in datasets:
            
            if dataset == 'Eliceiri':
                dataroot_real = f'./Datasets/{dataset}_patches'
                w = 834
                folds = ['']
            else:
                dataroot_real = f'./Datasets/{dataset}_patches/fold{{fold}}'
                w = np.asarray((210, 210, 70)).mean() if dataset == 'RIRE' else 300
                folds = [1, 2, 3]
            
            for reg_method in reg_methods:
                l_success = [success_rate(patches_dir=dataroot_real.format(fold=fold), 
                                          method=reg_method, 
                                          gan_name=row_name, 
                                          preprocess=preprocess, 
                                          mode=direction
                                          ) for fold in folds]
                success_mean = np.mean(l_success)
                if dataset == 'Eliceiri':
                    row_dict[f'{dataset}_{reg_method}'] = f"{100*success_mean:.1f}"
                else:
                    success_std = np.std(l_success)
                    row_dict[f'{dataset}_{reg_method}'] = f"{100*success_mean:.1f}$\pm${100*success_std:.1f}"
    
        df = df.append(row_dict, ignore_index=True)  
        
    for row_name in ['Mind', 'MindMSE', 'NGF', 'MI', 'CA']:
        if row_name == 'CA' and ds == 3:
            continue
        row_dict = {}
        direction = 'b2a'
        row_dict['Method'] = row_name
        if ds == 3:
            data_cols = ['RIRE']
        else:
            data_cols = ['Eliceiri'] if row_name == 'CA' else ['Balvan', 'Eliceiri', 'Zurich']
            
        for dataset in data_cols:
            if dataset == 'Eliceiri':
                dataroot_real = f'./Datasets/{dataset}_patches'
                w = 834
                folds = ['']
            else:
                dataroot_real = f'./Datasets/{dataset}_patches/fold{{fold}}'
                w = np.asarray((210, 210, 70)).mean() if dataset == 'RIRE' else 300
                folds = [1, 2, 3]
            l_success = [success_rate(patches_dir=dataroot_real.format(fold=fold), 
                                      method=row_name, 
                                      gan_name='', 
                                      preprocess=preprocess, 
                                      mode=direction
                                      ) for fold in folds]
            success_mean = np.mean(l_success)
            if dataset == 'Eliceiri':
                row_dict[f'{dataset}_aAMD'] = f"{100*success_mean:.1f}"
            else:
                success_std = np.std(l_success)
                row_dict[f'{dataset}_aAMD'] = f"{100*success_mean:.1f}$\pm${100*success_std:.1f}"
        df = df.append(row_dict, ignore_index=True)  
    
    result_dir = './Datasets'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    df.to_csv(f'{result_dir}/success_table_{preprocess}_{ds}d.csv')

    return

# %%
if __name__ == '__main__':
    make_success_table(preprocess='nopre', ds=3)