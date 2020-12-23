# -*- coding: utf-8 -*-
# make plots from csv data
import pandas as pd
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import itertools


# %% 
def scatter_plot(dataset, method, gan_name='', preprocess='', mode='b2a', dark=True):
    if dark == True:
        bg_color = '#181717'
        plt.style.use(['ggplot','dark_background'])
        plt.rcParams['axes.facecolor'] = '#212020'
        plt.rcParams['figure.facecolor'] = bg_color
        plt.rcParams['grid.color'] = bg_color
        plt.rcParams['axes.edgecolor'] = bg_color
        label_color = 'white'
    else:
        plt.style.use('ggplot')
        label_color = 'black'
        

    # dataset-specific variables
    assert dataset in ['Eliceiri', 'Balvan', 'Zurich'], "supervision must be in ['Eliceiri', 'Balvan', 'Zurich']"
    if dataset == 'Eliceiri':
        target_root = './Datasets/Eliceiri_patches'
        w = 834
    elif dataset == 'Balvan':
        target_root = './Datasets/Balvan_patches/fold1'
        w = 300
    elif dataset == 'Zurich':
        target_root = './Datasets/Zurich_patches/fold1'
        w = 300
    
    # read results
    dfs = [pd.read_csv(csv_path) for csv_path 
           in glob(f'{target_root}/patch_tlevel*/results/{method+gan_name}_{mode}_{preprocess}.csv')]
    whole_df = pd.concat(dfs)
    #whole_df.loc[:, ['Displacement', 'Error']]

    # make scatter plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7), sharex='col', sharey='row')
    # set colour
    color_cycler = plt.style.library['tableau-colorblind10']['axes.prop_cycle']
    colors = color_cycler.by_key()['color']
    ax.set_prop_cycle(color_cycler)
    # plot
    ax.scatter(whole_df['Displacement'], whole_df['Error'], alpha=0.6)
    ax.set_yscale('log')
    if dataset == 'Eliceiri':
        ax.set_xlim(left=0, right=225)
    ax.set_ylim(bottom=1e-2, top=2000)
        
    # plot identity line
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10000)
    y = x
    ab, = ax.plot(x, y, linestyle='dotted', color='grey', scalex=False, scaley=False, label='$\epsilon = d$')
    
    # plot threshold
    ac = ax.axhline(y=w*0.02, linestyle="--", color="#52854C", label='success threshold $\delta_0$')
    ax.legend(handles=[ac, ab], fontsize='large', framealpha=0.4, loc='lower right')
    
    ax.set_xlabel('Initial displacement $d$ [px]', fontsize=15, color =label_color)
    ax.set_ylabel('Absolute registration error $\epsilon$ [px]', fontsize=15, color =label_color)
    ax.tick_params(labelsize='large')
    
    # Secondary Axis
    def forward(x):
        return x / w
    def inverse(x):
        return x * w
    secaxy = ax.secondary_yaxis('right', functions=(forward, inverse))
    secaxy.set_ylabel('Relative registration error $\delta$', fontsize=15, color=label_color)
    secaxy.tick_params(labelsize='large')
    secaxx = ax.secondary_xaxis('top', functions=(forward, inverse))
    secaxx.set_xlabel('Relative initial displacement to image width', fontsize=15, color=label_color)
    secaxx.tick_params(labelsize='large')
    if dataset in ['Balvan', 'Zurich']:
        secaxx.set_xlim(left=0, right=0.27)
    
#    plt.show()
    save_dir = f'{target_root}/result_imgs/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if dark == True:
        plt.savefig(save_dir + f'dark_scatter_{method+gan_name}_{mode}_{preprocess}.png', 
                    format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.savefig(save_dir + f'dark_scatter_{method+gan_name}_{mode}_{preprocess}.svg', 
                    format='svg', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(save_dir + f'scatter_{method+gan_name}_{mode}_{preprocess}.png', 
                    format='png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir + f'scatter_{method+gan_name}_{mode}_{preprocess}.pdf', 
                    format='pdf', bbox_inches='tight')
    
    return

# %% for local testing 
scatter_plot(
        dataset='Zurich', 
#        dataset='Balvan', 
#        dataset='Eliceiri', 
        method='MI', 
        gan_name='', 
        preprocess='nopre', 
        mode='b2a',
        dark=True)

# %%
DARK=True

for gan in tqdm(['p2p_A', 'p2p_B', 'cyc_A', 'cyc_B', 'drit_A', 'drit_B']):
    for pre in ['nopre', 'hiseq']:
        for method in ['SIFT', 'aAMD']:
            scatter_plot(
                    target_root='./Datasets/Eliceiri_patches', 
                    method=method, 
                    gan_name=gan, 
                    preprocess=pre, 
                    mode='b2a',
                    dark=DARK)

for mode in tqdm(['a2a', 'b2a', 'b2b']):
    for pre in ['nopre', 'hiseq']:
        scatter_plot(
                target_root='./Datasets/Eliceiri_patches', 
                method='aAMD', 
                gan_name='', 
                preprocess=pre, 
                mode=mode,
                dark=DARK)
    scatter_plot(
            target_root='./Datasets/Eliceiri_patches', 
            method='SIFT', 
            gan_name='', 
            preprocess='nopre', 
            mode=mode,
            dark=DARK)
        
for method in ['MI', 'CA']:
    scatter_plot(
            target_root='./Datasets/Eliceiri_patches', 
            method=method, 
            gan_name='', 
            preprocess='nopre', 
            mode='b2a',
            dark=DARK)

for pre in ['su', 'us']:
    scatter_plot(
            target_root='./Datasets/Eliceiri_patches', 
            method='VXM', 
            gan_name='', 
            preprocess=pre, 
            mode='b2a',
            dark=DARK)



# %% Success rate 
def plot_success_rate(dataset, plot_method, pre='nopre', fold=1, dark=True):
    if dark == True:
        bg_color = '#181717'
        plt.style.use(['ggplot','dark_background'])
        plt.rcParams['axes.facecolor'] = '#212020'
        plt.rcParams['figure.facecolor'] = bg_color
        plt.rcParams['grid.color'] = bg_color
        plt.rcParams['axes.edgecolor'] = bg_color
        label_color = 'white'
    else:
        plt.style.use('ggplot')
        label_color = 'black'    
    markers = itertools.cycle(('p', '*', 'P', 'X', '+', '.', 'x', 'h', 'H', '1')) 

    assert pre in ['', 'nopre', 'PCA', 'hiseq'], "pre must be in ['', 'nopre', 'PCA', 'hiseq']"
    
    # dataset-specific variables
    assert dataset in ['Eliceiri', 'Balvan', 'Zurich'], "supervision must be in ['Eliceiri', 'Balvan', 'Zurich']"
    if dataset == 'Eliceiri':
        root_dir = './Datasets/Eliceiri_patches'
        w = 834
    elif dataset == 'Balvan':
        root_dir = f'./Datasets/Balvan_patches/fold{fold}'
        w = 300
    elif dataset == 'Zurich':
        root_dir = f'./Datasets/Zurich_patches/fold{fold}'
        w = 300
    
    def plot_single_curve(method, mode='b2a', preprocess='nopre'):
        # read results
        dfs = [pd.read_csv(csv_path) for csv_path 
               in glob(f'{root_dir}/patch_tlevel*/results/{method}_{mode}_{preprocess}.csv')]
        
        whole_df = pd.concat(dfs)
        
        # success rate
        whole_df['binning'], bin_edges = pd.qcut(whole_df['Displacement'], q=10, retbins=True)
        n_success = whole_df[whole_df['Error'] < w*0.02].groupby('binning').count()['Error']
        success_rates = n_success / whole_df['binning'].value_counts(sort=False)
        bin_centres = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges) - 1)]
        
        z = None    # zorder
        m = None    # marker
        if method in ['MI', 'CA']:
            linestyle = '--'
            z=4
        elif method != 'VXM' and '_' not in method and 'comir' not in method:
            linestyle = '-.'
            z=4.1
        else:
            linestyle = '-'
            m = next(markers)
            
        if method == 'VXM':
            ax.plot(bin_centres, success_rates, linestyle=linestyle, marker=m, label=f'{method}_{mode}_{preprocess}')
        else:
            ax.plot(bin_centres, success_rates, linestyle=linestyle, marker=m, zorder=z, label=f'{method}_{mode}')
    
        return bin_edges
        
    # %
    
#    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7), sharex='col', sharey='row')
    # set colour
    color_cycler = plt.style.library['tableau-colorblind10']['axes.prop_cycle']
    colors = color_cycler.by_key()['color']
    ax.set_prop_cycle(color_cycler)
    
    # read results
    results = [os.path.basename(res_path) for res_path in glob(f'{root_dir}/patch_tlevel2/results/*_*_*.csv')]
    
    # baselines
    bin_edges = plot_single_curve(method='MI', mode='b2a', preprocess='nopre')
    
    # other lines
    for result in results:
        parts = result.split('_')
        preprocess = parts[-1].split('.')[0]
        mode = parts[-2]
        i_ = [i for i, ltr in enumerate(result) if ltr == '_']
        method = result[:i_[-2]].replace('results_','')
        if plot_method == 'SIFT':
    #        if 'aAMD' not in method and preprocess=='nopre':
            if plot_method in method and preprocess==pre:
                bin_edges = plot_single_curve(method=method, mode=mode, preprocess=preprocess)
        elif plot_method == 'aAMD':
    #        if 'SIFT' not in method and preprocess=='nopre':
            if plot_method in method and preprocess==pre:
                bin_edges = plot_single_curve(method=method, mode=mode, preprocess=preprocess)
        elif plot_method == 'VXM':
    #        if 'SIFT' not in method and 'aAMD' not in method:
            if plot_method in method:
                bin_edges = plot_single_curve(method=method, mode=mode, preprocess=preprocess)
    
    ax.legend(fontsize='large', loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.0)
    # bin edges
    for edge in bin_edges:
        ax.axvline(x=edge, linestyle='dotted', color='grey', zorder=1.5)
    if dataset == 'Eliceiri':
        ax.set_xlim(left=0, right=225)
    ax.set_ylim(bottom=-0.05, top=1.05)
    
    ax.set_xlabel('Initial displacement $d$ [px]', fontsize=15, color = label_color)
    ax.set_ylabel('Success rate $\lambda$', fontsize=15, color = label_color)
    ax.tick_params(labelsize='large')

    # Secondary Axis
    def forward(x):
        return x / w
    def inverse(x):
        return x * w
    secaxx = ax.secondary_xaxis('top', functions=(forward, inverse))
    secaxx.set_xlabel('Relative initial displacement to image width', fontsize=15, color=label_color)
    secaxx.tick_params(labelsize='large')
    if dataset in ['Balvan', 'Zurich']:
        secaxx.set_xlim(left=0, right=0.27)

    #plt.show()
    save_dir = f'{root_dir}/result_imgs/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if dark == True:
        plt.savefig(save_dir + f'dark_success_{plot_method}_{pre}.png', 
                    format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.savefig(save_dir + f'dark_success_{plot_method}_{pre}.svg', 
                    format='svg', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(save_dir + f'success_{plot_method}_{pre}.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir + f'success_{plot_method}_{pre}.pdf', format='pdf', bbox_inches='tight')

    return

# %%
DARK=True
for method in ['SIFT', 'aAMD']:
    plot_success_rate(dataset='Balvan', plot_method=method, pre='nopre', fold=2, dark=DARK)
