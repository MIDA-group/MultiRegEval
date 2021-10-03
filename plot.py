# -*- coding: utf-8 -*-
# make plots from csv data
import pandas as pd
import os, cv2, random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import itertools
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import skimage.transform as skt


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
    markers = itertools.cycle(('p', '*', 'P', 'X', '+', '.', 'x', 'h', 'H', '1')) 

    # dataset-specific variables
    assert dataset in ['Eliceiri', 'Balvan', 'Zurich', 'RIRE'], "dataset must be in ['Eliceiri', 'Balvan', 'Zurich', 'RIRE']"
    if dataset == 'Eliceiri':
        target_root = './Datasets/Eliceiri_patches'
        w = 834
    elif dataset == 'Balvan':
        target_root = './Datasets/Balvan_patches/fold1'
        w = 300
    elif dataset == 'Zurich':
        target_root = './Datasets/Zurich_patches/fold1'
        w = 300
    elif dataset == 'RIRE':
        target_root = './Datasets/RIRE_patches/fold1'
        w = np.asarray((210, 210, 70)).mean()
    
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
    
    ax.set_xlabel('Initial displacement $d_{\mathrm{Init}}$ [px]', fontsize=15, color =label_color)
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
#        dataset='RIRE', 
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
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c_out = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    if type(color) is str:
        c_out = mc.to_hex(c_out)
    return c_out

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
    markers = itertools.cycle(('x', 'h', 'P', 'X', '+', '.', 'p', '*', 'H', '1')) 

    assert pre in ['', 'nopre', 'PCA', 'hiseq'], "pre must be in ['', 'nopre', 'PCA', 'hiseq']"
    
    # dataset-specific variables
    assert dataset in ['Eliceiri', 'Balvan', 'Zurich', 'RIRE'], "dataset must be in ['Eliceiri', 'Balvan', 'Zurich', 'RIRE']"
    if dataset == 'Eliceiri':
        root_dir = './Datasets/Eliceiri_patches'
        w = 834
        fold = 1
    elif dataset == 'Balvan':
        root_dir = f'./Datasets/Balvan_patches/fold{fold}'
        w = 300
    elif dataset == 'Zurich':
        root_dir = f'./Datasets/Zurich_patches/fold{fold}'
        w = 300
    elif dataset == 'RIRE':
        root_dir = f'./Datasets/RIRE_patches/fold{fold}'
        w = np.asarray((210, 210, 70)).mean()
    if fold == 'all':
        root_dir = f'./Datasets/{dataset}_patches'
        
    label_dict = {
            'SIFTcyc_A_b2a': 'cyc_A',
            'SIFTcyc_B_b2a': 'cyc_B',
            'SIFTdrit_A_b2a': 'drit_A',
            'SIFTdrit_B_b2a': 'drit_B',
            'SIFTp2p_A_b2a': 'p2p_A',
            'SIFTp2p_B_b2a': 'p2p_B',
            'SIFTstar_A_b2a': 'star_A',
            'SIFTstar_B_b2a': 'star_B',
            'SIFTcomir_b2a': 'comir',
            'SIFT_b2a': 'B2A',
            'SIFT_a2a': 'a2a',
            'SIFT_b2b': 'b2b',
            'aAMDcyc_A_b2a': 'cyc_A',
            'aAMDcyc_B_b2a': 'cyc_B',
            'aAMDdrit_A_b2a': 'drit_A',
            'aAMDdrit_B_b2a': 'drit_B',
            'aAMDp2p_A_b2a': 'p2p_A',
            'aAMDp2p_B_b2a': 'p2p_B',
            'aAMDstar_A_b2a': 'star_A',
            'aAMDstar_B_b2a': 'star_B',
            'aAMDcomir_b2a': 'comir',
            'aAMD_b2a': 'B2A',
            'aAMD_a2a': 'a2a',
            'aAMD_b2b': 'b2b',
            'MI_b2a': 'MI',
            'CA_b2a': 'CA',
            'Mind_b2a': 'MIND',
            'NGF_b2a': 'NGF',
            }
    
    def plot_single_curve(method, mode='b2a', preprocess='nopre', color=None):
        # read results
        if fold == 'all':
            dfs = [pd.read_csv(csv_path) for csv_path 
                   in glob(f'{root_dir}/fold*/patch_tlevel*/results/{method}_{mode}_{preprocess}.csv')]
        else:
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
        lw = None   # linewidth
        if method in ['MI', 'CA'] or 'MI' in method:
            linestyle = '--'
            z=4
            if 'MI' in method:
                color = 'black'
        elif method in ['Mind', 'NGF']:
            linestyle = (0, (3, 1, 1, 1))
        elif method != 'VXM' and '_' not in method and 'comir' not in method:
            linestyle = ':'
            lw=2
            z=4.1
        else:
            if 'comir' in method:
                linestyle = '-.'
                color = adjust_lightness(c, amount=1.3)
            else:
                linestyle = '-'
            m = next(markers)
        
        if method == 'VXM':
            ax.plot(bin_centres, success_rates, linestyle=linestyle, marker=m, color=color, alpha=0.7, markersize=10, 
                    label=f'{method}_{mode}_{preprocess}')
        else:
            ax.plot(bin_centres, success_rates, linestyle=linestyle, marker=m, color=color, alpha=0.7, markersize=10, 
                    linewidth=lw, zorder=z, 
                    label=label_dict[f'{method}_{mode}'])
    
        return bin_edges
        
    # %
    
#    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7), sharex='col', sharey='row')
    # set colour
#    color_cycler = plt.style.library['tableau-colorblind10']['axes.prop_cycle']
#    colors = color_cycler.by_key()['color']
#    ax.set_prop_cycle(color_cycler)
    colors = sns.color_palette("Paired").as_hex()
    colors = itertools.cycle(colors)
    
    # other lines
    i_row = 0
    for k in label_dict.keys():
        parts = k.split('_')
        mode = parts[-1]
        i_ = [i for i, ltr in enumerate(k) if ltr == '_']
        method = k[:i_[-1]].replace('results_','')
        if plot_method in method and mode == 'b2a':
            c = next(colors)
            if label_dict[k] == 'B2A':
                c = next(colors)
                i_row += 1
            if dark != True:      # darken bright colors for bright mode
                c = adjust_lightness(c, amount=0.4) if i_row % 2 == 0 else adjust_lightness(c, amount=1.2)
            bin_edges = plot_single_curve(method=method, mode=mode, preprocess=pre, color=c)
            i_row += 1

    # baselines
    bin_edges = plot_single_curve(method='MI', mode='b2a', preprocess='nopre')
#    bin_edges = plot_single_curve(method='MI3', mode='b2a', preprocess='nopre')
#    bin_edges = plot_single_curve(method='MI5', mode='b2a', preprocess='nopre')
    bin_edges = plot_single_curve(method='Mind', mode='b2a', preprocess='nopre')
    bin_edges = plot_single_curve(method='NGF', mode='b2a', preprocess='nopre')
    
    if dataset == 'Eliceiri':
        bin_edges = plot_single_curve(method='CA', mode='b2a', preprocess='nopre')
        # un-comment to enable legend
        ax.legend(fontsize=22, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.5), framealpha=0.0)

    if dataset == 'RIRE':
        ax.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
        ax.legend(fontsize=22, loc='center left', ncol=2, bbox_to_anchor=(1.05, 0.5), framealpha=0.0)
    
    # bin edges
    for edge in bin_edges:
        ax.axvline(x=edge, linestyle='dotted', color='grey', zorder=1.5)
    if dataset == 'Eliceiri':
        ax.set_xlim(left=0, right=225)
    ax.set_ylim(bottom=-0.05, top=1.05)
    
    ax.set_xlabel('Initial displacement $d_{\mathrm{Init}}$ [px]', fontsize=15, color=label_color)
    ax.set_ylabel('Registration success rate $\lambda$', fontsize=15, color=label_color)
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
        ax.set_xlim(left=-1, right=81)
        secaxx.set_xlim(left=0, right=0.27)

    #plt.show()
    save_dir = f'{root_dir}/result_imgs/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if dark == True:
        plt.savefig(save_dir + f'dark_{dataset}_success_{plot_method}_{pre}.png', 
                    format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.savefig(save_dir + f'dark_{dataset}_success_{plot_method}_{pre}.svg', 
                    format='svg', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(save_dir + f'{dataset}_success_{plot_method}_{pre}.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir + f'{dataset}_success_{plot_method}_{pre}.pdf', format='pdf', bbox_inches='tight')

    return

# %%
plot_success_rate(dataset='Eliceiri', plot_method='aAMD', pre='nopre', fold='all', dark=False) # for legend test

DARK=False
for method in ['SIFT', 'aAMD']:
    for dataset in ['Balvan', 'Zurich', 'Eliceiri']:
        plot_success_rate(dataset=dataset, plot_method=method, pre='nopre', fold='all', dark=DARK)

# %%

def fid_scatter(dataset, preprocess='nopre', dark=True):
#    dataset='Zurich'
#    preprocess='nopre'
#    dark=True
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

    assert preprocess in ['', 'nopre', 'PCA', 'hiseq'], "preprocess must be in ['', 'nopre', 'PCA', 'hiseq']"
    
    # dataset-specific variables
    assert dataset in ['Eliceiri', 'Balvan', 'Zurich'], "dataset must be in ['Eliceiri', 'Balvan', 'Zurich']"

    root_dir = f'./Datasets/{dataset}_patches'
    result_dir = f'./Datasets/{dataset}_patches_fake'

#    gan_names = ['A2A', 'B2B', 
#                 'cyc_A', 'cyc_B', 'drit_A', 'drit_B', 'p2p_A', 'p2p_B', 'star_A', 'star_B', 'comir']
    gan_names = ['cyc_A', 'cyc_B', 'drit_A', 'drit_B', 'p2p_A', 'p2p_B', 'star_A', 'star_B', 'comir', 'B2A']


    # read results
    df = pd.read_csv(f'{result_dir}/FID_success_{preprocess}.csv', index_col='Method')


    # make scatter plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7), sharex='col', sharey='row')
    # set colour
#    color_cycler = plt.style.library['tableau-colorblind10']['axes.prop_cycle']
#    ax.set_prop_cycle(color_cycler)
    colors = sns.color_palette("Paired").as_hex()
    colors = itertools.cycle(colors)
    
    # plot
    i_row = 0
    legend1_elements = []
    for method_row in df.itertuples():
        if method_row.Index in gan_names:
            c = next(colors)
            if method_row.Index == 'B2A':
                c = next(colors)
                i_row += 1
            if dark != True:      # darken bright colors for bright mode
                c = adjust_lightness(c, amount=0.4) if i_row % 2 == 0 else adjust_lightness(c, amount=1.2)
            ax.scatter(method_row.FID_mean, method_row.Success_aAMD_mean, 
#                       label=method_row.Index, 
                       c=c, s=12**2, marker='o', alpha=0.6, zorder=2.5)
            ax.scatter(method_row.FID_mean, method_row.Success_SIFT_mean, 
#                       label=method_row.Index, 
                       c=c, s=12**2, marker='X', alpha=0.6, zorder=2.5)
            if dataset != 'Eliceiri':
                # Error bars
                ax.errorbar(method_row.FID_mean, method_row.Success_aAMD_mean, 
                            xerr=method_row.FID_STD, yerr=method_row.Success_aAMD_STD, 
                            c=c, capsize=2, alpha=0.3)
                ax.errorbar(method_row.FID_mean, method_row.Success_SIFT_mean, 
                            xerr=method_row.FID_STD, yerr=method_row.Success_SIFT_STD, 
                            c=c, capsize=2, alpha=0.3)
            legend1_elements.append(Patch(color=c, label=method_row.Index, alpha=0.6))
            i_row += 1
#    ax.scatter(df['FID_mean'], df['Success_aAMD_mean'], alpha=0.6)
#    ax.scatter(df['FID_mean'], df['Success_SIFT_mean'], alpha=0.6)
    
    # un-comment to enable legend
    if dataset == 'Eliceiri' or dark == True:    
        legend1 = ax.legend(handles=legend1_elements, 
                            fontsize=22, loc='center left', bbox_to_anchor=(1.2, 0.5), framealpha=0.0)
        ax.add_artist(legend1)
    
#    # FID baselines
#    baselineA = ax.axvline(x=df.loc['train2testA', 'FID_mean'], 
#                           linestyle="--", color=next(colors), alpha=0.5, label='train2test_A')
#    baselineB = ax.axvline(x=df.loc['train2testB', 'FID_mean'], 
#                           linestyle="--", color=next(colors), alpha=0.5, label='train2test_B')
    
    # 2nd legend
    # un-comment to enable legend
    if dataset == 'Eliceiri' or dark == True:    
        legend2_elements = [Line2D([],[], linewidth=0, marker='o', markersize=12, c='grey', label='aAMD'),
                            Line2D([],[], linewidth=0, marker='X', markersize=12, c='grey', label='SIFT'),
#                            baselineA, 
#                            baselineB,
                            ]
        ax.legend(handles=legend2_elements, fontsize=22, loc='center left', bbox_to_anchor=(1.5, 0.5), framealpha=0.0)
    
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.set_xlabel('Fréchet Inception Distance ($FID$)', fontsize=15, color=label_color)
    ax.set_ylabel('Registration success rate $\lambda$', fontsize=15, color=label_color)
    ax.tick_params(labelsize='large')
    
    #plt.show()
    save_dir = f'{root_dir}/result_imgs/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if dark == True:
        plt.savefig(save_dir + f'dark_{dataset}_fid_{preprocess}.png', 
                    format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.savefig(save_dir + f'dark_{dataset}_fid_{preprocess}.svg', 
                    format='svg', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        plt.savefig(save_dir + f'{dataset}_fid_{preprocess}.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir + f'{dataset}_fid_{preprocess}.pdf', format='pdf', bbox_inches='tight')

    return

# %%
fid_scatter(dataset='Eliceiri', preprocess='nopre', dark=False) # for legend test

DARK=False
for pre in ['nopre', 'hiseq']:
    for dataset in ['Balvan', 'Eliceiri', 'Zurich']:
        fid_scatter(dataset=dataset, preprocess=pre, dark=DARK)

# %%
def unpad_sample(img, wo, ho):
    (wi, hi) = img.shape[:2]
    assert wo <= wi and ho <= hi
    wl = (wi - wo) // 2
    hl = (hi - ho) // 2
    return img[wl:wl+wo, hl:hl+ho]
# %%
def result_montage(dataset, n=3):
#    dataset='Eliceiri'
#    modality='A'
    assert dataset in ['Balvan', 'Eliceiri', 'Zurich', 'RIRE'], "dataset must be in ['Balvan', 'Eliceiri', 'Zurich', 'RIRE']"
    if dataset == 'Eliceiri':
        dataroot_real = f'./Datasets/{dataset}_patches/patch_tlevel1'
        dataroot_fake = f'./Datasets/{dataset}_patches_fake/patch_tlevel1'
    elif dataset == 'RIRE':
        dataroot_real = f'./Datasets/{dataset}_temp/fold{{fold}}'
        dataroot_fake = f'./Datasets/{dataset}_slices_fake/fold{{fold}}'
    else:
        dataroot_real = f'./Datasets/{dataset}_patches/fold{{fold}}/patch_tlevel1'
        dataroot_fake = f'./Datasets/{dataset}_patches_fake/fold{{fold}}/patch_tlevel1'
    # dataroot_real.format(fold=fold) for fold in folds
    
    direction = {'A': 'R', 'B': 'T'}
    title_dict = {
            'A':{'ori':'Modality A', 'cyc':'CycleGAN', 'drit':'DRIT++', 'p2p':'Pix2pix', 'star':'StarGANv2', 'comir':'CoMIR'},
            'B':{'ori':'Modality B', 'cyc':'CycleGAN', 'drit':'DRIT++', 'p2p':'Pix2pix', 'star':'StarGANv2', 'comir':'CoMIR'},
            }
    gan_names = ['cyc_A', 'cyc_B', 'drit_A', 'drit_B', 'p2p_A', 'p2p_B', 'star_A', 'star_B', 'comir_A', 'comir_B']
    modalities = ['A', 'B']

    f_names = {}
    for i_sample in range(n):
        fold = None if dataset == 'Eliceiri' else i_sample % 3 + 1
        if dataset == 'RIRE':
            f_name = os.path.basename(random.choice(glob(f'{dataroot_real}/A/test/*_z1.*'.format(fold=fold))))
            while f_name in f_names:
                f_name = os.path.basename(random.choice(glob(f'{dataroot_real}/A/test/*_z1.*'.format(fold=fold))))
        else:
            f_name = os.path.basename(random.choice(glob(f'{dataroot_real}/A/test/*_R.*'.format(fold=fold)))).split('.')[0][:-2]
            while f_name in f_names:
                f_name = os.path.basename(random.choice(glob(f'{dataroot_real}/A/test/*_R.*'.format(fold=fold)))).split('.')[0][:-2]
        f_names[f_name] = fold
    
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
                if dataset == 'RIRE':
                    if i_gan == 0:
                        title = 'ori'
                        img = cv2.imread(f'{dataroot_real}/{modality}/test/{f_name}'.format(fold=fold))
                        size_ori = img.shape
                    else:
                        title, modality_gan = gan_types[i_gan-1].split('_')
                        img = cv2.imread(f'{dataroot_fake}/{title}_{modality_gan}/{f_name}'.format(fold=fold))
                        img = unpad_sample(img, size_ori[0], size_ori[1])
                    img = skt.resize(img, (320, 320))
                else:
                    if i_gan == 0:
                        title = 'ori'
                        suffix = os.path.basename(glob(f'{dataroot_real}/{modality}/test/*_{direction[modality]}.*'.format(fold=fold))[0]).split('.')[-1]
                        img = cv2.imread(f'{dataroot_real}/{modality}/test/{f_name}_{direction[modality]}.{suffix}'.format(fold=fold))
                    else:
                        title, modality_gan = gan_types[i_gan-1].split('_')
                        suffix = os.path.basename(glob(f'{dataroot_fake}/{title}_{modality_gan}/*_{direction[modality]}.*'.format(fold=fold))[0]).split('.')[-1]
                        img = cv2.imread(f'{dataroot_fake}/{title}_{modality_gan}/{f_name}_{direction[modality]}.{suffix}'.format(fold=fold))
                axs[i_sample, i_gan].imshow(img)
                axs[i_sample, i_gan].label_outer()
                axs[i_sample, i_gan].set_axis_off()
                if modality == 'A' and i_sample == n - 1:
                    axs[i_sample, i_gan].set_title(title_dict[modality][title], y=-0.25, fontsize=12, color='black')
                if modality == 'B' and i_sample == 0:
                    axs[i_sample, i_gan].set_title(title_dict[modality][title], y=0.98, fontsize=12, color='black')
            i_sample += 1

        save_dir = f'./Datasets/{dataset}_patches/result_imgs/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.savefig(save_dir + f'{dataset}_samples_{modality}.png', format='png', dpi=300, bbox_inches='tight')
    return
    
# %%
for dataset in ['Balvan', 'Zurich']:
    result_montage(dataset, n=3)
