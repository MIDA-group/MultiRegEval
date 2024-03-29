"""
Example script to train an affine VoxelMorph model.

For the CVPR and MICCAI papers, we have data arranged in train, validate, and test folders. Inside each folder
are normalized T1 volumes and segmentations in npz (numpy) format. You will have to customize this script slightly
to accommodate your own data. All images should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed. Otherwise,
registration will be scan-to-scan.
"""

import os
import random
import cv2
import argparse
from glob import glob
import math
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import skimage.io as skio
import skimage.transform as skt

import voxelmorph as vxm

# %%
# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
#parser.add_argument('--datadir', default='./data/', help='base data directory')

parser.add_argument("--data_root", type=str,
                    dest="data_root", default='../Datasets/Balvan_patches/fold1/patch_tlevel2/',
                    help="data folder")
parser.add_argument("--model_name", type=str,
                    dest="model_name", default='unsupervised',
                    help="models folder")
parser.add_argument("--supervised", dest="supervised", action="store_true")
parser.set_defaults(supervised=False)
parser.add_argument("--max_clip", type=float,
                    dest="max_clip", default=1,
                    help="maximum input value to calculate bins")
parser.add_argument("--num_bins", type=int,
                    dest="num_bins", default=48,
                    help="number of bins when calculating mutual information")
parser.add_argument("--a2b", type=int, default=0, choices=[0, 1], help='A:SHG, B:BF')
parser.add_argument('--model', type=str,
                    default="1500.h5", help='run nonlinear registration - must specify keras model file')

#parser.add_argument('--atlas', help='atlas filename')
#parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs (default: 1500)')
parser.add_argument('--steps_per_epoch', type=int, help='frequency of model saves (default: 100)')
parser.add_argument('--load_weights', help='optional weights file to initialize with')
parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.00001)')
parser.add_argument('--prob-same', type=float, default=0.3, help='likelihood that source/target training images will be the same (default: 0.3)')

# network architecture parameters
parser.add_argument('--rigid', action='store_true', help='force rigid registration')
parser.set_defaults(rigid=True)
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.set_defaults(bidir=True)
parser.add_argument('--blurs', type=float, nargs='+', default=[1], help='levels of gaussian blur kernel for each scale (default: 1)')
#parser.add_argument('--padding', type=int, nargs='+', default=[256, 256, 256], help='padded image target shape (default: 256 256 256')
#parser.add_argument('--resize', type=float, default=0.25, help='after-padding image resize factor (default: 0.25)')
args = parser.parse_args()

batch_size = args.batch_size
data_root = args.data_root #'../Datasets/Eliceiri_patches/example'
#args.bidir = True
#args.steps_per_epoch = 100
#mode='train'
#args.supervised=False
#i_model='0500'
#model_name='balvan_fold1_us_b2a'

# %%
def dist_coords(coords1, coords2):
    ''' Calculate the point-to-point distances between two coordinates, return a list.
    '''
    return [sum((coords1[i] - coords2[i]) ** 2) ** 0.5 for i in range(len(coords1))]

def unpad_sample(img, d):
    # crop the image size to multiple of divisor d    
    (wi, hi) = img.shape[:2]
    wo = wi // d * d
    ho = hi // d * d
    assert wo <= wi and ho <= hi
    wl = (wi - wo) // 2
    hl = (hi - ho) // 2
    return img[wl:wl+wo, hl:hl+ho]

def load_image(img_path):
    # load image in gretscale and unpad to multiple of 32
    img = cv2.imread(img_path, 0)
    img = unpad_sample(img, d=32)
    return img

def tform_centred_rec(radian, translation, center):
    tform1 = skt.SimilarityTransform(translation=translation)
    tform2 = skt.SimilarityTransform(translation=center)
    tform3 = skt.SimilarityTransform(rotation=radian)
    tform4 = skt.SimilarityTransform(translation=-center)
    tform = tform4 + tform3 + tform2 + tform1
    return tform


def evaluate_model(data_root, model_name, i_model='0500', display=None):
#    data_root='../Datasets/Balvan_patches/fold1/patch_tlevel4/'
#    model_name='balvan_fold1_us_b2a'
#    i_model='1500'

    # dataset-specific variables
    if 'eliceiri' in model_name:
        img_root='../Datasets/HighRes_Splits/WSI'
        w=834
        o=608
    elif 'balvan' in model_name:
        img_root='../Datasets/Balvan_1to4tiles'
        w=300 # patch width
        o=w//2 # offset: upper-left corner of patch
    elif 'zurich' in model_name:
        img_root='../Datasets/Zurich_tiles'
        w=300 # patch width
        o=w//2 # offset: upper-left corner of patch
    
    method='VXM'
    gan_name=''
    mode = model_name.split('_')[-1]
    
    
    coords_ref = np.array(([0,0], [0,w], [w,w], [w,0]))
    centre_patch = np.array((w, w)) / 2. - 0.5
    
    dir_A = data_root + 'A/test/'
    dir_B = data_root + 'B/test/'
    
    supervision = model_name.split('_')[-2]
    assert supervision in ['su', 'us'], "supervision must be in ['su', 'us']"
    supervision_dict = {'su':'supervised', 'us':'unsupervised'}
    
    assert mode in ['a2b', 'b2a', 'a2a', 'b2b'], "mode must be in ['a2b', 'b2a', 'a2a', 'b2b']"
    if mode=='a2b':
        dir_src = dir_A
        dir_tar = dir_B
    elif mode=='b2a':
        dir_src = dir_B
        dir_tar = dir_A
    elif mode=='a2a':
        dir_src = dir_A
        dir_tar = dir_A
    elif mode=='b2b':
        dir_src = dir_B
        dir_tar = dir_B

    
    load_model_file = glob(f'./models/{model_name}/{i_model}*')[-1]
    
    # device handling
    if args.gpu and (args.gpu != '-1'):
        device = '/gpu:' + args.gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        tf.keras.backend.set_session(tf.Session(config=config))
    else:
        device = '/cpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # load the affine model
    with tf.device(device):
        model = vxm.networks.VxmAffine.load(load_model_file)
    
    

    suffix_src = '_' + os.listdir(dir_src)[0].split('_')[-1]
    name_srcs = set([name[:-len(suffix_src)] for name in os.listdir(dir_src)])
    suffix_tar = '_' + os.listdir(dir_tar)[0].split('_')[-1]
    name_tars = set([name[:-len(suffix_tar)] for name in os.listdir(dir_tar)])
    f_names = name_srcs & name_tars
    f_names = list(f_names)
    f_names.sort()

    df = pd.read_csv(data_root + 'info_test.csv', index_col='Filename')
    
    cnt_disp = 0
    for f_name in tqdm(f_names):    
        # extract transformed patch coordinates
        coords_trans = df.loc[
                f_name, 
                ['X1_Trans', 'Y1_Trans', 'X2_Trans', 'Y2_Trans', 'X3_Trans', 'Y3_Trans', 'X4_Trans', 'Y4_Trans']
                ].to_numpy().reshape((4, 2))
        
        # load test pair
        img_src = load_image(dir_src + f"{f_name}_T.{suffix_src.split('.')[-1]}")
        img_tar = load_image(dir_tar + f"{f_name}_R.{suffix_tar.split('.')[-1]}")
        if img_src.ndim == 2:
            img_src = img_src[np.newaxis, ..., np.newaxis]
        if img_tar.ndim == 2:
            img_tar = img_tar[np.newaxis, ..., np.newaxis]
        img_src = img_src.astype('float') / 255
        img_tar = img_tar.astype('float') / 255
        
        # register
        with tf.device(device):
            affine = model.register(img_src, img_tar)
        affine_matrix = np.concatenate([affine.squeeze().reshape((2, 3)), np.zeros((1, 3))], 0) + np.eye(3)
#        coords_rec = skt.matrix_transform(coords_trans, affine_matrix)
        tform = skt.SimilarityTransform(affine_matrix)
        tform_patch_rec = tform_centred_rec(radian=tform.rotation, 
                                            translation=(tform.translation[0], tform.translation[1]), 
                                            center=centre_patch)
        coords_rec = skt.matrix_transform(coords_trans, tform_patch_rec.params)
        
#        img_rec = vxm.tf.utils.transform(img_src, affine, rescale=1.0)
        
        # calculate error
        disp_error = dist_coords(coords_rec, coords_ref)
        
        result = {
            'X1_Recover': coords_rec[0][0], 'Y1_Recover': coords_rec[0][1], 
            'X2_Recover': coords_rec[1][0], 'Y2_Recover': coords_rec[1][1], 
            'X3_Recover': coords_rec[2][0], 'Y3_Recover': coords_rec[2][1], 
            'X4_Recover': coords_rec[3][0], 'Y4_Recover': coords_rec[3][1], 
            'Error': np.mean(disp_error)}
        # update result
        df.loc[f_name, 
               ['X1_Recover', 'Y1_Recover', 'X2_Recover', 'Y2_Recover', 
                'X3_Recover', 'Y3_Recover', 'X4_Recover', 'Y4_Recover', 
                'Error']
               ] = result


        # display patch outline in original image
        if display:
            if cnt_disp < display:               
                suffix = os.path.basename(os.listdir(f'{img_root}/A/')[0]).split('.')[-1]
                imgB = skio.imread(f"{img_root}/B/{f_name}.{suffix}")
                dispdirB = f'{data_root}/display/B/test'
                if not os.path.exists(dispdirB):
                    os.makedirs(dispdirB)
                if 'eliceiri' in model_name:
                    imgB_disp = imgB
                elif 'balvan' in model_name:
                    imgB_disp = np.pad(imgB, w//2, mode='reflect')
                if len(imgB_disp.shape) == 2:
                    imgB_disp = np.repeat(imgB_disp.reshape(imgB_disp.shape[0], imgB_disp.shape[1], 1), 3, axis=-1)
                imgB_disp = cv2.polylines(imgB_disp, pts=[(o+coords_ref).reshape((-1,1,2))], isClosed=True, color=(0,255,0), thickness=2)
                imgB_disp = cv2.polylines(imgB_disp, pts=[np.int32(o+coords_trans).reshape((-1,1,2))], isClosed=True, color=(0,0,255), thickness=2)
                imgB_disp = cv2.polylines(imgB_disp, pts=[np.int32(o+coords_rec).reshape((-1,1,2))], isClosed=True, color=(255,0,0), thickness=2)
#                skio.imshow(imgB_disp)
                skio.imsave(f'{dispdirB}/{f_name}_{method+gan_name}_{mode}_{supervision}.{suffix}', imgB_disp)
                cnt_disp += 1
#                img_rec = vxm.tf.utils.transform(img_src, affine, rescale=1.0)
#                skio.imshow(img_rec[0, ..., 0])
            else:
                return
    
    df.to_csv(data_root + f'results/{method}_{mode}_{supervision}.csv')
    
    return

# %%
if __name__ == '__main__':
# %%
    for data_dir in glob('../Datasets/Balvan_patches/fold1/patch_tlevel*/'):
        evaluate_model(
                data_root=data_dir, 
                model_name='balvan_fold1_us_b2a', 
                i_model='1500', 
                display=None)
# %%
    for data_dir in glob('../Datasets/Balvan_patches/fold1/patch_tlevel*/'):
        evaluate_model(
                data_root=data_dir, 
                model_name='balvan_fold1_su_b2a', 
                i_model='1500', 
                display=None)

# %%
    for data_dir in glob('../Datasets/Eliceiri_patches/patch_tlevel*/'):
        evaluate_model(
                data_root=data_dir, 
                model_name='eliceiri_su_b2a', 
                i_model='1050', 
                display=None)

# %%
    for data_dir in glob('../Datasets/Eliceiri_patches/patch_tlevel*/'):
        evaluate_model(
                data_root=data_dir, 
                model_name='eliceiri_us_b2a', 
                i_model='1450', 
                display=None)


# %%
# let's test it
train_generator = vxm.generators.eliceiri_data_generator(
        data_root=data_root,
        mode='train',
        a2b=args.a2b,
        batch_size=batch_size,
        load_GT=args.supervised,
        bidir=args.bidir)
input_sample, output_sample = next(train_generator)
steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch else math.ceil(len(glob(f'{data_root}/A/train/*_R.*')) / batch_size)

# %% visualize
slices_2d = [f[0,...,0] for f in input_sample + output_sample[:1]]
titles = ['input moving $M$', 'input fixed $F$', 'output moved target $T$']
vxm.tf.neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=False, 
                          save_dir=f'{data_root}/display', save_name=f'train_sample_supervise{args.supervised}.png');

# %%

# %%
with tf.device(device):
    # load the affine model, predict the transform(s), and merge
    model = vxm.networks.VxmAffine.load(load_model_file)
    affine = model.register(val_input[0], val_input[1])

    # apply the transform and crop back to the target space
    aligned = vxm.tf.utils.transform(img_src, affine, rescale=1.0)[0, ..., 0]

# %%
# test
val_generator = vxm.generators.eliceiri_data_generator(
        data_root='../Datasets/Balvan_patches/fold1/patch_tlevel4/',
        mode='test',
        a2b=args.a2b,
        batch_size=args.batch_size,
        load_GT=True,
        bidir=args.bidir)

# %%
model_name='balvan_fold1_us_b2a'
i_model='0960'
load_model_file = glob(f'./models/{model_name}/{i_model}*')[-1]
model = vxm.networks.VxmAffine.load(load_model_file)



# %%
val_input, val_output = next(val_generator)

val_pred = model.predict(val_input)

# visualize
slices_2d = [f[0,...,0] for f in val_input + val_output[:1] + val_pred[:1]]
titles = ['input moving $M$', 'input fixed $F$', 'output ground truth', 'predicted moved $M_\phi$']
vxm.tf.neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True, 
                          save_dir=f'./result_samples', save_name=f'test_sample_supervise{args.supervised}.png');

