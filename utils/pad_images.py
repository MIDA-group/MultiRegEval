# -*- coding: utf-8 -*-
# Python Standard Libraries
from glob import glob
import os, argparse

# ~ Scientific
import numpy as np
# ~ Image manipulation / visualisation
import skimage.io as skio
# ~ Other
from tqdm import tqdm

# %%

def pad_images(in_dir, out_dir, d):
#    in_dir='./pytorch-CycleGAN-and-pix2pix/datasets/zurich_p2p_train/fold1/train'
#    out_dir='../datasets/balvan_cyc_train/fold1/testA'
#    d=256  
    def pad_sample(img, d=256):
        # pad the image size to multiple of divisor d
        img = np.asarray(img)
        (w, h) = img.shape[:2]
        w_pad = (w // d + 1) * d - w
        h_pad = (h // d + 1) * d - h            
        wl = w_pad // 2
        wr = w_pad - wl
        hl = h_pad // 2
        hr = h_pad - hl
        if img.ndim == 2:
            img_pad = np.pad(img, ((wl, wr), (hl, hr)), 'reflect')
        else:
            img_pad = np.pad(img, ((wl, wr), (hl, hr), (0, 0)), 'reflect')
        return img_pad

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
#    img_dirs = glob(f'{in_dir}/*')
    for img_name in tqdm(os.listdir(in_dir)):
        img = skio.imread(f'{in_dir}/{img_name}')
        img = pad_sample(img, d)
        skio.imsave(f'{out_dir}/{img_name}', img)
    return

# %%
parser = argparse.ArgumentParser(description='Pad the image size to multiple of divisor d.')
parser.add_argument(
        '--indir', '-i', 
        required=True,
        help="in_dir of images")
parser.add_argument(
        '--outdir', '-o', 
        required=True,
        help="out_dir of images")
parser.add_argument(
        '--divisor', '-d', 
        help="pad the image size to multiple of divisor d", 
        type=int, 
        default=256)
args = parser.parse_args()

pad_images(args.indir, args.outdir, args.divisor)
#unpad_results('./pytorch-CycleGAN-and-pix2pix/results/eliceiri_p2p_rotation_b2a/test_latest/images', 834, 834)

