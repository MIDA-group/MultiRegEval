#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register a
scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    python register.py moving.nii.gz fixed.nii.gz --dense-model model.h5 --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered, but if not, an initial linear registration
can be run by specifing an affine model with the --affine-model flag, which expects an affine model file as input.
"""

import os
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='keras model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load moving and fixed images
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(args.fixed, add_feat_axis=add_feat_axis, ret_affine=True)

moving = moving[np.newaxis]
fixed = fixed[np.newaxis]

with tf.device(device):
    # load model and predict
    warp = vxm.networks.VxmDense.load(args.dense_model).register(moving, fixed)
    moved = vxm.tf.utils.transform(moving, warp)

# save warp
if args.warp:
    vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

# save moved image
vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)
