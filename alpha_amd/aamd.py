
#
# Py-Alpha-AMD Registration Framework
# Author: Johan Ofverstedt
# Reference: Fast and Robust Symmetric Image Registration Based on Distances Combining Intensity and Spatial Information
#
# Copyright 2019 Johan Ofverstedt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#

#
# Example script for affine registration
#

# Import Numpy/Scipy
import numpy as np
#import scipy as sp
#import scipy.misc
import skimage.io as io
import skimage.transform as skt


# Import transforms
#from transforms import CompositeTransform
#from transforms import AffineTransform
from transforms import Rigid2DTransform
#from transforms import Rotate2DTransform
#from transforms import TranslationTransform
#from transforms import ScalingTransform
import transforms

# Import optimizers
from optimizers import GradientDescentOptimizer

# Import generators and filters
#import generators
import filters

# Import registration framework
from register_multi_channel import RegisterMultiChannel

# Import misc
#import math
#import sys
import time
#import os
import cv2

import pandas as pd
#from csv import reader
# %%
# Registration Parameters
alpha_levels = 7
# Pixel-size
spacing = np.array([1.0, 1.0])
# Run the symmetric version of the registration framework
symmetric_measure = True
# Use the squared Euclidean distance
squared_measure = False

# The fraction of the points to sample randomly (0.0-1.0)
#param_sampling_fraction = 0.005
# The channel mode (sum or decompose)
param_channel_mode = 'sum'
#param_channel_mode = 'decompose'
#param_channel_mode = 'decompose_pre'
# Number of iterations between each printed output (with current distance/gradient/parameters)
param_report_freq = 1e999
# %%
def find_best_transform(ts, scores):
    best_t = ts[0]
    best_score = scores[0]
    
    for i in range(len(ts)):
        if scores[i] < best_score:
            best_t = ts[i]
            best_score = scores[i]

    return best_t, best_score

def register_aamd(ref_im, flo_im, iterations=1.0, param_sampling_fraction=0.01, 
                  param_multi_start=True, do_sigmoid=False):
#    def _no_report_callback(opt):
#        pass
    np.random.seed(1000)
    # The number of iterations

    param_iterations = [int(iterations*3000), int(iterations*1000), int(iterations*200)] # 500 -> 200?

#    do_grayscale = False
#
#    if do_grayscale == True:
#        ref_im = io.imread(ref_im_path, as_gray=True)
#        flo_im = io.imread(flo_im_path, as_gray=True)
#        ref_im = np.squeeze(ref_im)
#        flo_im = np.squeeze(flo_im)
#        ref_im = ref_im.reshape(ref_im.shape + (1,))
#        flo_im = flo_im.reshape(flo_im.shape + (1,))
#    else:
#        ref_im = io.imread(ref_im_path, as_gray=False)
#        flo_im = io.imread(flo_im_path, as_gray=False)
#        ref_im = np.squeeze(ref_im)
#        flo_im = np.squeeze(flo_im)
#        if ref_im.ndim == 2:
#            ref_im = ref_im.reshape(ref_im.shape + (1,))
#        if flo_im.ndim == 2:
#            flo_im = flo_im.reshape(flo_im.shape + (1,))
#    
#    print(ref_im.shape)
#    print(flo_im.shape)
    if ref_im.ndim == 2:
        ref_im = np.expand_dims(ref_im, axis=-1)
    if flo_im.ndim == 2:
        flo_im = np.expand_dims(flo_im, axis=-1)

    flo_mask = None

    ch = ref_im.shape[-1]

#    ref_im_orig = ref_im.copy()

    ref_im = filters.channels_to_list(ref_im)
    flo_im = filters.channels_to_list(flo_im)

    #weights1 = generators.make_circular_hann_window_like_image(ref_im[0], rad_factor = 1.0, spacing=None, p=0.25)
    weights1 = np.ones(ref_im[0].shape)
    mask1 = np.ones(ref_im[0].shape, 'bool')
    #weights2 = generators.make_circular_hann_window_like_image(flo_im[0], rad_factor = 1.0, spacing=None, p=0.25)
    weights2 = np.ones(flo_im[0].shape)
    if flo_mask is None:
        mask2 = np.ones(flo_im[0].shape, 'bool')
    else:
        mask2 = (flo_mask >= 0.5)

    # Save copies of original images
#    flo_im_orig = flo_im.copy()

    def inv_sigmoid(x):
        return np.log((x+1e-7)/(1-x+1e-7))
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(x))

    if do_sigmoid:
        for k in range(ch):
            ref_im[k] = sigmoid(ref_im[k])
            flo_im[k] = sigmoid(flo_im[k])
    else:
        for k in range(ch):
            ref_im[k] = filters.normalize(ref_im[k], 0.0, None)
            flo_im[k] = filters.normalize(flo_im[k], 0.0, mask2)
    
    diag = 0.5 * (transforms.image_diagonal(ref_im[0], spacing) + transforms.image_diagonal(flo_im[0], spacing))

    # Initialize registration framework for 2d images
    reg = RegisterMultiChannel(2)

    reg.set_report_freq(param_report_freq)
#    reg.set_report_func(_no_report_callback)
    reg.set_alpha_levels(alpha_levels)
    reg.set_channel_mode(param_channel_mode)

    reg.set_reference_image(ref_im, spacing)
    reg.set_reference_mask(mask1)
    reg.set_reference_weights(weights1)

    reg.set_floating_image(flo_im, spacing)
    reg.set_floating_mask(mask2)
    reg.set_floating_weights(weights2)

    reg.set_squared_measure(squared_measure)

    # Setup the Gaussian pyramid resolution levels
    if iterations < 1:
        reg.add_pyramid_level(4, 7.0)
        reg.add_pyramid_level(2, 3.0)
        reg.add_pyramid_level(1, 1.0)
    else:
        reg.add_pyramid_level(4, 12.0)
        reg.add_pyramid_level(2, 5.0)
        reg.add_pyramid_level(1, 1.0)

    # Learning-rate / Step lengths [[start1, end1], [start2, end2] ...] (for each pyramid level)
    step_lengths = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.1]]) * 2.0 #* 5e-2

    scale = 0.1 / diag
    tscale = 5.0

    transform_count = 1

    t = Rigid2DTransform()
    reg.add_initial_transform(t, np.array([scale, tscale, tscale]))
    
    if param_multi_start:
        t = Rigid2DTransform()
        t.set_param(0, -0.4)
        reg.add_initial_transform(t, np.array([scale, tscale, tscale]))
        t = Rigid2DTransform()
        t.set_param(0, 0.4)
        reg.add_initial_transform(t, np.array([scale, tscale, tscale]))
        transform_count += 2
    
    # Set the parameters
    reg.set_iterations(param_iterations)
    reg.set_gradient_magnitude_threshold(0.0001)
    reg.set_sampling_fraction(param_sampling_fraction)
    reg.set_step_lengths(step_lengths)
    reg.set_optimizer('sgd')

    # Start the pre-processing
    reg.initialize('./test_images/output/')
    
    # Control the formatting of numpy
    np.set_printoptions(suppress=True, linewidth=200)

    # Start the registration
    reg.run()

    ts = []
    tval = []
    for i in range(transform_count):
        ti, vi = reg.get_output(i)
        ts.append(ti)
        tval.append(vi)
    transform, value = find_best_transform(ts, tval)

    c = transforms.make_image_centered_transform(transform, ref_im[0], flo_im[0], spacing, spacing)

    # Create the output image
    ref_im_warped = [np.zeros(ref_im[i].shape) for i in range(ch)]
    ref_im_copied = [np.zeros(ref_im[i].shape) for i in range(ch)]

    # Transform the floating image into the reference image space by applying transformation 'c'
    for k in range(ch):
        ref_im_copied[k] = ref_im[k]
        c.warp(In = flo_im[k], Out = ref_im_warped[k], in_spacing=spacing, out_spacing=spacing, mode='linear', bg_value = 0.0)

    ref_im_copied = np.squeeze(filters.list_to_channels(ref_im_copied))
    ref_im_warped = np.squeeze(filters.list_to_channels(ref_im_warped))

    return ref_im_warped, c

## points should be on the format [[y, x], [y, x], ..., [y, x]]
#def transform_points_from_ref_to_flo(t, points):
#    return t.transform(points)
#
#def transform_points_from_flo_to_ref(t, points):
#    return t.invert().transform(points)



def transform_coords(t, coords_in, centre_patch):
    def tform_centred_rec(radian, translation, center):
        tform1 = skt.SimilarityTransform(translation=translation)
        tform2 = skt.SimilarityTransform(translation=center)
        tform3 = skt.SimilarityTransform(rotation=radian)
        tform4 = skt.SimilarityTransform(translation=-center)
        tform = tform4 + tform3 + tform2 + tform1
        return tform
    param = t.get_params()
    rot_radian = -param[0]
    tx = param[2]
    ty = param[1]
    # transform the transformed patch coordinates back
    tform_patch_rec = tform_centred_rec(radian=rot_radian, translation=(tx, ty), center=centre_patch)
    coords_out = skt.matrix_transform(coords_in, tform_patch_rec.params)
    return coords_out

def tform_centred(radian, translation, center):
    # first translation, then rotation
    tform1 = skt.SimilarityTransform(translation=center)
    tform2 = skt.SimilarityTransform(rotation=radian)
    tform3 = skt.SimilarityTransform(translation=-center)
    tform4 = skt.SimilarityTransform(translation=translation)
    tform = tform4 + tform3 + tform2 + tform1
    return tform

def dist_coords(coords1, coords2):
    ''' Calculate the point-to-point distances between two coordinates, return a list.
    '''
    return [sum((coords1[i] - coords2[i]) ** 2) ** 0.5 for i in range(len(coords1))]


# %%
def main():

    f_name = '1B_A1'
    img1 = cv2.imread(f'../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/A/test/{f_name}_T.tif',0)
    img2 = cv2.imread(f'../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/p2p_A/{f_name}_R.png',0)
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)
    if img2.ndim == 2:
        img2 = np.expand_dims(img2, axis=-1)
    
    registered_im_out, t = register_aamd(ref_im=img2, flo_im=img1)
    
    w=834    
    coords_ref = np.array(([0,0], [0,w], [w,w], [w,0]))
    centre_patch = np.array((w, w)) / 2. - 0.5
    
#    rot_radian = -0.294152734
#    #rotation_trans = skt.SimilarityTransform(rotation=rot_radian).params[:2, :2]
#    tx_trans, ty_trans = 66.07627139, 77.59790515
#    #translation = skt.SimilarityTransform(translation=(tx_trans, ty_trans)).translation
#    tform_patch = tform_centred(radian=rot_radian, translation=(tx_trans, ty_trans), center=centre_patch)
#    coords_trans = skt.matrix_transform(coords_ref, tform_patch.params)


    df = pd.read_csv('../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/info_test.csv', index_col='Filename')
    coords_trans = df.loc[
            f_name, 
            ['X1_Trans', 'Y1_Trans', 'X2_Trans', 'Y2_Trans', 'X3_Trans', 'Y3_Trans', 'X4_Trans', 'Y4_Trans']
            ].to_numpy().reshape((4, 2))

    coords_rec = transform_coords(t, coords_in=coords_trans, centre_patch=centre_patch)
    dist_coords(coords_ref,coords_rec)
    

#    io.imsave(out_ref_im_path, ref_im_out)
#    io.imsave(out_registered_im_path, registered_im_out)
    io.imshow(registered_im_out)

if __name__ == '__main__':
    start_time = time.time()
    res = main()
    end_time = time.time()
    if res == True:
        print("Elapsed time: " + str((end_time-start_time)))
