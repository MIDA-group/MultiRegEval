
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
# Alpha distance and gradient transform data-structure/implementation.
#

import numpy as np
import math
import sys
import scipy.ndimage
import scipy.ndimage.morphology as morph
import scipy.interpolate as interpolate
import distances.q_image

def edt(A, spacing):
    return morph.distance_transform_edt(np.invert(A), spacing)

def edt_sq(A, spacing):
    return np.square(morph.distance_transform_edt(np.invert(A), spacing))

def alpha_distance_transform(qimage, alpha_levels, dshape, spacing, mask = None, dmax = 0.0, dt_fun = edt, mask_out_edges = True):
    #print(qimage)
    # Apply mask if supplied
    if mask is not None:
        qimage = qimage * mask.astype('int')
    #print(qimage)

    # The number of alpha levels extended with a 'zero' level
    ex_alpha_levels = alpha_levels+1
    
    # Store the image dimensionality in a local variable
    dim = qimage.ndim

    # Compute the shape of the gradient image
    grad_shape = dshape + (dim,)
    
    # Compute the shape of the grad and distance image
    dist_grad_shape = dshape + (dim+1,)

    # The thickness is given by the number of non-zero levels
    thickness = 1.0 / (alpha_levels)

    # Create the array and initialize with a zero first level
    dg_tf = [np.zeros(dist_grad_shape, dtype = 'float32')]

    d = np.zeros(dshape, dtype = 'float32')

    # Construct the alpha gradient and distance transform
    for i in range(1, ex_alpha_levels):
        b = (qimage >= i)
        #assert(np.any(b))
        #assert(np.any(np.invert(b)))
        
        # Compute the euclidean distance transform for alpha-cut
        d[:] = dt_fun(b, spacing)        # Compute gradient of distance transform
        #morph.distance_transform_edt(b, spacing)
        if dmax > 0.0:
            np.clip(d, None, dmax, out = d)
        
        grad_im = np.gradient(d, *spacing)

        #print(grad_im)

        # Create the image for the level (with gradient and distance)
        level = np.zeros(dist_grad_shape, dtype = 'float32')

        # Create indicator function multiplier
        if mask_out_edges == True:
            b_comp_float = (1.0-b.astype('float32'))

        # Write each gradient component into level image (with zeroing of
        # gradients for object points).
        for k in range(dim):
            if mask_out_edges == True:
                level[..., k] = grad_im[k] * b_comp_float
            else:
                level[..., k] = grad_im[k]

        # Write the distance as the last entry
        level[..., dim] = d

        # Multiply the level vector field by the level thickness
        # and add the previous level's cumulative sum.
        level[:] = (thickness * level[:]) + dg_tf[i-1]

        # Append the level to the alpha-grad-distance transform array
        dg_tf.append(level)
    
    # Return the computed alpha-grad-distance transform array
    return dg_tf

def alpha_distance_transform_bd(q, alpha_levels, dshape, spacing, mask=None, dmax=0.0, dt_fun=edt, mask_out_edges=True):
    dg_tf1 = alpha_distance_transform(q.get_image(), alpha_levels, dshape, spacing, mask, dmax, dt_fun, mask_out_edges)
    dg_tf2 = alpha_distance_transform(alpha_levels - q.get_image(), alpha_levels, dshape, spacing, mask, dmax, dt_fun, mask_out_edges)
    for i in range(alpha_levels+1):
        dg_tf1[i][:] = dg_tf1[i][:] + dg_tf2[alpha_levels-i][:]
    return dg_tf1

def interp_vec(A, p, order=1):
    sdim = p.shape[0]
    d = A.shape[A.ndim-1]
    
    results = np.zeros((p.shape[1], d), dtype='float64')
    c = np.zeros((1,), dtype = A.dtype)

    for k in range(d):
        scipy.ndimage.interpolation.map_coordinates(A[..., k], coordinates=p, output = results[..., k], order = order, cval = c, prefilter = False)

    return results

def interp_scalar(A, p, order=1):
    c = np.zeros((1,), dtype = A.dtype)
    results = scipy.ndimage.interpolation.map_coordinates(A, coordinates=p, mode = 'constant', order = order, cval = c, prefilter = False)

    return results

class AlphaAMD:
    def __init__(self, Q, alpha_levels, dmax, spacing=None, mask_pre=None, mask_post=None, interpolator_mode='linear', dt_fun=None, mask_out_edges = True):
        if dt_fun is None:
            dt_fun = edt
        
        self.dim = Q.get_image_dim()
        self.alpha_levels = alpha_levels
        self.dmax = dmax
        shape = Q.get_image_shape()
        if spacing is None:
            self.spacing = np.ones(self.dim)
        else:
            self.spacing = np.array(spacing)
        if mask_pre is None:
            self.mask_pre = np.ones(shape, dtype='boolean')
        if mask_post is None:
            self.mask_post = np.ones(shape, dtype='boolean')
        self.mask_pre = mask_pre
        self.mask_post = mask_post
        if interpolator_mode == 'nearest':
            self.interpolator_order = 0
        elif interpolator_mode == 'linear':
            self.interpolator_order = 1
        else:
            raise 'Illegal interpolator mode'
        self.alpha_dg_tf = alpha_distance_transform_bd(Q, self.alpha_levels, Q.get_distance_shape(), self.spacing, self.mask_pre, self.dmax, dt_fun, mask_out_edges)

    def compute_spatial_grad_and_value(self, pnts, w, q):
        scaled_pnts = np.transpose(pnts * (1.0 / self.spacing))
        #scaled_pnts = pnts

        evaluated_pnts = interp_vec(self.alpha_dg_tf[q], scaled_pnts, order = self.interpolator_order)

        evaluated_mask = interp_scalar(self.mask_post, scaled_pnts, order = 0).reshape(w.shape).astype('float64')

        evaluated_mask[:] = w[:] * evaluated_mask[:]

        #evaluated_pnts = np.transpose(evaluated_pnts)
        #evaluated_pnts[..., :] = evaluated_pnts[..., :] * evaluated_mask[..., :]
        for i in range(evaluated_pnts.shape[1]):
            eval_i = evaluated_pnts[..., i]
            eval_mask = evaluated_mask.reshape([evaluated_mask.size])
            evaluated_pnts[..., i] = eval_i * eval_mask
        
        return (evaluated_pnts, evaluated_mask)

    def get_gradient_and_distance_transform(self):
        return self.alpha_dg_tf

if __name__ == '__main__':
    alpha_levels = 1
    im = np.zeros((3, 4))
    print(im.ndim)
    im[0, 1] = 1.0
    im[0, 2] = 1.0
    spacing = [2, 3]

    mask = np.ones((3, 4))
    mask[0, 2] = 0

    weights = np.ones((3, 4))
    #A, alpha_levels, weights, spacing, remove_zero_weight_pnts = True):
    qimage = q_image.QuantizedImage(im, alpha_levels, weights, spacing, True)
    qimage.print_image()

    metric = AlphaAMD(qimage, alpha_levels, 128, spacing, mask)
