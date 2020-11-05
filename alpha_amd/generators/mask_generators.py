
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
# Mask generation.
#

# Import Numpy/Scipy
import numpy as np
import scipy as sp
import scipy.misc

def image_center_point(image, spacing):
    shape = image.shape
    if spacing is None:
        return (np.array(shape)-1) * 0.5
    else:
        return ((np.array(shape)-1) * spacing) * 0.5

def move_axis_first_to_last(image):
    ax = [i for i in range(1, image.ndim)] + [0]
    return np.transpose(image, np.array(ax))
    #np.arange(1, image.ndim)

def make_image_grid(image, spacing):
    if spacing is None:
        spacing = np.ones((image.ndim,))
    
    linspaces = [np.linspace(0, image.shape[i]*spacing[i], image.shape[i], endpoint=False) for i in range(image.ndim)]
    grid = np.array(np.meshgrid(*linspaces, indexing='ij'))

    return grid

def make_rect_mask_like_image(image, spacing=None):
    return np.ones(image.shape, dtype='bool')

def make_quad_mask_like_image(image, rad_factor = 1.0, spacing=None):
    mask = np.zeros(image.shape, dtype='bool')
    cp = image_center_point(image, spacing)
    rad = rad_factor * (np.min(cp)+0.5)
    
    grid = make_image_grid(image, spacing)
    grid = move_axis_first_to_last(grid)
    grid_t = np.amax((np.abs(grid-cp)*(1.0/rad)), axis=image.ndim)
    mask[grid_t <= 1.0] = 1.0
    return mask

def make_circular_hann_window_like_image(image, rad_factor = 1.0, spacing=None, p=1.0):
    cp = image_center_point(image, spacing)
    rad = rad_factor * (np.min(cp)+0.5)
    
    grid = make_image_grid(image, spacing)
    grid = move_axis_first_to_last(grid)
    grid_diff = (grid-cp)*(1.0/rad)
    grid_t = np.sqrt(np.square(grid_diff).sum(axis=image.ndim))
    
    outside = grid_t > 1.0
    if p == 1.0:
        hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * (0.5 + 0.5 * grid_t)))
    else:
        hann = np.power(0.5 * (1.0 - np.cos(2.0 * np.pi * (0.5 + 0.5 * grid_t))), p)
    hann[outside] = 0.0

    return hann

# Create a circular mask of the same size as the input image
def make_circular_mask_like_image(image, rad_factor = 1.0, spacing=None):
    mask = np.zeros(image.shape, dtype='bool')
    cp = image_center_point(image, spacing)
    rad = rad_factor * (np.min(cp)+0.5)
    
    grid = make_image_grid(image, spacing)
    grid = move_axis_first_to_last(grid)
    grid_diff = (grid-cp)*(1.0/rad)
    grid_t = np.sqrt(np.square(grid_diff).sum(axis=image.ndim))
    mask[grid_t <= 1.0] = 1.0
    return mask

# Create a ellipse mask of the same size as the input image
def make_ellipse_mask_like_image(image, rad_factor = 1.0, spacing=None):
    mask = np.zeros(image.shape, dtype='bool')
    cp = image_center_point(image, spacing)
    rad = rad_factor * (cp+0.5)

    grid = make_image_grid(image, spacing)
    grid = move_axis_first_to_last(grid)
    grid_diff = (grid-cp)*(1.0/rad)
    grid_t = np.sqrt(np.square(grid_diff).sum(axis=image.ndim))
    mask[grid_t <= 1.0] = 1.0
    return mask

if __name__ == '__main__':
    # Test masks
    em_im = np.ones((20, 10))
    cm = make_circular_mask_like_image(em_im, 0.8, np.array([0.5, 1.0]))
    em = make_ellipse_mask_like_image(em_im, 0.8, np.array([1.0, 1.0]))
    qm = make_quad_mask_like_image(em_im, 0.8, np.array([1.0, 1.0]))

    print(cm)
    print(em)
    print(qm)
