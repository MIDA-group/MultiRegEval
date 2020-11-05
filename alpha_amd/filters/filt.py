
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
# Image filters
#

# Import Numpy/Scipy
import numpy as np
import scipy as sp
import scipy.misc

def gaussian_filter(image, sigma):
    if sigma <= 0.0:
        return image
    else:
        return scipy.ndimage.filters.gaussian_filter(image, sigma)

def downsample(image, n):
    n = np.int(n) # make sure we have an integer
    if image.ndim == 1:
        return image[0::n]
    elif image.ndim == 2:
        return image[0::n, 0::n]
    elif image.ndim == 3:
        return image[0::n, 0::n, 0::n]
    elif image.ndim == 4:
        return image[0::n, 0::n, 0::n, 0::n]
    else:
        raise 'Not implemented yet for dimensions other than 1-4.'

def _normalize_and_clip(image, mn, mx):
    return np.clip((image-mn)/(mx-mn+(1e-15)), 0.0, 1.0)

def _normalize_with_mask(image, percentile, mask):
    n = image.size
    vec = image.reshape((n,))
    vec_mask = mask.reshape((n,))
    sorted_vec = np.sort(vec[vec_mask])
    m = sorted_vec.size
    mn_ind = np.clip(np.int(n*percentile), 0, m-1)
    mx_ind = np.clip(m-mn_ind-1, 0, m-1)
    return _normalize_and_clip(image, sorted_vec[mn_ind], sorted_vec[mx_ind])    

def _normalize_with_no_mask(image, percentile):
    n = image.size
    vec = image.reshape((n,))
    sorted_vec = np.sort(vec)
    mn_ind = np.clip(np.int(n*percentile), 0, n-1)
    mx_ind = np.clip(n-mn_ind-1, 0, n-1)
    return _normalize_and_clip(image, sorted_vec[mn_ind], sorted_vec[mx_ind])

def _normalize_with_zero_percentile_no_mask(image):
    return _normalize_and_clip(image, np.amin(image), np.amax(image))

def _normalize_with_zero_percentile_with_mask(image, mask):
    return _normalize_and_clip(image, np.amin(image[mask]), np.amax(image[mask]))
    
def normalize(image, percentile=0.0, mask=None):
    if mask is None:
        if percentile == 0.0:
            return _normalize_with_zero_percentile_no_mask(image)
        else:
            return _normalize_with_no_mask(image, percentile)
    else:
        if percentile == 0.0:
            return _normalize_with_zero_percentile_with_mask(image, mask)
        else:
            return _normalize_with_mask(image, percentile, mask)

def channels_to_list(im):
    return [im[..., i] for i in range(im.shape[-1])]

def channels_first_to_list(im):
    return [im[i, ...] for i in range(im.shape[0])]

def list_to_channels(im):
    return np.moveaxis(np.array(im), 0, im[0].ndim)

   

if __name__ == '__main__':
    np.random.seed(1000)

    im = np.arange(24).reshape((4, 6)) / 48.0

    im_comp = im + 0.2
    im_comp = np.random.permutation(im_comp)
    
    # Add outliers
    im_comp[0, 0] = -0.5
    im_comp[3, 4] = 2.0

    print(im_comp)

    res0 = normalize(im_comp, 0.0, None)
    print(res0)

    res1 = normalize(im_comp, 0.1, None)
    print(res1)

    mask = np.zeros(im.shape, 'bool')
    mask[1:-1, 1:-1] = True
    res2 = normalize(im_comp, 0.0, mask)
    print(res2)

    res3 = normalize(im_comp, 0.05, mask)
    print(res3)    

    im1 = np.arange(16*8).reshape((16, 8))
    print(im1)
    im1_ds2 = downsample(im1, 2)
    print(im1_ds2)
    im1_ds4 = downsample(im1, 4)
    print(im1_ds4)    