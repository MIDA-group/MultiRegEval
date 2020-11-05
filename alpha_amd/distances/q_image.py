
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
# Quantized image representation.
#

import numpy as np
import math
import sys

### class: QuantizedImage
### Represents an image quantized to a given number of equally sized levels
### with related point-sets (with a weight for each point) partitioned by
### level.
###
### Random sampling (with replacement) which preserves the partitioning
### so that a list of randomly selected points for each distinct level
### is obtained.

class QuantizedImage:
    def __init__(self, A, alpha_levels, weights, spacing, remove_zero_weight_pnts = True, center_point = None, contraction_factor=1):
        if A.dtype == 'int32':
            self.im = A
        else:
            self.im = (1 + np.floor(A * (alpha_levels) - 0.5)).astype('int')
        self.alpha_levels = alpha_levels
        self.contraction_factor = contraction_factor
        self.distance_shape = tuple([int(A.shape[i] / contraction_factor) for i in range(A.ndim)])
        
        self.spacing = spacing
        if remove_zero_weight_pnts == True:
            self.im[weights <= 0.0] = -1
        #linspaces = [np.linspace(0, A.shape[i]*self.spacing[i], A.shape[i], endpoint=False) for i in range(A.ndim)]
        linspaces = [make_space_1d(A.shape[i], contraction_factor, self.spacing[i]) for i in range(A.ndim)]

        #self.spacing = self.spacing * contraction_factor

        if center_point is None:
            center_point = self.spacing * (np.array(self.get_image_shape())-1.0) / 2.0
        # Store the center point as a member, and
        # bake the contraction offset into the stored offset
        self.center_point = center_point + get_contraction_offset(contraction_factor, spacing)

        grid1 = np.array(np.meshgrid(*linspaces,indexing='ij'))
        grid = np.zeros(A.shape + (A.ndim+1,), dtype = 'float64')
        for i in range(A.ndim):
            grid[..., i] = grid1[i, ...] - center_point[i]
        #grid[..., :-1] = grid[..., :-1] - center_point
        grid[..., A.ndim] = weights

        self.weights = weights
        self.dense_point_count = np.prod(A.shape)

        self.point_count = 0
        self.freq = np.zeros((alpha_levels+1,), dtype = 'int')
        self.pnts = []
        all_indices = np.arange(np.prod(A.shape)).reshape(self.get_image_shape())
        self.indices = []#np.arange(np.prod(A.shape[i]))
        self.grid = grid[..., :]
        for i in range(alpha_levels+1):
            filt = (self.im == i)
            if remove_zero_weight_pnts:
                filt[weights <= 0.0] = False
            cnt = np.count_nonzero(filt)
            self.freq[i] = cnt
            self.point_count = self.point_count + cnt
            filt_pnts = grid[filt]
            filt_indices = all_indices[filt]
            self.pnts.append(filt_pnts)
            self.indices.append(filt_indices)
        if self.point_count > 0:
            self.freq = self.freq * (1.0 / self.point_count)
        else:
            self.freq = self.freq
        self.grid = self.grid.reshape((self.dense_point_count, A.ndim+1))

    def print_image(self):
        print(self.im)

    def print_point_sets(self):
        print(self.pnts)
    
    def get_distance_shape(self):
        return self.distance_shape

    def get_alpha_levels(self):
        return self.alpha_levels

    def get_weights(self):
        return self.weights
    
    def get_dense_point_count(self):
        return self.dense_point_count
    
    def get_image_dim(self):
        return self.im.ndim
    
    def get_image_shape(self):
        return self.im.shape
    
    def get_image(self):
        return self.im

    def get_center_point(self):
        return self.center_point

    def get_sampling_fraction_count(self, f):
        return np.int(np.round(f * self.point_count))

    def get_grid(self):
        return self.grid
    
    def get_indices(self):
        return self.indices
    
    def random_from_level(self, n, level):
        arr = self.pnts[level]
        m = arr.shape[0]
        if m == 0:
            return arr
        else:
            return arr[np.random.random_integers(0, m-1, n), :]
    def random_integers(self, m, n):
        if m > 0 and n > 0:
            return np.random.random_integers(0, m, n)
        else:
            return np.zeros((0, self.pnts[0].shape[1]), dtype='int')


    def random_sample(self, n):
        if n == self.point_count:
            return self.pnts
        else:
            cnt = np.random.multinomial(n, self.freq)
            return [self.pnts[i][self.random_integers(len(self.pnts[i])-1, cnt[i]), :] for i in range(self.alpha_levels+1)]

### Helper functions ###

# Generate a space of n pixels, with given spacing and contraction_factor
# contraction_factor means compressing a number of pixels into the center
# of the super-pixel.
#
# 0, 1, 2, 3, 4, 5, 6 -> 1, 1, 1, 4, 4, 4, 6, 6 (n=7, spacing=1, contraction_factor=3)
def make_space_1d(n, contraction_factor, spacing):
    assert(n > 0)
    assert(contraction_factor >= 1)
    assert(spacing > 0.0)
    
    if contraction_factor == 1:
        return np.linspace(0, n*spacing, n, endpoint=False)
    
    superpix_spacing = spacing * contraction_factor
    
    midp = np.mean(np.arange(contraction_factor)*spacing)

    whole_pix = int(n / contraction_factor)
    rem = n - whole_pix * contraction_factor

    seq_whole_pix = midp + (np.arange(whole_pix) * superpix_spacing)
    exp_seq = np.repeat(seq_whole_pix, contraction_factor)

    if rem > 0:
        out = np.empty(shape=[n])

        # Fill the first part with the expanded sequence
        out[0:exp_seq.size] = exp_seq[:]
        
        # Fill the remainder with constant value
        out[exp_seq.size:] = (midp + (whole_pix * superpix_spacing))

        return out
    else:
        return exp_seq

# Compute the offset required for look-ups due to contraction/fusing
# of pixels, which is dependent on contraction_factor and spacing.
def get_contraction_offset(contraction_factor, spacing=None):
    if spacing is None:
        return -(contraction_factor-1.0) / 2.0
    else:
        return -(spacing * (contraction_factor-1.0)) / 2.0    

# A few tests to exercise the code

def main():
    test_space_1d_a = make_space_1d(11, 2.0, 1)
    test_space_1d_b = make_space_1d(11, 2.0, 2)
    test_space_1d_c = make_space_1d(11, 2.0, 3)
    test_space_1d_d = make_space_1d(11, 2.0, 4)
    test_space_1d_e = make_space_1d(12, 2.0, 4)

    print(test_space_1d_a)
    print(test_space_1d_b)
    print(test_space_1d_c)
    print(test_space_1d_d)
    print(test_space_1d_e)
    
    return

    im = np.zeros((3, 4, 5))
    w = np.ones((3, 4, 5))
    im[1, 1, 3] = 1.0
    im[1, 2, 3] = 1.0
    q_im = QuantizedImage(im, 1, w, [2, 2, 2], True)

    np.random.seed(1000)

    sampling = q_im.random_sample(13)
    print(sampling)

if __name__ == '__main__':
    main()