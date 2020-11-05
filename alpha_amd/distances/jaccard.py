
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
# Jaccard index functions
#

import numpy as np
import scipy as sp
from scipy.spatial import distance

def jaccard_index(ref, flo):
    ref_1d = ref.reshape([ref.size])
    flo_1d = flo.reshape([flo.size])
    
    values = np.unique(np.concatenate((ref_1d, flo_1d)))

    # If no background
    if values[0] > 0:
        values = np.insert(values, 0, 0, axis=0)
    
    jac = np.zeros([values.size+1])

    for i in range(values.size):
        ref_1d_bin = (ref_1d == values[i])
        flo_1d_bin = (flo_1d == values[i])
        jac[i] = distance.jaccard(ref_1d_bin, flo_1d_bin)

    jac[values.size] = distance.jaccard(ref_1d > 0, flo_1d > 0)

    return 1-jac

def main():
    ref_im = np.zeros([4, 4], dtype='int32')
    flo_im = np.zeros([4, 4], dtype='int32')

    ref_im[2, 2] = 1
    flo_im[2, 2] = 1

    ref_im[2, 3] = 2
    flo_im[2, 3] = 3

    ref_im[3, 3] = 3
    flo_im[3, 3] = 2
    
    ref_im[0, 0] = 3
    flo_im[0, 0] = 3

    ref_im[1, 1] = 1

    jac = jaccard_index(ref_im, flo_im)

    print(jac)

if __name__ == '__main__':
    main()

