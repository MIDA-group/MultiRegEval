
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
# Stochastic Distance Transform
#

import numpy as np
import scipy as sp
import scipy.spatial as spatial
from scipy.spatial import cKDTree
import scipy.ndimage as ndimage
import scipy.ndimage.morphology

import time

# Module parameters
MB_TEMP_SIZE = 64.0

# det_sdt - Computes the determinstic SDT for a given image
# ---------------------------------------------------------
# A - numpy array of arbitrary dimension
#     bool (sets) / integer typed (multisets)
# p - the uncertainty factor (see SDT paper)
# k - number of nearest neighbours to consider (if k is not None,
#     m must be None)
# m - the probability mass to capture (if m is not None,
#     k must be None)
# dmax - the maximum distance; a value of 0 means that
#        the diameter of the domain of image A will be used
# spacing - the pixel size (scalar or vector with the same
#           number of elements as dimensions in A)
# processors - the number of processors used in the tree
#              lookup stage
def det_sdt(A, p, k, m = None, dmax=0, spacing=None, processors=1):
    # Exactly one of k and m must be none
    assert((k is None or m is None) and not (k is None and m is None))

    if m is not None:
        k = compute_k(p, m)

    input_shape = A.shape
        
    if spacing is None:
        spacing_vec = np.ones([A.ndim])
    else:
        spacing_vec = spacing.reshape([A.ndim])

    # If no dmax is specified, use the diameter of the image grid
    if dmax <= 0:
        dmax = _grid_diameter(input_shape, spacing_vec)

    if A.dtype == np.dtype('bool'):
        A = A.astype('int8')

    assert(issubclass(A.dtype.type, np.integer))

    pnts_multiplicities = A.reshape([A.size])
    grid_points = _array_grid(A, indexing = 'ij', spacing=spacing)
    G = np.reshape(grid_points, [A.size, A.ndim])

    pnts = point_set_to_multiset(G, pnts_multiplicities)

    # If we have an empty set, just return
    if pnts.shape[0] == 0:
        return np.full(input_shape, fill_value=dmax, dtype='float32')
    # If we have a singleton set, just return the distances to this point
    if pnts.shape[0] == 1:
        return (np.sqrt(np.sum(np.square(G - pnts), axis=-1)) * (1.0-p) + p * dmax).reshape(input_shape)
    # If we have fewer than k points, reduce k
    if pnts.shape[0] < k:
        k = pnts.shape[0]
    
    tree = cKDTree(pnts)
    
    prob = np.array([np.power(p, (i-1)) * (1-p) for i in range(1, k+1)])
    
    subset_size = int((MB_TEMP_SIZE * 1024.0 * 1024.0) / (k*8))
    subsets = int(np.ceil(A.size / float(subset_size)))

    drem = dmax * np.power(p, k)
    res = np.full(shape=[A.size], fill_value=drem, dtype = 'float32')

    # To save on required working memory, we only do the k-NN search
    # and subsequent computation of the distances aggregate distances
    # on subsets of the points, which are then written to the result
    # array.
    for ss_ind in range(subsets):
        start_index = ss_ind * subset_size
        end_index = min(start_index + subset_size, A.size)
        res[start_index:end_index] += _knn_query(tree, G[start_index:end_index, :], k, dmax, prob, processors)

    return res.reshape(A.shape)

"""
# Computes the sum pooling of an image subject to the neighborhood
# corresponding to a downsampling factor, and then downsample
# the image.
def sum_pool(A, pool_size, stride, padding_mode='valid'):
    if stride is None:
        stride = pool_size

    # Create the convolution kernel
    kernel = np.ones([pool_size]*A.ndim)
    # Perform the convolution
    B = ndimage.convolve(A, kernel, mode='constant', cval = 0)

    # Create a slicer object depending on the chosen padding mode
    if padding_mode == 'zero':
        slicer = tuple(slice(0, A.shape[i], stride) for i in range(A.ndim))
    elif padding_mode == 'valid':
        slicer = tuple(slice(0, A.shape[i] - A.shape[i] % stride, stride) for i in range(A.ndim))
    else:
        raise ValueError('padding_mode must be either \'zero\' or \'valid\'')
    
    # Slice and return the output
    return B[slicer]
"""
# Computes the sum pooling of an image subject to the neighborhood
# corresponding to a downsampling factor, and then downsample
# the image.
def sum_pool(A, pool_size, stride, padding_mode='valid'):
    if stride is None:
        stride = pool_size

    if A.dtype == np.dtype('bool'):
        A = A.astype(dtype='int32')
    
    # Create the convolution kernel
    kernel = np.ones([pool_size]*A.ndim)
    # Perform the convolution
    origin = int(pool_size/2) - (1 - (pool_size % 2))
    B = ndimage.convolve(A, kernel, mode='constant', cval = 0, origin = origin)

    # Create a slicer object depending on the chosen padding mode
    if padding_mode == 'zero':
        slicer = tuple(slice(0, A.shape[i], stride) for i in range(A.ndim))
    elif padding_mode == 'valid':
        slicer = tuple(slice(0, A.shape[i] - A.shape[i] % stride, stride) for i in range(A.ndim))
    else:
        raise ValueError('padding_mode must be either \'zero\' or \'valid\'')
    
    # Slice and return the output
    return B[slicer]

# Computes the sum pooling of an image subject to the neighborhood
# corresponding to a downsampling factor, and then downsample
# the image.
def or_pool(A, pool_size, stride, padding_mode='valid'):
    return sum_pool(A, pool_size, stride, padding_mode) > 0

def compute_k(p, m):
    if np.isclose(p, 0.0):
        return 1
    else:
        return int(np.ceil(np.log(1-m)/np.log(p)))

def det_sdt_multiset_naive(A, p, k, dmax=0, spacing=None):
    if spacing is None:
        spacing_vec = np.ones([A.ndim])
    else:
        spacing_vec = spacing.reshape([A.ndim])

    # If no dmax is specified, use the diameter of the image grid
    if dmax <= 0:
        dmax = _grid_diameter(A.shape, spacing_vec)#np.sqrt(np.sum([np.power((A.shape[i]*spacing_vec[i])-1, 2.0) for i in range(A.ndim)]))

    pnt_tup = np.nonzero(A)
    pnts_multiplicities = A[pnt_tup].astype(dtype='int32')
    
    total_pnts_count = np.sum(pnts_multiplicities) 
    if k > total_pnts_count:
        k = total_pnts_count

    G = np.reshape(_array_grid(A, indexing = 'xy', spacing=spacing), [A.size, A.ndim])
    pnts = point_set_to_multiset(np.transpose(pnt_tup) * (spacing_vec), pnts_multiplicities)

    prob = np.array([np.power(p, (i-1)) * (1-p) for i in range(1, k+1)])

    res = np.zeros([A.size])
    for i in range(A.size):
        grid_p = G[i, :]
        dists = np.zeros([total_pnts_count])
        #print(total_pnts_count)
        #print(np.sum(pnts_multiplicities))
        #print(pnts.shape[0])
        for j in range(total_pnts_count):
            obj_p = pnts[j, :]
            d = np.sqrt(np.sum(np.square(obj_p-grid_p)))
            dists[j] = d
        dists=np.sort(dists)
        res[i] = np.sum(prob * dists[:k])
    
    drem = dmax * np.power(p, k)
    res = res + drem

    return res.reshape(A.shape)

# Helper functions

def _array_grid(A, indexing = 'ij', spacing=None):
    if spacing is None:
        spacing = np.ones([A.ndim])
    ranges = [spacing[i] * np.arange(0, A.shape[i]) for i in range(A.ndim)]
    ax = range(1, A.ndim+1) + [0]
    return np.transpose(np.meshgrid(*ranges, indexing = indexing), axes=ax)
    #return np.transpose(np.meshgrid(*ranges, indexing = indexing))

def _grid_diameter(Sz, spacing=None):
    n = len(Sz)
    if spacing is None:
        spacing = np.ones([n])
    assert(spacing.size == n)

    return np.sqrt(np.sum([np.square((Sz[i]-1)*spacing[i]) for i in range(n)])).astype('float32')

def _knn_query(T, pnts, k, dmax, probabilities, processors):
    d,_ = T.query(pnts, k=k, distance_upper_bound=dmax, n_jobs=processors)
    np.clip(d, a_min=None, a_max=dmax, out = d)
    
    dp = d.dot(probabilities)
    return dp

def point_set_to_multiset(pnts, multiplicities):
    return np.repeat(pnts, repeats=multiplicities, axis=0)

# Test cases
def test_empty_set_2d():
    A = np.zeros([3, 4], dtype='int32')
    dmax = _grid_diameter(A.shape, spacing=np.array([1.0, 1.0]))
    D = det_sdt(A, 0.5, k = None, m = 0.99, dmax = 0, spacing = None, processors = 1)
    assert(np.all(D == dmax))

def test_singleton_set_2d():
    A = np.zeros([4, 5], dtype='int32')
    A[1, 2] = 1
    dmax = _grid_diameter(A.shape, spacing=np.array([1.0, 1.0]))
    D = det_sdt(A, 0.5, k = None, m = 0.99, dmax = 0, spacing = None, processors = 1)
    print(D)
    #assert(np.all(D == dmax))

def test_empty_set_3d():
    A = np.zeros([3, 4, 5], dtype='int32')
    dmax = _grid_diameter(A.shape, spacing=np.array([1.0, 1.0, 1.0]))
    D = det_sdt(A, 0.5, k = None, m = 0.99, dmax = 0, spacing = None, processors = 1)
    assert(np.all(D == dmax))

def test_singleton_set_3d():
    A = np.zeros([4, 5, 6], dtype='int32')
    A[1, 2, 2] = 1
    dmax = _grid_diameter(A.shape, spacing=np.array([1.0, 1.0, 1.0]))
    D = det_sdt(A, 0.5, k = None, m = 0.99, dmax = 0, spacing = None, processors = 1)
    print(D)
    #assert(np.all(D == dmax))

def main():
    # run test-cases
    test_empty_set_2d()
    test_singleton_set_2d()
    #return

    sz = 1024
    np.random.seed(1024)
    #B = np.random.rand(sz, sz) >= 0.75
    shp = (256, 256, 124)
    dim = len(shp)
    B = np.random.rand(*shp) >= 0.75
    print(B)
    start1 = time.time()
    res_dt = sp.ndimage.morphology.distance_transform_edt(-B)
    end1 = time.time()
    start2 = time.time()
    print(end1-start1)

    dmax = 0

    downsampling = 4
    spacing = float(downsampling)
    C = sum_pool(B, downsampling, downsampling, 'zero')

    slicer = tuple(slice(int(C.shape[i]/2)-5, int(C.shape[i]/2)+5, 1) for i in range(C.ndim))
    slicer2 = tuple(slice(int(C.shape[i]/2)-5, int(C.shape[i]/2)+5, 1) for i in range(C.ndim))
    print('C')
    print(C[slicer])

    p = 0.9
    m = 0.999
    k = compute_k(p, m)
    processors = 2
    spacing_vec = np.full([dim], fill_value=spacing)

    start2 = time.time()
    res = det_sdt(C, p, k = None, m = m, dmax = dmax, spacing = spacing_vec, processors = processors)
    end2 = time.time()

    print('res')
    print(res[slicer])
    print(end2-start2)
    
    return

    start3 = time.time()
    res2 = det_sdt(B, p, k, dmax=dmax, spacing=np.ones([dim]), processors=processors)
    end3 = time.time()
    res2 = res2[int(downsampling / 2)::downsampling, int(downsampling / 2)::downsampling, int(downsampling / 2)::downsampling]
    #end2 = time.time()

    #print res.shape
    #print res2.shape
    #return
    print('res2')
    print(res2[slicer2])
    print(end3-start3)

if __name__ == '__main__':
    main()
    
