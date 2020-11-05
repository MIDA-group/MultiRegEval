
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
# Base class of transformations
#

import math
import numpy as np
import scipy.ndimage.interpolation

class TransformBase:

    def __init__(self, dim, nparam):
        self.dim = dim
        self.param = np.zeros((nparam,))

    def get_dim(self):
        return self.dim

    def get_params(self):
        return self.param
    
    def set_params(self, params):
        self.param[:] = params[:]

    def get_param(self, index):
        return self.param[index]
    
    def set_param(self, index, value):
        self.param[index] = value

    def set_params_const(self, value):
        self.param[:] = value

    def step_param(self, index, step_length):
        self.param[index] = self.param[index] + step_length

    def step_params(self, grad, step_length):
        self.param = self.param + grad * step_length

    def get_param_count(self):
        return self.param.size

    def copy(self):
        t = self.copy_child()
        t.set_params(self.get_params())

        return t

    def copy_child(self):
        raise NotImplementedError

    def __call__(self, pnts):
        return self.transform(pnts)
    
    def transform(self, pnts):
        raise NotImplementedError

    def warp(self, In, Out, in_spacing=None, out_spacing=None, mode='spline', bg_value=0.0):
        linspaces = [np.linspace(0, Out.shape[i]*out_spacing[i], Out.shape[i], endpoint=False) for i in range(Out.ndim)]

        grid = np.array(np.meshgrid(*linspaces,indexing='ij'))

        grid = grid.reshape((Out.ndim, np.prod(Out.shape)))
        grid = np.moveaxis(grid, 0, 1)#.reshape((np.prod(Out.shape), Out.ndim))
        #grid = np.transpose(grid).reshape((np.prod(Out.shape), Out.ndim))
        #print(grid)

        grid_transformed = self.transform(grid)
        if in_spacing is not None:
            grid_transformed[:, :] = grid_transformed[:, :] * (1.0 / in_spacing[:])
        
        grid_transformed = np.moveaxis(grid_transformed, 0, 1)
        grid_transformed = grid_transformed.reshape((Out.ndim,) + Out.shape)
        #np.transpose(grid_transformed).reshape((Out.ndim,) + Out.shape)
        #print(grid_transformed)
        
        if mode == 'spline':
            scipy.ndimage.interpolation.map_coordinates(In, coordinates=grid_transformed, output=Out, cval = bg_value)
        elif mode == 'linear':
            scipy.ndimage.interpolation.map_coordinates(In, coordinates=grid_transformed, output=Out, order=1, cval = bg_value)
        elif mode == 'nearest':
            scipy.ndimage.interpolation.map_coordinates(In, coordinates=grid_transformed, output=Out, order=0, cval = bg_value)

    def itk_transform_string(self):
        s = '#Insight Transform File V1.0\n'
        #s = s + '#Transform 0\n'
        return s + self.itk_transform_string_rec(0)

    def itk_transform_string_rec(self, index):
        raise NotImplementedError        

    def grad(self, pnts, gradients, output_gradients):
        raise NotImplementedError

    def invert(self):
        raise NotImplementedError

    # Must be called on the forward transform
    # Default falls back on numerical differentiation
    def grad_inverse_to_forward(self, inv_grad):
        D = self.inverse_to_forward_matrix()
        if D is None:
            D = self.inverse_to_forward_matrix_num()
        #print(D)
        return D.dot(inv_grad)

    def inverse_to_forward_matrix(self):
        return None

    def diff(self, index, pnts, eps=1e-6):
        f = self.copy()
        b = self.copy()
        f.step_param(index, eps)
        b.step_param(index, -eps)
        fpnts = f.transform(pnts)
        bpnts = b.transform(pnts)
        delta = (fpnts - bpnts) * (1.0 / (2.0 * eps))
        return delta

    def grad_num(self, pnts, gradients, eps=1e-6):
        res = np.zeros((self.get_param_count(),))
        if self.get_param_count() == 1:
            d = self.diff(0, pnts, eps)
            res[0] = res[0] + (d * gradients).sum()
        else:
            for i in range(self.get_param_count()):
                d = self.diff(i, pnts, eps)
                summed = (d * gradients).sum()
                res[i] = res[i] + summed
        return res
        
    # Utility function to differentiate the inverse transformation
    # with respect to the forward transformation numerically
    def diff_inv(self, index, eps=1e-6):
        f = self.copy()
        b = self.copy()
        f.step_param(index, eps)
        b.step_param(index, -eps)
        return (f.invert().get_params() - b.invert().get_params()) / (2.0 * eps)

    def inverse_to_forward_matrix_num(self, eps=1e-6):
        D = np.zeros((self.get_param_count(),self.get_param_count()))
        for i in range(self.get_param_count()):
            d = self.diff_inv(i, eps)
            D[i, :] = d
        return D
        
    # Must be called on the forward transform
    def grad_inverse_to_forward_num(self, inv_grad, eps=1e-6):
        D = self.inverse_to_forward_matrix_num(eps)
        #print(D)
        G = D.dot(inv_grad)
        #print(G)
        return G
        #res = np.zeros((self.get_param_count(),))
        #for i in range(self.get_param_count()):
        #    d = self.diff_inv(i, eps)
        #    print("d(%d): %s" % (i, str(d)))
        #    res[i] = (d.dot(inv_grad))#.sum()
        #return res



if __name__ == '__main__':
    t = TransformBase(2, 3)

    print(t.get_param_count())
    print(t.get_params())
