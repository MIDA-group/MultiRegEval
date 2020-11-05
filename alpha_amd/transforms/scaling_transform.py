
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
# Scaling Transform
#

import math
import numpy as np
from transforms.transform_base import TransformBase

class ScalingTransform(TransformBase):
    def __init__(self, dim, uniform=True):
        if uniform == True:
            TransformBase.__init__(self, dim, 1)
        else:
            TransformBase.__init__(self, dim, dim)
        self.set_params_const(1.0)
        self.uniform = uniform

    def copy_child(self):
        return ScalingTransform(self.get_dim(), self.uniform)
    
    def transform(self, pnts):
        s = self.get_params()
        return pnts * s

    def grad(self, pnts, gradients, output_gradients):
        res = np.zeros((self.get_param_count(),))
        if self.uniform == True or self.get_dim() == 1:
            res[:] = (pnts * gradients).sum()
            if output_gradients == True:
                upd_gradients = gradients * self.get_params()
                return res, upd_gradients
            else:
                return res
        else:
            res[:] = (pnts * gradients).sum(axis=0)
            if output_gradients == True:
                upd_gradients = gradients * self.get_params()
                return res, upd_gradients
            else:
                return res

    def invert(self):
        self_inv = self.copy()
        self_inv.set_params(1.0 / self_inv.get_params())

        return self_inv

    # (-1.0 / s^2) * (dD/dT^-1)
    def inverse_to_forward_matrix(self):
        param_count = self.get_param_count()
        res = np.zeros((np.square(param_count),))
        
        # Fill diagonal with the derivatives
        res[::param_count+1] = -1.0 / np.square(self.get_params())

        return res.reshape((param_count, param_count))

if __name__ == '__main__':
    #t = ScalingTransform(2, True)
    #t.set_params_const(4.0)
    
    t = ScalingTransform(2, False)
    t.set_param(0, 2.0)
    t.set_param(1, 4.0)

    pnts = np.array([[1.0, 2.0], [3.0, 4.0]])
    spatial_grad = np.array([[1.0, 1.0], [1.0, 1.0]])

    tpnts = t.transform(pnts)
    tgrad = t.grad(pnts, spatial_grad, False)
    tgrad_num = t.grad_num(pnts, spatial_grad)

    print("Transformed pnts: " + str(tpnts))
    print("Gradient: " + str(tgrad))
    print("Gradient num: " + str(tgrad_num))

    tinv = t.invert()

    tinvpnts = tinv.transform(tpnts)
    tinvgrad = tinv.grad(tpnts, spatial_grad, False)
    tinvgrad_num = tinv.grad_num(tpnts, spatial_grad)

    tinvgrad_conv = t.grad_inverse_to_forward(tinvgrad)
    tinvgrad_conv_num = t.grad_inverse_to_forward_num(tinvgrad)
    tinvgrad_conv_num_num = t.grad_inverse_to_forward_num(tinvgrad_num)

    mat = t.inverse_to_forward_matrix()
    mat_num = t.inverse_to_forward_matrix_num()

    print("Inverted pnts: " + str(tinvpnts))
    print("Inverted gradient: " + str(tinvgrad))
    print("Inverted gradient num: " + str(tinvgrad_num))
    print("Inverse to forward gradient: " + str(tinvgrad_conv))
    print("Inverse to forward gradient num: " + str(tinvgrad_conv_num))
    print("Inverse to forward gradient num num: " + str(tinvgrad_conv_num_num))

    print(str(mat))
    print(str(mat_num))


