
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
# Identity Transform
#

import math
import numpy as np
from transforms.transform_base import TransformBase

class IdentityTransform(TransformBase):
    def __init__(self, dim):
        TransformBase.__init__(self, dim, 0)

    def copy_child(self):
        return IdentityTransform(self.get_dim())
    
    def transform(self, pnts):
        return pnts

    def grad(self, pnts, gradients, output_gradients):
        #res = gradients.sum(axis=0)
        if output_gradients == True:
            return np.array([]), gradients
        else:
            return np.array([])

    def invert(self):
        self_inv = self.copy()

        return self_inv

    def inverse_to_forward_matrix(self):
        return np.array([])

    def itk_transform_string_rec(self, index):
        s = '#Transform %d\n' % index
        s = s + 'Transform: IdentityTransform_double_%d_%d\n' % (self.get_dim(), self.get_dim())
        s = s + 'Parameters:\n'
        s = s + 'FixedParameters:\n'

        return s 

if __name__ == '__main__':
    t = IdentityTransform(2)

    print(t.get_param_count())
    print(t.get_params())

    pnts = np.array([[2.0, 3.0]])

    tpnts = t.transform(pnts)

    print(tpnts)

    tinv = t.invert()

    print(tinv.get_params())

    tinvpnts = tinv.transform(tpnts)

    print(tinvpnts)

    dd_dv = np.array([[2, 4]])

    # forward grad
    print(t.grad(tpnts, dd_dv, False))

    # inverse grad    
    print(tinv.grad_inverse_to_forward(tinv.grad(tpnts, np.array([-1,1])*dd_dv, False)))
