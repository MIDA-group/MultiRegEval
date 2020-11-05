
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
# Rotate2d Transform
#

import math
import numpy as np
from transforms.transform_base import TransformBase

class Rotate2DTransform(TransformBase):
    def __init__(self):
        TransformBase.__init__(self, 2, 1)

    def copy_child(self):
        return Rotate2DTransform()
    
    def transform(self, pnts):
        theta = self.get_param(0)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        M = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        
        return pnts.dot(M)

    def grad(self, pnts, gradients, output_gradients):
        res = np.zeros((1,))
        theta = self.get_param(0)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        Mprime = np.array([[-sin_theta, cos_theta], [-cos_theta, -sin_theta]])
        Mprimepnts = pnts.dot(Mprime)
        res[:] = (Mprimepnts * gradients).sum()

        if output_gradients == True:
            M = np.transpose(np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]]))
        
            return res, gradients.dot(M)
        else:
            return res

    def invert(self):
        self_inv = self.copy()
        self_inv.set_params(-self_inv.get_params())

        return self_inv

    def inverse_to_forward_matrix(self):
        return np.array([[-1.0]])
        
    #def grad_inverse_to_forward(self, inv_grad):
    #    res = np.zeros((1,))
    #    res[:] = -inv_grad
    #    return res
