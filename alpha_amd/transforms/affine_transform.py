
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
# Affine Transform
#

import math
import numpy as np
from transforms.transform_base import TransformBase

# Format, in homogeneous coordinates
# | a00 ... a0n t0 | |x1|
# | a10 ... a1n t1 | |x2|
# | ........... .. | |..|
# | an0 ... ann tn | |xn|
# |  0  ...  0  1  | |1 |
#

class AffineTransform(TransformBase):
    def __init__(self, dim):
        TransformBase.__init__(self, dim, (dim*dim) + dim)
        param = np.zeros((dim*(dim+1),))
        param[:dim*dim:dim+1] = 1
        self.set_params(param)

    def copy_child(self):
        return AffineTransform(self.get_dim())
    
    def transform(self, pnts):
        param = self.get_params()
        dim = self.get_dim()
        m = np.transpose(param[:dim*dim].reshape((dim, dim)))
        t = param[dim*dim:]

        return pnts.dot(m) + t

    ### Special affine functions

    def get_matrix(self):
        dim = self.get_dim()
        param = self.get_params()
        
        return param[:dim*dim].reshape((dim, dim))

    def get_translation(self):
        dim = self.get_dim()
        param = self.get_params()
        
        return param[dim*dim:]

    def set_matrix(self, M):
        dim = self.get_dim()
        param = self.get_params()
        
        param[:dim*dim] = M.reshape((dim*dim,))
    def set_translation(self, t):
        dim = self.get_dim()
        param = self.get_params()
        
        param[dim*dim:] = t[:]

    # Generates homogeneous coordinate matrix
    def homogeneous(self):
        dim = self.get_dim()
        param = self.get_params()
        h = np.zeros([dim+1, dim+1])
        
        h[0:dim, 0:dim] = self.get_matrix()#param[:dim*dim].reshape((dim, dim))
        h[:dim, dim] = self.get_translation()#param[dim*dim:]
        h[dim, dim] = 1
        return h

    # Convert from homogeneous coordinate matrix
    def convert_from_homogeneous(self, h):
        dim = self.get_dim()
        self.set_matrix(h[:dim, :dim])
        self.set_translation(h[:dim, dim])

    ### End of Special affine functions

    # Invert transformation
    def invert(self):
        dim = self.get_dim()
        self_inv = AffineTransform(dim)
        h = self.homogeneous()
        h = np.linalg.inv(h)

        self_inv.convert_from_homogeneous(h)
        #self_inv.set_params(h[0:dim+1, 0:dim].reshape((self.get_param_count(),)))

        #self_inv.set_params(h[0:dim, 0:dim])

        return self_inv
        
    def grad(self, pnts, gradients, output_gradients):
        g_out = np.zeros((self.get_param_count(),))

        for i in range(self.dim):
            for j in range(self.dim):
                g_out[i*self.dim + j] = (pnts[:, j] * gradients[:, i]).sum()
        #for i in range(self.dim):
        #    for j in range(self.dim):
        #        g_out[i*self.dim:(i+1)*self.dim] = (pnts[:, j] * gradients[:, i]).sum()
        g_out[(self.dim*self.dim):] = gradients.sum(axis=0)

        if output_gradients == True:
            param = self.get_params()
            dim = self.get_dim()
            m = param[:dim*dim].reshape((dim, dim))

            upd_gradients = gradients.dot(m)

            return g_out, upd_gradients
        else:
            return g_out

    def inverse_to_forward_matrix(self):
        if self.get_dim() == 2:
            return self._inverse_to_forward_matrix_2d(self.get_params())
        elif self.get_dim() == 3:
            return self._inverse_to_forward_matrix_3d(self.get_params())
        else:
            return self.inverse_to_forward_matrix_num()
    
    def _inverse_to_forward_matrix_2d(self, param):
    
        # Generate local variables for each parameter
        a_0_0 = param[0]
        a_0_1 = param[1]
        a_1_0 = param[2]
        a_1_1 = param[3]
        a_0_2 = param[4]
        a_1_2 = param[5]
    
        # Compute determinant
        det = a_0_0*a_1_1 - a_0_1*a_1_0
    
        # Compute and return final matrix
        return np.array(
            [
                [-a_1_1**2/det**2, a_0_1*a_1_1/det**2, a_1_0*a_1_1/det**2, -a_0_1*a_1_0/det**2, -a_1_1*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, (a_1_1*(a_0_0*a_1_2 - a_0_2*a_1_0) - a_1_2*det)/det**2],
                [a_1_0*a_1_1/det**2, -a_0_0*a_1_1/det**2, -a_1_0**2/det**2, a_0_0*a_1_0/det**2, (a_1_0*(a_0_1*a_1_2 - a_0_2*a_1_1) + a_1_2*det)/det**2, -a_1_0*(a_0_0*a_1_2 - a_0_2*a_1_0)/det**2],
                [a_0_1*a_1_1/det**2, -a_0_1**2/det**2, -a_0_0*a_1_1/det**2, a_0_0*a_0_1/det**2, a_0_1*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, (-a_0_1*(a_0_0*a_1_2 - a_0_2*a_1_0) + a_0_2*det)/det**2],
                [-a_0_1*a_1_0/det**2, a_0_0*a_0_1/det**2, a_0_0*a_1_0/det**2, -a_0_0**2/det**2, -(a_0_0*(a_0_1*a_1_2 - a_0_2*a_1_1) + a_0_2*det)/det**2, a_0_0*(a_0_0*a_1_2 - a_0_2*a_1_0)/det**2],
                [0, 0, 0, 0, -a_1_1/det, a_1_0/det],
                [0, 0, 0, 0, a_0_1/det, -a_0_0/det]
            ])

    def _inverse_to_forward_matrix_3d(self, param):
    
        # Generate local variables for each parameter
        a_0_0 = param[0]
        a_0_1 = param[1]
        a_0_2 = param[2]
        a_1_0 = param[3]
        a_1_1 = param[4]
        a_1_2 = param[5]
        a_2_0 = param[6]
        a_2_1 = param[7]
        a_2_2 = param[8]
        a_0_3 = param[9]
        a_1_3 = param[10]
        a_2_3 = param[11]
    
        # Compute determinant
        det = a_0_0*a_1_1*a_2_2 - a_0_0*a_1_2*a_2_1 - a_0_1*a_1_0*a_2_2 + a_0_1*a_1_2*a_2_0 + a_0_2*a_1_0*a_2_1 - a_0_2*a_1_1*a_2_0
    
        # Compute and return final matrix
        return np.array(
            [
                [-(a_1_1*a_2_2 - a_1_2*a_2_1)**2/det**2, (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, -(a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, (a_1_0*a_2_2 - a_1_2*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, (a_2_2*det - (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (-a_1_2*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, -(a_1_0*a_2_1 - a_1_1*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, (-a_2_1*det + (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (a_1_1*det - (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (a_1_1*a_2_2 - a_1_2*a_2_1)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1)/det**2, (det*(a_1_2*a_2_3 - a_1_3*a_2_2) - (a_1_1*a_2_2 - a_1_2*a_2_1)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, (det*(-a_1_1*a_2_3 + a_1_3*a_2_1) + (a_1_1*a_2_2 - a_1_2*a_2_1)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [(a_1_0*a_2_2 - a_1_2*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, -(a_2_2*det + (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (a_1_2*det + (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, -(a_1_0*a_2_2 - a_1_2*a_2_0)**2/det**2, (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, -(a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, (a_1_0*a_2_1 - a_1_1*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, (a_2_0*det - (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (-a_1_0*det + (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (det*(-a_1_2*a_2_3 + a_1_3*a_2_2) - (a_1_0*a_2_2 - a_1_2*a_2_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, (a_1_0*a_2_2 - a_1_2*a_2_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0)/det**2, (det*(a_1_0*a_2_3 - a_1_3*a_2_0) - (a_1_0*a_2_2 - a_1_2*a_2_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [-(a_1_0*a_2_1 - a_1_1*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, (a_2_1*det + (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, -(a_1_1*det + (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, (a_1_0*a_2_1 - a_1_1*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, -(a_2_0*det + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, (a_1_0*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, -(a_1_0*a_2_1 - a_1_1*a_2_0)**2/det**2, (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_0*a_2_1 - a_1_1*a_2_0)/det**2, -(a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_0*a_2_1 - a_1_1*a_2_0)/det**2, (det*(a_1_1*a_2_3 - a_1_3*a_2_1) + (a_1_0*a_2_1 - a_1_1*a_2_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, (det*(-a_1_0*a_2_3 + a_1_3*a_2_0) - (a_1_0*a_2_1 - a_1_1*a_2_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, (a_1_0*a_2_1 - a_1_1*a_2_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0)/det**2],
                [(a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, -(a_0_1*a_2_2 - a_0_2*a_2_1)**2/det**2, (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, -(a_2_2*det + (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, (a_0_2*det - (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_1*a_2_2 - a_0_2*a_2_1))/det**2, (a_2_1*det + (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, -(a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, (-a_0_1*det + (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_1*a_2_2 - a_0_2*a_2_1))/det**2, -(a_0_1*a_2_2 - a_0_2*a_2_1)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1)/det**2, (det*(-a_0_2*a_2_3 + a_0_3*a_2_2) + (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, (det*(a_0_1*a_2_3 - a_0_3*a_2_1) - (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [(a_2_2*det - (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, -(a_0_2*det + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_1*a_1_2 - a_0_2*a_1_1))/det**2, (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, -(a_0_0*a_2_2 - a_0_2*a_2_0)**2/det**2, (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_2_2 - a_0_2*a_2_0)/det**2, -(a_2_0*det + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_0*a_2_2 - a_0_2*a_2_0)/det**2, (a_0_0*det - (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_2_2 - a_0_2*a_2_0))/det**2, (det*(a_0_2*a_2_3 - a_0_3*a_2_2) + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, -(a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0)/det**2, (det*(-a_0_0*a_2_3 + a_0_3*a_2_0) + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [(-a_2_1*det + (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, -(a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, (a_0_1*det + (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_1*a_1_2 - a_0_2*a_1_1))/det**2, (a_2_0*det - (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_0*a_2_2 - a_0_2*a_2_0)/det**2, -(a_0_0*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_2_1 - a_0_1*a_2_0))/det**2, (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_0*a_2_1 - a_1_1*a_2_0)/det**2, -(a_0_0*a_2_1 - a_0_1*a_2_0)**2/det**2, (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_2_1 - a_0_1*a_2_0)/det**2, (det*(-a_0_1*a_2_3 + a_0_3*a_2_1) - (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, (det*(a_0_0*a_2_3 - a_0_3*a_2_0) + (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, -(a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0)/det**2],
                [-(a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, -(a_0_1*a_1_2 - a_0_2*a_1_1)**2/det**2, (a_1_2*det + (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, -(a_0_2*det + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_1*a_1_2 - a_0_2*a_1_1))/det**2, (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, -(a_1_1*det + (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, (a_0_1*det + (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_1*a_1_2 - a_0_2*a_1_1))/det**2, -(a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1)/det**2, (det*(a_0_2*a_1_3 - a_0_3*a_1_2) - (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, (det*(-a_0_1*a_1_3 + a_0_3*a_1_1) + (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [(-a_1_2*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (a_0_2*det - (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_1*a_2_2 - a_0_2*a_2_1))/det**2, (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, -(a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_2_2 - a_0_2*a_2_0)/det**2, -(a_0_0*a_1_2 - a_0_2*a_1_0)**2/det**2, (a_1_0*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, -(a_0_0*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_2_1 - a_0_1*a_2_0))/det**2, (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_1_2 - a_0_2*a_1_0)/det**2, (det*(-a_0_2*a_1_3 + a_0_3*a_1_2) - (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0)/det**2, (det*(a_0_0*a_1_3 - a_0_3*a_1_0) - (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [(a_1_1*det - (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (-a_0_1*det + (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_1*a_2_2 - a_0_2*a_2_1))/det**2, -(a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, (-a_1_0*det + (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (a_0_0*det - (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_2_2 - a_0_2*a_2_0))/det**2, (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_1_2 - a_0_2*a_1_0)/det**2, -(a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_0*a_2_1 - a_1_1*a_2_0)/det**2, (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_2_1 - a_0_1*a_2_0)/det**2, -(a_0_0*a_1_1 - a_0_1*a_1_0)**2/det**2, (det*(a_0_1*a_1_3 - a_0_3*a_1_1) + (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, (det*(-a_0_0*a_1_3 + a_0_3*a_1_0) - (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0)/det**2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, (-a_1_1*a_2_2 + a_1_2*a_2_1)/det, (a_1_0*a_2_2 - a_1_2*a_2_0)/det, (-a_1_0*a_2_1 + a_1_1*a_2_0)/det],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, (a_0_1*a_2_2 - a_0_2*a_2_1)/det, (-a_0_0*a_2_2 + a_0_2*a_2_0)/det, (a_0_0*a_2_1 - a_0_1*a_2_0)/det],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, (-a_0_1*a_1_2 + a_0_2*a_1_1)/det, (a_0_0*a_1_2 - a_0_2*a_1_0)/det, (-a_0_0*a_1_1 + a_0_1*a_1_0)/det]
            ])
    
    def itk_transform_string_rec(self, index):
        s = '#Transform %d\n' % index        
        s = s + 'Transform: AffineTransform_double_%d_%d\n' % (self.get_dim(), self.get_dim())
        s = s + 'Parameters:'
        for i in range(self.get_param_count()):
            s = s + (' %f' % self.get_param(i))
        s = s + '\n'
        s = s + 'FixedParameters:'
        for i in range(self.get_dim()):
            s = s + ' 0.0'
        s = s + '\n'

        return s 

if __name__ == '__main__':
    t = AffineTransform(2)

    M = np.array([[0.9, 0.1], [-0.1, 0.95]])
    T = np.array([2.0, -3.0])
    t.set_matrix(M)
    t.set_translation(T)
    #t.set_param(0, 0.9)
    #t.set_param(1, 0.1)
    #t.set_param(2, -0.1)
    #t.set_param(3, 0.95)
    #t.set_param(4, 2.0)
    #t.set_param(5, -3.0)

    inv_mat = t.inverse_to_forward_matrix()
    inv_mat_num = t.inverse_to_forward_matrix_num()
    diff = np.abs(inv_mat - inv_mat_num).sum()

    np.set_printoptions(suppress = True)

    print(inv_mat)
    print(inv_mat_num)
    print(diff)

    t = AffineTransform(3)

    M = np.array([[0.9, 0.1, 0.05], [-0.1, 0.95, 0.15], [0.25, 0.3, 1.02]])
    T = np.array([2.0, -3.0, 8.0])
    t.set_matrix(M)
    t.set_translation(T)
    
    inv_mat = t.inverse_to_forward_matrix()
    inv_mat_num = t.inverse_to_forward_matrix_num()
    diff = np.abs(inv_mat - inv_mat_num).sum()

    np.set_printoptions(suppress = True)

    print(inv_mat)
    print(inv_mat_num)

    print(diff)
