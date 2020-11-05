
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
# Rigid 2d Transform
#

import math
import numpy as np
from transforms.transform_base import TransformBase

class Rigid2DTransform(TransformBase):
    def __init__(self):
        TransformBase.__init__(self, 2, 3)

    def copy_child(self):
        return Rigid2DTransform()
    
    def transform(self, pnts):
        param = self.get_params()
        theta = param[0]
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        M = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        
        res = pnts.dot(M)
        res[..., :] = res[..., :] + param[1:]
        return res
    '''
    def transform(self, pnts):
        res = np.zeros_like(pnts)
        param = self.get_params()
        theta = param[0]
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        M = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        
        pnts.dot(M, out = res)
        res[..., :] = res[..., :] + param[1:]
        return res
        #return pnts.dot(M) + param[1:]
    '''
    def grad(self, pnts, gradients, output_gradients):
        res = np.zeros((3,))
        theta = self.get_param(0)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        Mprime = np.array([[-sin_theta, cos_theta], [-cos_theta, -sin_theta]])
        #Mprimepnts = pnts.dot(Mprime)
        res[0] = (pnts.dot(Mprime) * gradients).sum()
        res[1:] = gradients.sum(axis=0)

        if output_gradients == True:
            M = np.transpose(np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]]))
        
            return res, gradients.dot(M)
        else:
            return res

    def transform_jacobian(self, pnts, output_gradients):
        res = np.zeros((3,2))
        theta = self.get_param(0)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        Mprime = np.array([[-sin_theta, cos_theta], [-cos_theta, -sin_theta]])
        #Mprimepnts = pnts.dot(Mprime)
        res[0] = (pnts.dot(Mprime) * gradients).sum()
        res[1:] = gradients.sum(axis=0)

        if output_gradients == True:
            M = np.transpose(np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]]))
        
            return res, gradients.dot(M)
        else:
            return res


    def invert(self):
        self_inv = self.copy()

        inv_theta = -self.get_param(0)
        self_inv.set_param(0, inv_theta)

        cos_theta = math.cos(inv_theta)
        sin_theta = math.sin(inv_theta)

        M = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        t = self.get_params()[1:]

        tinv = M.dot(-t)

        self_inv.set_param(1, tinv[0])
        self_inv.set_param(2, tinv[1])
        
        return self_inv

    def inverse_to_forward_matrix(self):
        theta = self.get_param(0)
        inv_theta = -theta

        cos_theta_inv = math.cos(inv_theta)
        sin_theta_inv = math.sin(inv_theta)

        Mprime = np.array([[-sin_theta_inv, -cos_theta_inv], [cos_theta_inv, -sin_theta_inv]])
        t = self.get_params()[1:]

        trot = Mprime.dot(t)
        
        D0 = [-1.0, trot[0], trot[1]]
        D1 = [0.0, -cos_theta_inv, -sin_theta_inv]
        D2 = [0.0, sin_theta_inv, -cos_theta_inv]

        return np.array([D0, D1, D2])

    def itk_transform_string_rec(self, index):
        s = '#Transform %d\n' % index        
        s = s + 'Transform: Rigid2DTransformBase_double_%d_%d\n' % (self.get_dim(), self.get_dim())
        s = s + 'Parameters:'
        for i in range(self.get_param_count()):
            s = s + (' %f' % self.get_param(i))
        s = s + '\n'
        s = s + 'FixedParameters:'
        s = s + '\n'

        return s 

if __name__ == '__main__':
    t = Rigid2DTransform()
    
    theta = -math.pi / 2.5
    tx = 7.0
    ty = -9.0

    #theta = math.pi / 4
    #tx = 1.5
    #ty = -2.5
    
    t.set_param(0, theta)
    t.set_param(1, tx)
    t.set_param(2, ty)
    #t.set_param(0, math.pi / 4)
    #t.set_param(1, 1.5)
    #t.set_param(2, -2.5)

    print(t.get_params())
    pnts = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -2.0], [2.0, -1.0], [0.1, 2.5]])

    print(pnts)
    tpnts = t.transform(pnts)

    print(tpnts)

    tinv = t.invert()

    print(tinv.get_params())

    tinv_pnts = tinv.transform(tpnts)

    print(tinv_pnts)

    spatial_grad = np.array([[1., 2.], [-0.5, 3.0], [0.5, 0.5], [-1.0, -1.0], [0.0, 0.0]])
    grad = tinv.grad(tpnts, spatial_grad, False)

    print("grad: " + str(grad))
    inv_grad = t.grad_inverse_to_forward(grad)

    print(inv_grad)

    inv_grad_num = t.grad_inverse_to_forward_num(grad)

    print(inv_grad_num)
    
