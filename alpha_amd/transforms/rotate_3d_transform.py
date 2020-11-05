
# Rotate3d Transform
# using Quaternions
# Author: Johan Ofverstedt

import math
import numpy as np
from transforms.transform_base import TransformBase

class Rotate3DTransform(TransformBase):
    def __init__(self):
        TransformBase.__init__(self, 2, 4)

    def copy_child(self):
        return Rotate3DTransform()
    
    def to_matrix(self, q):
        s = 1.0 / (np.square(q).sum())
        R_00 = 1.0 - 2.0*s*(np.square(q[2]) + np.square(q[3]))
        R_01 = 2.0*s*(q[1]*q[2]-q[3]*q[0])
        R_02 = 2.0*s*(q[1]*q[3]+q[2]*q[0])

        R_10 = 2.0*s*(q[1]*q[2]+q[3]*q[0])
        R_11 = 1.0 - 2.0*s*(np.square(q[1]) + np.square(q[3]))
        R_12 = 2.0*s*(q[2]*q[3]+q[1]*q[0])
        
        R_20 = 2.0*s*(q[1]*q[3]-q[2]*q[0])
        R_21 = 2.0*s*(q[2]*q[3]+q[1]*q[0])
        R_22 = 1.0 - 2.0*s*(np.square(q[1]) + np.square(q[2]))
        
        R = np.array([[R_00, R_01, R_02], [R_10, R_11, R_12], [R_20, R_21, R_22]])

        return R

    def transform(self, pnts):
        q = self.get_params()
        M = np.transpose(self.to_matrix(q))
        
        return pnts.dot(M)

    def grad(self, pnts, gradients):
        res = np.zeros((4,))
        q = self.get_params()

        Mprime = np.zeros((4, 3))
        Mprime[0, :] = q[1:]

        Mprime[1, 0] = -q[2]
        Mprime[1, 1] = q[1]
        Mprime[1, 2] = q[0]

        Mprime[2, 0] = -q[3]
        Mprime[2, 1] = -q[0]
        Mprime[2, 2] = q[1]

        Mprime[3, 0] = -q[0]
        Mprime[3, 1] = q[3]
        Mprime[3, 2] = -q[2]

        Mprime = np.transpose(Mprime * 2.0)

        jac_0 = pnts.dot(Mprime)
        jac_1 = np.array([-jac_0[1, :], jac_0[0, :], jac_0[3, :], -jac_0[2, :]])
        jac_2 = np.array([-jac_0[2, :], -jac_0[3, :], jac_0[0, :], jac_0[1, :]])

        Mprime = 2.0 * np.array([
            [q[1], q[2], q[3]], [-q[2], q[1], q[0]], [-q[3], - q[0] + q[1]] 
        ])
        return res
    #def grad(self, pnts, gradients, output_gradients):
    #    res = np.zeros((1,))
    #    theta = self.get_param(0)
    #    cos_theta = math.cos(theta)
    #    sin_theta = math.sin(theta)
    #    Mprime = np.array([[-sin_theta, cos_theta], [-cos_theta, -sin_theta]])
    #    Mprimepnts = pnts.dot(Mprime)
    #    res[:] = (Mprimepnts * gradients).sum()

    #    if output_gradients == True:
    #        M = np.transpose(np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]]))
        
    #        return res, gradients.dot(M)
    #    else:
    #        return res

    def invert(self):
        self_inv = self.copy()
        self_inv.set_params(-self_inv.get_params())
        self_inv.set_param(0, -self_inv.get_param(0))

        return self_inv

    def inverse_to_forward_matrix(self):
        inv_mat = np.eye((4,4))
        inv_mat[0, 0] = 1.0
        return inv_mat
        
    #def grad_inverse_to_forward(self, inv_grad):
    #    res = np.zeros((1,))
    #    res[:] = -inv_grad
    #    return res

if __name__ == '__main__':
    pass