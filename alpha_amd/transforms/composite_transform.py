
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
# Composite Transform
#

import math
import numpy as np
from transforms.transform_base import TransformBase

class CompositeTransform(TransformBase):
    def __init__(self, dim, transforms, active_flags = None):
        self.dim = dim

        if active_flags is None:
            active_flags = np.ones(len(transforms), dtype='bool')
        
        self.active_flags = active_flags

        self.transforms = []
        
        cnt = 0
        for i in range(len(transforms)):
            t = transforms[i]

            if active_flags[i] == True:
                cnt = cnt + t.get_param_count()

            self.transforms.append(t.copy())

        self.param_count = cnt

    def get_transforms(self):
        return self.transforms
    
    def get_dim(self):
        return self.dim

    def get_params(self):
        res = np.zeros((self.param_count,))
        ind = 0
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                res[ind:ind + cnt] = t.get_params()
                ind = ind + cnt
        return res
    
    def set_params(self, params):
        ind = 0
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                t.set_params(params[ind:ind+cnt])
                ind = ind + cnt

    def get_param(self, index):
        assert(index >= 0)
        assert(index < self.param_count)

        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                if index < cnt:
                    return t.get_param(index)
                else:
                    index = index - cnt
    
    def set_param(self, index, value):
        assert(index >= 0)
        assert(index < self.param_count)
        
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                if index < cnt:
                    t.set_param(index, value)
                    return
                else:
                    index = index - cnt

    def set_params_const(self, value):
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                t.set_params_conts(value)

    def step_param(self, index, step_length):
        self.set_param(index, self.get_param(index) + step_length)

    def step_params(self, grad, step_length):
        params = self.get_params()
        params = params + grad * step_length
        self.set_params(params)

    def get_param_count(self):
        return self.param_count

    def copy_child(self):
        return CompositeTransform(self.get_dim(), self.transforms, self.active_flags)

    def copy(self):
        return self.copy_child()

    def transform(self, pnts):
        self.input_pnts = []

        p = pnts
        for i, t in enumerate(self.transforms):
            self.input_pnts.append(p)
            p = t.transform(p)
        
        #self.input_pnts.append(p)
        return p

    def grad(self, pnts, gradients, output_gradients):
        res = np.zeros((self.param_count,))
        ind = self.param_count
        p = pnts
        gr = gradients
        tlen = len(self.transforms)
        for i, t in enumerate(reversed(self.transforms)):
            t_index = tlen - i - 1
            #print("Input points: " + str(self.input_pnts[i]))
            #print("Gr: " + str(gr))
            if output_gradients == True or i < tlen-1:
                g, gr = t.grad(self.input_pnts[t_index], gr, True)
            else:
                g = t.grad(self.input_pnts[t_index], gr, False)
            
            if self.active_flags[t_index] == True:
                cnt = t.get_param_count()
                res[ind-cnt:ind] = g
                ind = ind - cnt
        
        if output_gradients == True:
            return res, gr
        else:
            return res

    def invert(self):
        inv_transforms = []

        tlen = len(self.transforms)
        for i in range(tlen):
            inv_transforms.append(self.transforms[(tlen-1)-i].invert())

        return CompositeTransform(self.get_dim(), inv_transforms, np.flip(self.active_flags, 0))

    def inverse_to_forward_matrix(self):
        pcnt = self.get_param_count()
        res = np.zeros((pcnt, pcnt))
        find = 0
        rind = pcnt
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                rind = rind - cnt
                mat = t.inverse_to_forward_matrix()
                res[find:find+cnt, rind:rind+cnt] = mat
                find = find + cnt
        return res
    '''def inverse_to_forward_matrix(self):
        pcnt = self.get_param_count()
        res = np.zeros((pcnt, pcnt))
        ind = 0
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                mat = t.inverse_to_forward_matrix()
                res[ind:ind+cnt, ind:ind+cnt] = mat
                ind = ind + cnt
        return res
    '''

    def itk_transform_string_rec(self, index):
        s = '#Transform %d\n' % index        
        s = s + 'Transform: CompositeTransform_double_%d_%d\n' % (self.get_dim(), self.get_dim())
        for i in range(len(self.transforms)):
            index = index + 1
            s = s + self.transforms[i].itk_transform_string_rec(index)

        return s
        
    #def grad_inverse_to_forward(self, inv_grad):
    #    pcnt = self.get_param_count()
    #    res = np.zeros((pcnt,))
    #    ind = 0
    #    rev_ind = pcnt
    #    for t in self.transforms:
    #        cnt = t.get_param_count()
    #        inv_grad_t = inv_grad[rev_ind-cnt:rev_ind]
    #        res[ind:ind+cnt] = t.grad_inverse_to_forward(inv_grad_t)
    #        ind = ind + cnt
    #        rev_ind = rev_ind - cnt
    #    return res
