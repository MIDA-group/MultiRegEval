
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
# Symmetric Average Minimal Distances (AMD) Distance implemented as a class.
#

import numpy as np

class SymmetricAMDDistance:
    def __init__(self, symmetric_measure = True, squared_measure = False):
        self.ref_image_source = None
        self.flo_image_source = None
        self.ref_image_target = None
        self.flo_image_target = None

        self.sampling_fraction = 1.0
        self.sampling_count = np.nan
        self.symmetric_measure = symmetric_measure
        self.squared_measure = squared_measure

    def set_ref_image_source(self, image):
        self.ref_image_source = image

    def set_flo_image_source(self, image):
        self.flo_image_source = image

    def set_ref_image_target(self, image):
        self.ref_image_target = image

    def set_flo_image_target(self, image):
        self.flo_image_target = image

    def set_sampling_fraction(self, sampling_fraction):
        self.sampling_fraction = sampling_fraction

    def initialize(self):
        self.sampling_count_forward = self.ref_image_source.get_sampling_fraction_count(self.sampling_fraction)
        self.sampling_count_inverse = self.flo_image_source.get_sampling_fraction_count(self.sampling_fraction)

    def asymmetric_value_and_derivatives(self, transform, source, target, target_cp, sampling_count):
        w_acc = 0.0
        value_acc = 0.0
        grad_acc = np.zeros(transform.get_param_count())

        sampled_points = source.random_sample(sampling_count)
        
        for q in range(len(sampled_points)):
            sampled_points_q = sampled_points[q]

            if sampled_points_q.size == 0:
                continue

            w_q = sampled_points_q[:, -1:]
            pnts_q = sampled_points_q[:, 0:-1]
            tf_pnts = transform.transform(pnts_q) + target_cp

            (eval_pnts, eval_w) = target.compute_spatial_grad_and_value(tf_pnts, w_q, q)

            values_q = eval_pnts[:, -1:]
            grads_q = eval_pnts[:, :-1]
            if self.squared_measure:
                grads_q = 2.0 * values_q * grads_q
                values_q = np.square(values_q)

            value_acc = value_acc + np.sum(values_q)
            w_acc = w_acc + np.sum(eval_w)
            grad_q_2 = transform.grad(pnts_q, grads_q, False)
            grad_acc[:] = grad_acc[:] + grad_q_2
            #print("grad_acc: " + str(grad_acc))

        if w_acc < 0.000001:
            w_acc = 1.0
        #print("w_acc: " + str(w_acc))
        #print("grad_acc: " + str(grad_acc))
        w_rec = 1.0 / w_acc
        value_acc = value_acc * w_rec
        grad_acc[:] = grad_acc[:] * w_rec
        #print("grad_acc: " + str(grad_acc))
        return (value_acc, grad_acc)       

    def value_and_derivatives(self, transform):
        ref_cp = self.ref_image_source.get_center_point()
        flo_cp = self.flo_image_source.get_center_point()

        (forward_value, forward_grad) = self.asymmetric_value_and_derivatives(transform, self.ref_image_source, self.flo_image_target, flo_cp, self.sampling_count_forward)
        if self.symmetric_measure:
            inv_transform = transform.invert()

            (inverse_value, inverse_grad) = self.asymmetric_value_and_derivatives(inv_transform, self.flo_image_source, self.ref_image_target, ref_cp, self.sampling_count_inverse)

            inverse_grad = transform.grad_inverse_to_forward(inverse_grad)

            value = 0.5 * (forward_value + inverse_value)
            grad = 0.5 * (forward_grad + inverse_grad)
        else:
            value = forward_value
            grad = forward_grad

        return (value, grad)
