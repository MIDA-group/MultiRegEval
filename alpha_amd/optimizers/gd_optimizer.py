
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
# Gradient descent optimizer for image registration.
#

import numpy as np

def _default_report_callback(opt):
    iteration = opt.get_iteration()
    value = opt.get_value()
    grad = opt.get_grad()
    param = opt.get_transform().get_params()
    print("#%d. --- Value: " % (iteration) + str(value) + ", Grad: " + str(grad) + ", Param: " + str(param))
def _no_report_callback(opt):
    pass

class GradientDescentOptimizer:
    def __init__(self, measure, transform):
        self.measure = measure
        self.transform = transform
        self.step_length = 1.0
        self.end_step_length = None
        self.gradient_magnitude_threshold = 0.0001
        self.param_scaling = None#np.ones((transform.get_param_count(),))
        self.last_value = np.nan
        self.last_grad = np.zeros((transform.get_param_count(),))
        self.iteration = 0
        self.termination_reason = ""
        self.report_freq = 1
        self.report_func = _no_report_callback
        self.value_history = []
        self.mem = np.zeros_like(transform.get_params())

    def set_step_length(self, step_length, end_step_length = None):
        self.step_length = step_length
        self.end_step_length = end_step_length
    
    def set_gradient_magnitude_threshold(self, gmt):
        self.gradient_magnitude_threshold = gmt
    
    def set_scaling(self, index, scale):
        if self.param_scaling is None:
            self.param_scaling = np.ones((self.transform.get_param_count(),))
        self.param_scaling[index] = scale
    
    def set_scalings(self, scalings):
        if self.param_scaling is None:
            self.param_scaling = scalings
        else:
            self.param_scaling[:] = scalings[:]
    
    def get_iteration(self):
        return self.iteration
    
    def get_termination_reason(self):
        return self.termination_reason
    
    def set_report_freq(self, freq):
        if freq < 0:
            freq = 0
        self.report_freq = freq
    
    def set_report_callback(self, func, additive = True):
        if additive == True:
            if func is not None:
                old_func = self.report_func
                if old_func is not None:
                    def glue(opt):
                        old_func(opt)
                        func(opt)
                    self.report_func = glue
                else:
                    self.report_func = func
        else:
            self.report_func = func
    
    def get_value(self):
        return self.last_value

    def get_value_history(self):
        return self.value_history

    def get_grad(self):
        return self.last_grad

    def get_transform(self):
        return self.transform
    
    def step(self, step_length, report):
        # check for list of measures or single measure
        if isinstance(self.measure, list):
            for i in range(len(self.measure)):
                (value_i, grad_i) = self.measure[i].value_and_derivatives(self.transform)
                if i == 0:
                    value = value_i
                    grad = grad_i
                else:
                    value = value + value_i
                    grad = grad + grad_i
        else:
            (value, grad) = self.measure.value_and_derivatives(self.transform)
        self.iteration = self.iteration + 1

        par = self.transform.get_params()
        parabs = np.abs(par)
        parsign = np.sign(par)
        # L1 Reg
        l1_factor = 0.0
        l1_grad = np.clip(l1_factor*parsign/(self.param_scaling+1e-15), -parabs*step_length, parabs*step_length)
        # L2 Regularization
        l2_factor = 0.0
        l2_grad = np.clip(l2_factor*par/(self.param_scaling+1e-15), -parabs*step_length, parabs*step_length)

        grad = np.clip(grad, -1.0, 1.0)

        alpha = 0.9
        grad = self.mem * alpha + grad * (1.0-alpha)
        self.mem[:] = grad[:]

        # Update parameters
        if self.param_scaling is not None:
            self.transform.step_params(-l1_grad-l2_grad -grad * self.param_scaling, step_length)
        else:
            self.transform.step_params(-l1_grad-l2_grad -grad, step_length)
        
        res = 0
        if self.gradient_magnitude_threshold > 0.0:
            grad_mag = np.sqrt(np.square(grad).sum())

            if grad_mag < self.gradient_magnitude_threshold:
                self.termination_reason = 'Gradient magnitude (%f) below threshold (%f) after (%d) iterations.' % (grad_mag, self.gradient_magnitude_threshold, self.iteration)
                res = 1

        if self.iteration > 1:
            alpha = 0.5
            self.last_value = self.last_value * alpha + value * (1.0-alpha)
        else:
            self.last_value = value
        #self.last_value = value
        self.last_grad[:] = grad[:]

        self.value_history.append(value)

        if report == True or (self.report_freq > 0 and res == 1):
            #if self.report_func is None:
            #    print("#%d. --- Value: " % (self.iteration) + str(value) + ", Grad: " + str(grad) + ", Param: " + str(self.transform.get_params()) + ", Step-length: " + str(step_length))
            #else:
            if self.report_func is not None:
                self.report_func(self)

        return res
    
    def optimize(self, iterations):
        step_length = self.step_length
        for i in range(iterations):
            if self.end_step_length is not None:
                cur_step_length = step_length + (float(i)/float(iterations)) * (self.end_step_length - step_length)
            else:
                cur_step_length = step_length

            if self.report_freq > 0 and (i % self.report_freq == 0 or i == iterations-1):
                report = True
            else:
                report = False
            if not self.step(cur_step_length, report) == 0:
                return iterations - i
        self.termination_reason = 'Maximum iteration count reached (%d).' % iterations
        return 0
