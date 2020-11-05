
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
# Utility functions
#

import numpy as np

from transforms.transform_base import TransformBase
from transforms.translation_transform import TranslationTransform
from transforms.composite_transform import CompositeTransform

def image_center_point(image, spacing = None):
    shape = image.shape
    if spacing is None:
        return (np.array(shape)-1) * 0.5
    else:
        return ((np.array(shape)-1) * spacing) * 0.5

def image_diagonal(image, spacing = None):
    shp = np.array(image.shape)-1
    if spacing is not None:
        shp = shp * spacing
    return np.sqrt(np.sum(np.square(shp)))

def make_centered_transform(t, cp1, cp2):
    dim = t.get_dim()
    t1 = TranslationTransform(dim)
    t2 = TranslationTransform(dim)
    t1.set_params(-cp1)
    t2.set_params(cp2)
    return CompositeTransform(dim, [t1, t, t2], [False, True, False])

def make_image_centered_transform(t, image1, image2, image1_spacing = None, image2_spacing = None):
    dim = image1.ndim
    t1 = TranslationTransform(dim)
    t2 = TranslationTransform(dim)
    t1.set_params(-image_center_point(image1, image1_spacing))
    t2.set_params(image_center_point(image2, image2_spacing))
    return CompositeTransform(dim, [t1, t, t2], [False, True, False])
