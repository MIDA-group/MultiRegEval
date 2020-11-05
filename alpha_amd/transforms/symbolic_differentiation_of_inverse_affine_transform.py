
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
# Symbolic affine differentiation
#

import sympy
import numpy as np
from sympy.abc import X

def affine_inv_diff(dim):
    A = sympy.Matrix(sympy.symarray('a', (dim+1, dim+1)))
    for i in range(dim):
        A[dim,i] = 0
    A[dim,dim] = 1

    det = sympy.simplify(A.det())
    det_symbol = sympy.Symbol('det')
    
    B = sympy.simplify(A.inv())

    M = [sympy.simplify(B.diff(A[i, j])).subs(det, det_symbol) for i in range(dim) for j in range(dim)]
    T = [sympy.simplify(B.diff(A[i, dim])).subs(det, det_symbol) for i in range(dim)]

    # Generate code

    print("def diff_inv_to_forward_transform_%dd(param):" % dim)
    
    print("    ")
    print("    # Generate local variables for each parameter")
    for i in range(dim):
        for j in range(dim):
            index = i * dim + j
            print("    a_%d_%d = param[%d]" % (i,j,index))
    for i in range(dim):
        index = dim*dim + i
        print("    a_%d_%d = param[%d]" % (i,dim,index))

    print("    ")
    print("    # Compute determinant")
    print("    det = " + str(det))

    s = ""
    for k in range(dim * dim):
        if k == 0:
            s = s + "["
        else:
            s = s + ", ["
        for i in range(dim):
            for j in range(dim):
                x = M[k][i, j]
                if i == 0 and j == 0:
                    s = s + str(x)
                else:
                    s = s + ", " + str(x)
        for i in range(dim):
            x = M[k][i, dim]
            s = s + ", " + str(x)
        s = s + "]"
    for k in range(dim):
        s = s + ", ["
        for i in range(dim):
            for j in range(dim):
                x = T[k][i, j]
                if i == 0 and j == 0:
                    s = s + str(x)
                else:
                    s = s + ", " + str(x)
        for i in range(dim):
            x = T[k][i, dim]
            s = s + ", " + str(x)
        s = s + "]"
    
    print("    ")
    print("    # Compute and return final matrix")
    print("    return np.array([" + s + "])")

# Example use, for 3d affine transforms
if __name__ == '__main__':
    affine_inv_diff(3)
