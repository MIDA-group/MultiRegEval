
import numpy as np
import scipy as sp
import itertools

def index_sets(subset_count, count):
    return list(itertools.combinations(range(count), subset_count))

def all_index_sets(count):
    lst = []
    for i in range(count+1):
        lst = lst + index_sets(count-i, count)
    return lst

def find(lst, x):
    return lst.index(x)

def complement_set_index(lst, ind):
    return len(lst) - ind - 1

def complement_set(lst, x):
    return lst[complement_set_index(lst, find(lst, x))]

def fidt(im, levels):
    if levels is None or levels == 0:
        return fidt_internal(im)
    else:
        im_q = [(np.floor((levels * im_i) + 0.5) / float(levels)) for im_i in im]
        fidt_q = fidt_internal(im_q)
        return fidt_q

def fidt_internal(im):
    dim = len(im)
    uni = all_index_sets(dim)
    unilen = len(uni)
    
    intersections = [None]*unilen
    unions = [None]*unilen

    # empty set
    intersections[unilen-1] = np.ones_like(im[0])
    unions[unilen-1] = np.zeros_like(im[0])

    inds = unilen - dim - 1
    
    # singleton set
    for i in range(dim):
        intersections[inds+i] = im[i]
        unions[inds+i] = im[i]

    # for sets of increasing cardinality
    for k in range(1, dim):
        sets = index_sets(k+1, dim)
        for i in range(len(sets)):
            set_i = sets[i]
            new_ind = set_i[-1]
            old_inds = set_i[0:-1]
            old_ind_index = find(uni, old_inds)
            new_index = find(uni, set_i)
            intersections[new_index] = np.minimum(intersections[old_ind_index], im[new_ind])
            unions[new_index] = np.maximum(unions[old_ind_index], im[new_ind])
    
    result = [None]*unilen

    for i in range(unilen):
        diff = intersections[i]-unions[complement_set_index(uni, i)]
        result[i] = np.clip(diff, a_min = 0.0, a_max = None)

    return result

#uni = all_index_sets(5)

#print(str(index_sets(2, 5)))
#print(str(all_index_sets(5)))
#print(str(find(all_index_sets(5), (1, 3, 4))))
#print(len(uni))
#print(uni)
#print(find(uni, (1, 3, 4)))
#print(uni[find(uni, (1, 3, 4))])
#print(complement_set(uni, (1, 2)))

def example():
    a1 = (np.arange(9)+1).reshape([3, 3]) / 9.0
    a2 = np.transpose(a1)

    ft = fidt([a1, a2])

    return ([a1, a2], ft)