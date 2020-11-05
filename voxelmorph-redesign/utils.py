# -*- coding: utf-8 -*-

import pandas as pd
import skimage.io as skio
import skimage.transform as skt
from tqdm import tqdm
from glob import glob
import os, random, math, cv2
import numpy as np

# %%
def tform_centred(radian, translation, center):
    # first translation, then rotation
    tform1 = skt.SimilarityTransform(translation=center)
    tform2 = skt.SimilarityTransform(rotation=radian)
    tform3 = skt.SimilarityTransform(translation=-center)
    tform4 = skt.SimilarityTransform(translation=translation)
    tform = tform4 + tform3 + tform2 + tform1
    return tform

def dist_coords(coords1, coords2):
    ''' Calculate the point-to-point distances between two coordinates, return a list.
    '''
    return [sum((coords1[i] - coords2[i]) ** 2) ** 0.5 for i in range(len(coords1))]

def unpad_sample(img, d):
    # crop the image size to multiple of divisor d    
    (wi, hi) = img.shape[:2]
    wo = wi // d * d
    ho = hi // d * d
    assert wo <= wi and ho <= hi
    wl = (wi - wo) // 2
    hl = (hi - ho) // 2
    return img[wl:wl+wo, hl:hl+ho]

def load_image(img_path):
    # load image in gretscale and unpad to multiple of 32
    img = cv2.imread(img_path, 0)
    img = unpad_sample(img, d=32)
    return img
