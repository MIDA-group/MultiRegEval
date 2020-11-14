# -*- coding: utf-8 -*-

import os, cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import skimage.io as skio
import skimage.transform as skt
from skimage.measure import ransac
from skimage import exposure
from skimage.util import img_as_ubyte
# %%
# =============================================================================
# DATA_ROOT = './Datasets/Stefan/'
# img1 = cv2.imread(DATA_ROOT+'HE/146558_2_HE.tif')
# #img_MPM = cv2.imread(DATA_ROOT+'MPM/146558_2_MPM.tif',0)
# img2 = cv2.imread(DATA_ROOT+'SHG/146558_2_SHG.tif', 0)
# img2 = np.rot90(img2, k=3)
# img2 = cv2.normalize(src=img2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# 
# =============================================================================
# %% Python replacement for cv2.drawMatches()
def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
#    if len(img1.shape) == 3:
#        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
#    elif len(img1.shape) == 2:
#        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    if len(img1.shape) == 2:
        img1 = np.repeat(img1.reshape(img1.shape[0], img1.shape[1], 1), 3, axis=-1)
    if len(img2.shape) == 2:
        img2 = np.repeat(img2.reshape(img2.shape[0], img2.shape[1], 1), 3, axis=-1)
    
    new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3)
    
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 5
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3)# if len(img1.shape) == 3 else np.random.randint(0,256)
            c = (int(c[0]), int(c[1]), int(c[2]))
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    return new_img

# %%
def register_sift(img1, img2, ttype='rigid', equalhist=False):
#    img1 = cv2.imread('./Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/B/test/1B_A4_T.tif', 0)
#    img2 = cv2.imread('./Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/B/test/1B_A4_R.tif', 0)
#img1 = cv2.imread('./Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/p2p_A/1B_A1_T.png', 0)
#img2 = cv2.imread('./Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/A/test/1B_A1_R.tif', 0)
    ''' Register img1 to img2
    '''
    img1_ori = img1
#    if equalhist:
#        if img1.ndim == 2:
#            img1 = cv2.equalizeHist(img1)
##            img1 = exposure.equalize_adapthist(img1, clip_limit=0.03)
#        if img2.ndim == 2:
#            img2 = cv2.equalizeHist(img2)
##            img2 = exposure.equalize_adapthist(img2, clip_limit=0.03)
    sift = cv2.xfeatures2d.SIFT_create(500)
    
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    #keypoints_MPM, descriptors_MPM = sift.detectAndCompute(img_MPM, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
# =============================================================================
#    # KNN Matcher with default params
#    bf = cv2.BFMatcher()
#    matches = bf.knnMatch(descriptors1,descriptors2,k=2)
#    # Apply ratio test
#    good = []
#    for m,n in matches:
#        if m.distance < 0.75*n.distance:
#            good.append([m])
#    img3 = cv2.drawMatchesKnn(img1,keypoints1,img2,keypoints2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    plt.imshow(img3)
# =============================================================================
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)
    
#    img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:15], None, flags=2)
    img3 = draw_matches(img1, keypoints1, img2, keypoints2, matches[:15], color=None)
#    plt.imshow(img3)
# =============================================================================
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    if ttype == 'rigid':
        
    # if RANSAC
        tform, inliers = ransac((points2, points1), skt.EuclideanTransform, min_samples=3,
                                       residual_threshold=2, max_trials=100)
        outliers = inliers == False
#    # else
#        tform = skt.estimate_transform('euclidean', points2, points1) # the returned tform is for skt.wrap ONLY!
        
        img1Reg = skt.warp(img1_ori, tform)
#        tform = skt.estimate_transform('euclidean', points1, points2)
#        img1Reg = skt.warp(img1_ori, tform.inverse)
    
    elif ttype == 'affine':
        # Find homography
    #    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        # affine
        h, mask = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)
        
        # Use homography
        height, width = img2.shape[:2]
    #    img1Reg = cv2.warpPerspective(img1, h, (width, height))
        # affine
        img1Reg = cv2.warpAffine(img1_ori, h, (width, height))
    #    plt.imshow(img1Reg)
    
    return img3, img1Reg, tform

def register_sift_batch_stefan(data_root, target_dir):
#    target_dir='./outputs/SIFT/Stefan/'
    dirA = data_root + 'HE/'
    dirB = data_root + 'SHG/'
    
    dir_matches = target_dir + 'matches'
    dir_results = target_dir + 'results'
    if not os.path.exists(dir_matches):
        os.makedirs(dir_matches)
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    
    suffixA = '_' + os.listdir(dirA)[0].split('_')[-1]
    nameAs = set([name[:-len(suffixA)] for name in os.listdir(dirA)])
    suffixB = '_' + os.listdir(dirB)[0].split('_')[-1]
    nameBs = set([name[:-len(suffixB)] for name in os.listdir(dirB)])
    f_names = nameAs & nameBs

    for f_name in tqdm(f_names):
        f_nameA = f_name + suffixA
        f_nameB = f_name + suffixB
        imgA = cv2.imread(dirA + f_nameA)
        imgB = cv2.imread(dirB + f_nameB, 0)
        imgB = np.rot90(imgB, k=3)
        imgB = cv2.normalize(src=imgB, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        try:
            img_match, imgA2, tform = register_sift(imgA, imgB)
        except:
            heightA, widthA = imgA.shape[:2]
            widthB = imgB.shape[1]
            img_match = np.zeros((heightA, widthA+widthB))
            imgA2 = np.zeros((heightA, widthA))
            
        
        skio.imsave(f'{dir_matches}/{f_name}.tif', img_match)
        skio.imsave(f'{dir_results}/{f_name}_HEreg.tif', imgA2)
    return

def register_sift_batch_eliceiri(data_root, target_dir, mode='a2b'):
#    data_root='./Datasets/Eliceiri_test/processed/'
#    target_dir = './outputs/SIFT/Eliceiri/'
#    assert mode in ['a2b', 'b2a'], "mode must be in ['a2b', 'b2a']"
#    dirA = data_root + 'A/test/'
#    dirB = data_root + 'B/test/'
    
    dir_matches = target_dir + 'matches'
    dir_results = target_dir + 'results'
    if not os.path.exists(dir_matches):
        os.makedirs(dir_matches)
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    
#    suffixA = '_' + os.listdir(dirA)[0].split('_')[-1]
#    nameAs = set([name[:-len(suffixA)] for name in os.listdir(dirA)])
#    suffixB = '_' + os.listdir(dirB)[0].split('_')[-1]
#    nameBs = set([name[:-len(suffixB)] for name in os.listdir(dirB)])
#    f_names = nameAs & nameBs
#
#    if mode=='a2b':
#        for f_name in tqdm(f_names):
#            f_nameA = f_name + '_T.tif'
#            f_nameB = f_name + '_R.tif'
#            imgA = cv2.imread(dirA + f_nameA, 0)
#            imgB = cv2.imread(dirB + f_nameB)
#            
#            try:
#                img_match, imgA2, tform = register_sift(imgA, imgB)
#            except:
#                heightA, widthA = imgA.shape
#                widthB = imgB.shape[1]
#                img_match = np.zeros((heightA, widthA+widthB))
#                imgA2 = np.zeros((heightA, widthA))
#            
#            skio.imsave(f'{dir_matches}/{f_name}.tif', img_match)
#            skio.imsave(f'{dir_results}/{f_name}_Areg.tif', imgA2)
#    elif mode=='b2a':
#        for f_name in tqdm(f_names):
#            f_nameA = f_name + '_R.tif'
#            f_nameB = f_name + '_T.tif'
#            imgA = cv2.imread(dirA + f_nameA, 0)
#            imgB = cv2.imread(dirB + f_nameB)
#            
#            try:
#                img_match, imgB2, tform = register_sift(imgB, imgA)
#            except:
#                heightB, widthB = imgB.shape[:2]
#                widthA = imgA.shape[1]
#                img_match = np.zeros((heightB, widthA+widthB))
#                imgB2 = np.zeros((heightB, widthB))
#            
#            skio.imsave(f'{dir_matches}/{f_name}.tif', img_match)
#            skio.imsave(f'{dir_results}/{f_name}_Breg.tif', imgB2)
#    
    assert mode in ['a2b', 'b2a', 'a2a', 'b2b'], "mode must be in ['a2b', 'b2a', 'a2a', 'b2b']"
    if mode=='a2b':
        dir_src = data_root + 'A/test/'
        dir_tar = data_root + 'B/test/'
    elif mode=='b2a':
        dir_src = data_root + 'B/test/'
        dir_tar = data_root + 'A/test/'
    elif mode=='a2a':
        dir_src = data_root + 'A/test/'
        dir_tar = data_root + 'A/test/'
    elif mode=='b2b':
        dir_src = data_root + 'B/test/'
        dir_tar = data_root + 'B/test/'
        
    suffix_src = '_' + os.listdir(dir_src)[0].split('_')[-1]
    name_srcs = set([name[:-len(suffix_src)] for name in os.listdir(dir_src)])
    suffix_tar = '_' + os.listdir(dir_tar)[0].split('_')[-1]
    name_tars = set([name[:-len(suffix_tar)] for name in os.listdir(dir_tar)])
    f_names = name_srcs & name_tars
    f_names = list(f_names)
    f_names.sort()
    
    for f_name in tqdm(f_names):
        img_src = cv2.imread(dir_src + f'{f_name}_T.tif')
        img_tar = cv2.imread(dir_tar + f'{f_name}_R.tif')
        try:
            img_match, img_rec, tform = register_sift(img_src, img_tar)
        except:
            height_src, width_src = img_src.shape
            width_tar = img_tar.shape[1]
            img_match = np.zeros((height_src, width_src+width_tar))
            img_rec = np.zeros((height_src, width_src))
        
        skio.imsave(f'{dir_matches}/{f_name}.tif', img_match)
        skio.imsave(f'{dir_results}/{f_name}_rec.tif', img_rec)

    
    
    return

# %%
if __name__ == '__main__':

    register_sift_batch_stefan(data_root='./Datasets/Stefan/', target_dir='./outputs/SIFT/Stefan/')
    
    register_sift_batch_eliceiri(
            data_root='./Datasets/Eliceiri_test/processed/', 
            target_dir='./outputs/SIFT/Eliceiri_a2b/',
            mode='a2b')
    register_sift_batch_eliceiri(
            data_root='./Datasets/Eliceiri_test/processed/', 
            target_dir='./outputs/SIFT/Eliceiri_b2a/',
            mode='b2a')
    register_sift_batch_eliceiri(
            data_root='./Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/', 
            target_dir='./outputs/SIFT/Eliceiri_rot15-20_a2b/',
            mode='a2b')
    register_sift_batch_eliceiri(
            data_root='./Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/', 
            target_dir='./outputs/SIFT/Eliceiri_rot15-20_b2a/',
            mode='b2a')

# %% Make sample images
    def make_sample(data_root, f_name, gan_name='', data_root_fake=None, preprocess='nopre', mode='b2a'):
    #    data_root='./Datasets/Eliceiri_patches/patch_tlevel2/'
    #    f_name='1B_A1'
    #    gan_name='p2p_A'
    #    data_root_fake='./Datasets/Eliceiri_patches_fake'
    #    preprocess='nopre'
    #    mode='b2a'
    
    
        dir_A = data_root + 'A/test/'
        dir_B = data_root + 'B/test/'
        
        if gan_name != '':
            assert data_root_fake, "data_root_fake must not be None when given gan_name."
            assert gan_name in ['cyc_A', 'cyc_B', 'p2p_A', 'p2p_B', 'drit_A', 'drit_B'], (
                    "gan_name must be in 'cyc_A', 'cyc_B', 'p2p_A', 'p2p_B', 'drit_A', 'drit_B'")
            if '_A' in gan_name:
                dir_B = f'{data_root_fake}/{os.path.split(data_root[:-1])[-1]}/{gan_name}/'
            elif '_B' in gan_name:
                dir_A = f'{data_root_fake}/{os.path.split(data_root[:-1])[-1]}/{gan_name}/'
        
        assert mode in ['a2b', 'b2a', 'a2a', 'b2b'], "mode must be in ['a2b', 'b2a', 'a2a', 'b2b']"
        if mode=='a2b':
            dir_src = dir_A
            dir_tar = dir_B
        elif mode=='b2a':
            dir_src = dir_B
            dir_tar = dir_A
        elif mode=='a2a':
            dir_src = dir_A
            dir_tar = dir_A
        elif mode=='b2b':
            dir_src = dir_B
            dir_tar = dir_B
            
        assert preprocess in ['', 'nopre', 'PCA', 'hiseq'], "preprocess must be in ['', 'nopre', 'PCA', 'hiseq']"
        
    
        suffix_src = '_' + os.listdir(dir_src)[0].split('_')[-1]
        suffix_tar = '_' + os.listdir(dir_tar)[0].split('_')[-1]
        
        img_src = cv2.imread(dir_src + f"{f_name}_T.{suffix_src.split('.')[-1]}")
        img_tar = cv2.imread(dir_tar + f"{f_name}_R.{suffix_tar.split('.')[-1]}")
        
#        # for debugging
#        img_src = cv2.imread('./Datasets/Eliceiri_patches_fake/patch_tlevel2/p2p_A/1B_A1_T.png')
#        img_tar = cv2.imread('./Datasets/Eliceiri_patches/patch_tlevel2/A/test/1B_A1_R.tif')
        
        try:
            img_match, img_rec, tform = register_sift(img_src, img_tar)
        except:
            height_src, width_src = img_src.shape[:2]
            width_tar = img_tar.shape[1]
            img_match = np.zeros((height_src, width_src+width_tar))
            img_rec = np.zeros((height_src, width_src))
    
    
        dir_matches = f'{os.path.dirname(data_root[:-1])}/example/sift'
        if not os.path.exists(dir_matches):
            os.makedirs(dir_matches)
        
    #    skio.imsave(f'{dir_matches}/sift_{gan_name}_{mode}_{preprocess}.png', img_match)
        cv2.imwrite(f'{dir_matches}/sift_{gan_name}_{mode}_{preprocess}.png', img_match)
    #    skio.imsave(f'{dir_results}/{f_name}_rec.png', img_rec)
        return
        
    # %%
    for gan in tqdm(['p2p_A', 'p2p_B', 'cyc_A', 'cyc_B', 'drit_A', 'drit_B']):
        make_sample(
                data_root='./Datasets/Eliceiri_patches/patch_tlevel2/', 
                f_name='1B_A1', 
                gan_name=gan,
                data_root_fake='./Datasets/Eliceiri_patches_fake', 
                preprocess='nopre', 
                mode='b2a')
    for mode in tqdm(['a2a', 'b2b']):
        make_sample(
                data_root='./Datasets/Eliceiri_patches/patch_tlevel2/', 
                f_name='1B_A1', 
                gan_name='',
                data_root_fake='./Datasets/Eliceiri_patches_fake', 
                preprocess='nopre', 
                mode=mode)


# %% testing script
# =============================================================================
# w=834
# coords_ref = np.array(([0,0], [0,w], [w,w], [w,0]))
# centre_patch = np.array((w, w)) / 2. - 0.5
# 
# rot_radian = -0.325329283
# #rotation_trans = skt.SimilarityTransform(rotation=rot_radian).params[:2, :2]
# tx_trans, ty_trans = -61.63485592, -75.82964144
# #translation = skt.SimilarityTransform(translation=(tx_trans, ty_trans)).translation
# tform_patch = tform_centred(radian=rot_radian, translation=(tx_trans, ty_trans), center=centre_patch)
# coords_trans = skt.matrix_transform(coords_ref, tform_patch.params)
# # equivalent: 
# # coords_trans = np.dot(rotation, coords_ref.T).T + tform_patch.translation
# # coords_trans = np.dot(rotation, (coords_ref + translation - centre_patch).T).T + centre_patch.T
# 
# tform_1to2 = skt.estimate_transform('euclidean', points2, points1)
# tx, ty = tform_1to2.translation
# rotation = tform_1to2.params[:2, :2]
# 
# # if rotate first, translate last, then the required translation T is:
# translation_last = tform_1to2.translation + np.dot(rotation, centre_patch.T) - centre_patch
# tform_patch_rec = tform_centred_rec(radian=tform_1to2.rotation, translation=translation_last, center=centre_patch)
# # tform_patch_rec == tform_1to2, useless
# 
# 
# img1to2 = skt.warp(img1_ori, tform_1to2)
# ## test warp function
# #t1 = skt.SimilarityTransform(translation=tform_1to2.translation)
# #t2 = skt.SimilarityTransform(rotation=tform_1to2.rotation)
# #img11 = skt.warp(img1_ori, t1)
# #img12 = skt.warp(img11, t2)
# 
# coords_rec1 = skt.matrix_transform(coords_trans, tform_1to2.params)
# trans_forcoords = np.dot(rotation, tform_1to2.translation.T)
# tform_patch_rec = skt.SimilarityTransform(rotation=tform_1to2.rotation, translation=trans_forcoords)
# coords_rec2 = skt.matrix_transform(coords_trans, tform_patch_rec.params)
# 
# tform_2to1 = skt.estimate_transform('euclidean', points1, points2)
# img2to1 = skt.warp(img1_ori, tform_2to1.inverse)
# 
# 
# # %%
# img1 = skio.imread('./pytorch-CycleGAN-and-pix2pix/results/eliceiri_p2p_rotation_b2a/1B_A1_R_fake_B.png', as_grey=True)
# img1 = img_as_ubyte(img1)
# img2 = skio.imread('./Datasets/Eliceiri_patches/example/A/test/1B_A1_R.tif')
# img_match, imgA2 = register_sift(img1, img2, equalhist=False)
# skio.imshow(img_match)
# skio.imsave(f'./Datasets/Eliceiri_patches/example/sift_b2a_p2p.tif', img_match)
# # %%
# MIN_MATCH_COUNT = 1
# 
# sift = cv2.xfeatures2d.SIFT_create()
# 
# keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
# #keypoints_MPM, descriptors_MPM = sift.detectAndCompute(img_MPM, None)
# keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
# 
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(descriptors1, descriptors2, k=2)
# 
# # store all the good matches as per Lowe's ratio test.
# good = [m for m, n in matches if m.distance < 0.7*n.distance]
# 
# if len(good) > MIN_MATCH_COUNT:
#     src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
#     dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#     h, w = img1.shape[:2]
# #    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# #    dst = cv2.perspectiveTransform(pts,M)
# #    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None
#     
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,good,None,**draw_params)
# 
# 
# =============================================================================
