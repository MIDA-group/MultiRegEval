import os
import sys
import random
import cv2
import glob
import numpy as np
import skimage.transform as skt

from . import py

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
    img = cv2.imread(img_path, 0)
    img = unpad_sample(img, d=32)
    return img

def tform_centred(radian, translation, center):
    # first translation, then rotation
    tform1 = skt.SimilarityTransform(translation=center)
    tform2 = skt.SimilarityTransform(rotation=radian)
    tform3 = skt.SimilarityTransform(translation=-center)
    tform4 = skt.SimilarityTransform(translation=translation)
    tform = tform4 + tform3 + tform2 + tform1
    return tform


def eliceiri_data_generator(data_root, mode, a2b=1, batch_size=1, load_GT=False, bidir=False):
#    data_root='./Datasets/Eliceiri_test/Eliceiri_Both'
#    batch_size=32
    """
    generator that takes in data of size [N, H, W], and yields data for our vxm model
    
    Note that we need to provide numpy data for each input, and each output
    
    inputs:  moving_image [bs, H, W, 1], fixed_image [bs, H, W, 1]
    outputs: moved_image  [bs, H, W, 1], zeros [bs, H, W, 2]
    """
    train_names_R = [os.path.basename(train_path) for train_path in glob.glob(f'{data_root}/A/{mode}/*_R.*')]
    random.shuffle(train_names_R)
    assert len(train_names_R) > 0, "Could not find any training data"
    assert a2b in [0, 1, 'a2a', 'b2b'], "a2b must be in [0, 1, 'a2a', 'b2b']"
    
    if a2b == 1:
        (modA, modB) = ('A', 'B')
    elif a2b == 0:
        (modA, modB) = ('B', 'A')
    elif a2b == 'a2a':
        (modA, modB) = ('A', 'A')
    elif a2b == 'b2b':
        (modA, modB) = ('B', 'B')
    
    phi_zeros = None
    while True:
        batch_names = np.random.choice(train_names_R, size=batch_size)
        X1s = []
        X2s = []
        if load_GT == True:
            X3s = []
            X4s = []
        for batch_name_R in batch_names:
            batch_name_T = batch_name_R.replace('_R', '_T')
            X1 = load_image(f'{data_root}/{modA}/{mode}/{batch_name_T}')
            X2 = load_image(f'{data_root}/{modB}/{mode}/{batch_name_R}')
            if X1.ndim == 2:
                X1 = X1[np.newaxis, ..., np.newaxis]
                X2 = X2[np.newaxis, ..., np.newaxis]
            X1s.append(X1)
            X2s.append(X2)
            if load_GT == True:
                X3 = load_image(f'{data_root}/{modA}/{mode}/{batch_name_R}')
                X4 = load_image(f'{data_root}/{modB}/{mode}/{batch_name_T}')
                if X3.ndim == 2:
                    X3 = X3[np.newaxis, ..., np.newaxis]
                    X4 = X4[np.newaxis, ..., np.newaxis]
                X3s.append(X3)
                X4s.append(X4)
        
        if batch_size > 1:
            X1s = np.concatenate(X1s, 0)
            X2s = np.concatenate(X2s, 0)
        else:
            X1s = X1s[0]
            X2s = X2s[0]
        
        X1s = X1s.astype('float') / 255
        X2s = X2s.astype('float') / 255
        
        inputs = [X1s, X2s]
        
        if phi_zeros is None:
            volshape = X1s.shape[1:-1]
            phi_zeros = np.zeros((batch_size, *volshape, len(volshape)))
        
        if load_GT == True:
            if batch_size > 1:
                X3s = np.concatenate(X3s, 0)
                X4s = np.concatenate(X4s, 0)
            else:
                X3s = X3s[0]
                X4s = X4s[0]
            X3s = X3s.astype('float') / 255
            X4s = X4s.astype('float') / 255
            outputs = [X3s, X4s] if bidir else [X3s]
        else:
            outputs = [X2s, X1s] if bidir else [X2s]
        
        yield inputs, outputs

def mnist_data_generator(x_data, a2b=1, batch_size=32, load_GT=False, bidir=True, tform='rigid'):
    """
    generator that takes in data of size [N, H, W], and yields data for our vxm model
    
    Note that we need to provide numpy data for each input, and each output
    
    inputs:  moving_image [bs, H, W, 1], fixed_image [bs, H, W, 1]
    outputs: moved_image  [bs, H, W, 1], zeros [bs, H, W, 2]
    """
    rot_min, rot_max = 10, 15
    trans_min, trans_max = 3, 8
    
    assert len(x_data) > 0, "Could not find any training data"
    assert a2b in [0, 1], "a2b must be in [0, 1]"
    
    if a2b == 1:
        (modA, modB) = ('A', 'B')
    else:
        (modA, modB) = ('B', 'A')
    
    zeros = None
    while True:
        idx2 = np.random.randint(0, len(x_data), size=batch_size)
        X2s = x_data[idx2, ..., np.newaxis] # modB_R
        if tform in ['rigid', 'mix']:
            X1s = []    # modA_T
            for X2 in X2s:
                # random transformation parameters
                rot_degree = random.choice((random.uniform(-rot_max, -rot_min), random.uniform(rot_min, rot_max)))
                tx = random.choice((random.uniform(-trans_max, -trans_min), random.uniform(trans_min, trans_max)))
                ty = random.choice((random.uniform(-trans_max, -trans_min), random.uniform(trans_min, trans_max)))
                rot_radian = np.deg2rad(rot_degree)
                
                # transform image
                centre_img = np.array((X2.shape[0], X2.shape[1])) / 2. - 0.5
                tform_img = tform_centred(radian=rot_radian, translation=(tx, ty), center=centre_img)
                X1 = skt.warp(X2, tform_img, preserve_range=True)
    #            skio.imshow(np.squeeze(X1))
                if X1.ndim == 3:
                    X1 = X1[np.newaxis, ...]
                X1s.append(X1)
        else:
            idx1 = np.random.randint(0, len(x_data), size=batch_size)
            X1s = x_data[idx1, ..., np.newaxis] # modA_T
            
        if tform in ['rigid', 'mix']:
            if batch_size > 1:
                X1s = np.concatenate(X1s, 0)
            else:
                X1s = X1s[0]
            
        X1s = X1s.astype('float')
        X2s = 1 - X2s
        
        inputs = [X1s, X2s]
        outputs = [X2s, X1s] if bidir else [X2s]
        
        if load_GT == True:
            X3s = 1 - X2s  # modA_R
            X4s = 1 - X1s  # modB_T
            outputs = [X3s, X4s] if bidir else [X3s]
        
        if tform in ['dense', 'mix']:
            if zeros is None:
                shape = x_data.shape[1:]
                zeros = np.zeros((batch_size, *shape, len(shape)))
            outputs.append(zeros)
        
        yield inputs, outputs


def volgen(
        vol_names,
        batch_size=1, 
        return_segs=False,
        np_var='vol',
        pad_shape=None,
        resize_factor=1,
        add_feat_axis=True
    ):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern or a list of file paths. Corresponding
    segmentations are additionally loaded if return_segs is set to True. If
    loading segmentations, npz files with variable names 'vol' and 'seg' are
    expected.

    Parameters:
        vol_names: Path, glob pattern or list of volume files to load.
        batch_size: Batch size. Default is 1.
        return_segs: Loads corresponding segmentations. Default is False.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis, pad_shape=pad_shape, resize_factor=resize_factor)
        imgs = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        # optionally load segmentations and concatenate
        if return_segs:
            load_params['np_var'] = 'seg'  # be sure to load seg
            segs = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
            vols.append(np.concatenate(segs, axis=0))

        yield tuple(vols)


def scan_to_scan(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan1 = next(gen)[0]
        scan2 = next(gen)[0]

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols  = [scan1, scan2]
        outvols = [scan2, scan1] if bidir else [scan2]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)


def scan_to_atlas(vol_names, atlas, bidir=False, batch_size=1, no_warp=False, **kwargs):
    """
    Generator for scan-to-atlas registration.

    TODO: This could be merged into scan_to_scan() by adding an optional atlas
    argument like in semisupervised().

    Parameters:
        vol_names: List of volume files to load.
        atlas: Atlas volume data.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan = next(gen)[0]
        invols  = [scan, atlas]
        outvols = [atlas, scan] if bidir else [atlas]
        if not no_warp:
            outvols.append(zeros)
        yield (invols, outvols)


def semisupervised(vol_names, labels, atlas_file=None, downsize=2):
    """
    Generator for semi-supervised registration training using ground truth segmentations.
    Scan-to-atlas training can be enabled by providing the atlas_file argument. It's
    expected that vol_names and atlas_file are npz files with both 'vol' and 'seg' arrays.

    Parameters:
        vol_names: List of volume npz files to load.
        labels: Array of discrete label values to use in training.
        atlas_file: Atlas npz file for scan-to-atlas training. Default is None.
        downsize: Downsize factor for segmentations. Default is 2.
    """
    # configure base generator
    gen = volgen(vol_names, return_segs=True, np_var='vol')
    zeros = None

    # internal utility to generate downsampled prob seg from discrete seg
    def split_seg(seg):
        prob_seg = np.zeros((*seg.shape[:4], len(labels)))
        for i, label in enumerate(labels):
            prob_seg[0, ..., i] = seg[0, ..., 0] == label
        return prob_seg[:, ::downsize, ::downsize, ::downsize, :]

    # cache target vols and segs if atlas is supplied
    if atlas_file:
        trg_vol = py.utils.load_volfile(atlas_file, np_var='vol', add_batch_axis=True, add_feat_axis=True)
        trg_seg = py.utils.load_volfile(atlas_file, np_var='seg', add_batch_axis=True, add_feat_axis=True)
        trg_seg = split_seg(trg_seg)

    while True:
        # load source vol and seg
        src_vol, src_seg = next(gen)
        src_seg = split_seg(src_seg)

        # load target vol and seg (if not provided by atlas)
        if not atlas_file:
            trg_vol, trg_seg = next(gen)
            trg_seg = split_seg(trg_seg)

        # cache zeros
        if zeros is None:
            shape = src_vol.shape[1:-1]
            zeros = np.zeros((1, *shape, len(shape)))

        invols  = [src_vol, trg_vol, src_seg]
        outvols = [trg_vol, zeros,   trg_seg]
        yield (invols, outvols)


def template_creation(vol_names, atlas, bidir=False, batch_size=1, **kwargs):
    """
    Generator for unconditional template creation.

    Parameters:
        vol_names: List of volume files to load.
        atlas: Atlas input volume data.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        kwargs: Forwarded to the internal volgen generator.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan = next(gen)[0]
        invols  = [atlas, scan]  # TODO: this is opposite of the normal ordering and might be confusing
        outvols = [scan, atlas, zeros, zeros] if bidir else [scan, zeros, zeros]
        yield (invols, outvols)


def conditional_template_creation(vol_names, atlas, attributes, batch_size=1, np_var='vol', pad_shape=None, add_feat_axis=True):
    """
    Generator for conditional template creation.

    Parameters:
        vol_names: List of volume files to load.
        atlas: Atlas input volume data.
        attributes: Dictionary of phenotype data for each vol name.
        batch_size: Batch size. Default is 1.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    while True:
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load pheno from attributes dictionary
        pheno = np.stack([attributes[vol_names[i]] for i in indices], axis=0)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis, pad_shape=pad_shape)
        vols = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = np.concatenate(vols, axis=0)

        invols  = [pheno, atlas, vols]
        outvols = [vols, zeros, zeros, zeros]
        yield (invols, outvols)


def surf_semisupervised(
        vol_names,
        atlas_vol,
        atlas_seg,
        nb_surface_pts,
        labels=None,
        batch_size=1,
        surf_bidir=True,
        surface_pts_upsample_factor=2,
        smooth_seg_std=1,
        nb_labels_sample=None,
        sdt_vol_resize=1,
        align_segs=False,
        add_feat_axis=True
    ):
    """
    Scan-to-atlas generator for semi-supervised learning using surface point clouds from segmentations.

    Parameters:
        vol_names: List of volume files to load.
        atlas_vol: Atlas volume array.
        atlas_seg: Atlas segmentation array.
        nb_surface_pts: Total number surface points for all structures.
        labels: Label list to include. If None, all labels in atlas_seg are used. Default is None.
        batch_size: Batch size. NOTE some features only implemented for 1. Default is 1.
        surf_bidir: Train with bidirectional surface distance. Default is True.
        surface_pts_upsample_factor: Upsample factor for surface pointcloud. Default is 2.
        smooth_seg_std: Segmentation smoothness sigma. Default is 1.
        nb_labels_sample: Number of labels to sample. Default is None.
        sdt_vol_resize: Resize factor for signed distance transform volumes. Default is 1.
        align_segs: Whether to pass in segmentation image instead. Default is False.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # some input checks
    assert nb_surface_pts > 0, 'number of surface point should be greater than 0'

    # prepare some shapes
    vol_shape = atlas_seg.shape
    sdt_shape = [int(f * sdt_vol_resize) for f in vol_shape]

    # compute labels from atlas, and the number of labels to sample.
    if labels is not None:
        atlas_seg = py.utils.filter_labels(atlas_seg, labels)
    else:
        labels = np.sort(np.unique(atlas_seg))[1:]

    # use all labels by default
    if nb_labels_sample is None:
        nb_labels_sample = len(labels)

    # prepare keras format atlases
    atlas_vol_bs = np.repeat(atlas_vol[np.newaxis, ..., np.newaxis], batch_size, axis=0)
    atlas_seg_bs = np.repeat(atlas_seg[np.newaxis, ..., np.newaxis], batch_size, axis=0)

    # prepare surface extraction function
    std_to_surf = lambda x, y: py.utils.sdt_to_surface_pts(x, y, surface_pts_upsample_factor=surface_pts_upsample_factor, thr=(1/surface_pts_upsample_factor + 1e-5))
    
    # prepare zeros, which will be used for outputs unused in cost functions
    zero_flow = np.zeros((batch_size, *vol_shape, len(vol_shape)))
    zero_surface_values = np.zeros((batch_size, nb_surface_pts, 1))

    # precompute label edge volumes
    atlas_sdt = [None] * len(labels) 
    atlas_label_vols = [None] * len(labels) 
    nb_edges = np.zeros(len(labels))
    for li, label in enumerate(labels):  # if only one label, get surface points here
        atlas_label_vols[li] = atlas_seg == label
        atlas_label_vols[li] = py.utils.clean_seg(atlas_label_vols[li], smooth_seg_std)
        atlas_sdt[li] = py.utils.vol_to_sdt(atlas_label_vols[li], sdt=True, sdt_vol_resize=sdt_vol_resize)
        nb_edges[li] = np.sum(np.abs(atlas_sdt[li]) < 1.01)
    layer_edge_ratios = nb_edges / np.sum(nb_edges)

    # if working with all the labels passed in (i.e. no label sampling per batch), 
    # pre-compute the atlas surface points
    atlas_surface_pts = np.zeros((batch_size, nb_surface_pts, len(vol_shape) + 1))
    if nb_labels_sample == len(labels):
        nb_surface_pts_sel = py.utils.get_surface_pts_per_label(nb_surface_pts, layer_edge_ratios)
        for li, label in enumerate(labels):  # if only one label, get surface points here
            atlas_surface_pts_ = std_to_surf(atlas_sdt[li], nb_surface_pts_sel[li])[np.newaxis, ...]
            # get the surface point stack indexes for this element
            srf_idx = slice(int(np.sum(nb_surface_pts_sel[:li])), int(np.sum(nb_surface_pts_sel[:li+1])))
            atlas_surface_pts[:, srf_idx, :-1] = np.repeat(atlas_surface_pts_, batch_size, 0)
            atlas_surface_pts[:, srf_idx,  -1] = li

    # generator
    gen = volgen(vol_names, return_segs=True, batch_size=batch_size, add_feat_axis=add_feat_axis)
    
    assert batch_size == 1, 'only batch size 1 supported for now'

    while True:

        # prepare data
        X = next(gen)
        X_img = X[0]
        X_seg = py.utils.filter_labels(X[1], labels)

        # get random labels
        sel_label_idxs = range(len(labels))  # all labels
        if nb_labels_sample != len(labels):
            sel_label_idxs = np.sort(np.random.choice(range(len(labels)), size=nb_labels_sample, replace=False))
            sel_layer_edge_ratios = [layer_edge_ratios[li] for li in sel_label_idxs]
            nb_surface_pts_sel = py.utils.get_surface_pts_per_label(nb_surface_pts, sel_layer_edge_ratios)
                
        # prepare signed distance transforms and surface point arrays
        X_sdt_k = np.zeros((batch_size, *sdt_shape, nb_labels_sample))
        atl_dt_k = np.zeros((batch_size, *sdt_shape, nb_labels_sample))
        subj_surface_pts = np.zeros((batch_size, nb_surface_pts, len(vol_shape) + 1))
        if nb_labels_sample != len(labels):
            atlas_surface_pts = np.zeros((batch_size, nb_surface_pts, len(vol_shape) + 1))

        for li, sli in enumerate(sel_label_idxs):
            # get the surface point stack indexes for this element
            srf_idx = slice(int(np.sum(nb_surface_pts_sel[:li])), int(np.sum(nb_surface_pts_sel[:li+1])))

            # get atlas surface points for this label
            if nb_labels_sample != len(labels):
                atlas_surface_pts_ = std_to_surf(atlas_sdt[sli], nb_surface_pts_sel[li])[np.newaxis, ...]
                atlas_surface_pts[:, srf_idx, :-1] = np.repeat(atlas_surface_pts_, batch_size, 0)
                atlas_surface_pts[:, srf_idx,  -1] = sli

            # compute X distance from surface
            X_label = X_seg == labels[sli]
            X_label = py.utils.clean_seg_batch(X_label, smooth_seg_std)
            X_sdt_k[..., li] = py.utils.vol_to_sdt_batch(X_label, sdt=True, sdt_vol_resize=sdt_vol_resize)[..., 0]

            if surf_bidir:
                atl_dt = atlas_sdt[li][np.newaxis, ...]
                atl_dt_k[..., li] = np.repeat(atl_dt, batch_size, 0)
                ssp_lst = [std_to_surf(f[...], nb_surface_pts_sel[li]) for f in X_sdt_k[..., li]]
                subj_surface_pts[:, srf_idx, :-1] = np.stack(ssp_lst, 0)
                subj_surface_pts[:, srf_idx,  -1] = li

        # check if returning segmentations instead of images
        # this is a bit hacky for basically building a segmentation-only network (no images)
        X_ret = X_img
        atlas_ret = atlas_vol_bs

        if align_segs:
            assert len(labels) == 1, 'align_seg generator is only implemented for single label'
            X_ret = X_seg == labels[0]
            atlas_ret = atlas_seg_bs == labels[0]

        # finally, output
        if surf_bidir:
            inputs  = [X_ret, atlas_ret, X_sdt_k, atl_dt_k, subj_surface_pts, atlas_surface_pts]
            outputs = [atlas_ret, X_ret, zero_flow, zero_surface_values, zero_surface_values]
        else:
            inputs  = [X_ret, atlas_ret, X_sdt_k, atlas_surface_pts]
            outputs = [atlas_ret, X_ret, zero_flow, zero_surface_values]

        yield (inputs, outputs)
