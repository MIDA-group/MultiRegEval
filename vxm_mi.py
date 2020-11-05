# -*- coding: utf-8 -*-
# imports
import os, sys, random

# third party imports
import numpy as np
import keras.layers
import matplotlib.pyplot as plt
from glob import glob
import cv2

# VoxelMorph
sys.path.append('./voxelmorph/ext/pynd-lib/')
sys.path.append('./voxelmorph/ext/pytools-lib/')
sys.path.append('./voxelmorph/ext/neuron/')
#sys.path.append('./voxelmorph/')
import voxelmorph.src as vxm
import neuron

# %% data loader
def pad_sample(img, d):
    # pad the image size to multiple of divisor d
    (w, h) = img.shape[:2]
    w_pad = (w // d + 1) * d - w
    h_pad = (h // d + 1) * d - h            
    wl = w_pad // 2
    wr = w_pad - wl
    hl = h_pad // 2
    hr = h_pad - hl
    if img.ndim == 2:
        img_pad = np.pad(img, ((wl, wr), (hl, hr)), 'mean')#'constant', constant_values=0)
    else:
        img_pad = np.pad(img, ((wl, wr), (hl, hr), (0, 0)), 'mean')#'constant', constant_values=0)
    return img_pad

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

def eliceiri_data_generator(data_root, mode, batch_size=1, load_GT=False):
#    data_root='./Datasets/Eliceiri_test/Eliceiri_Both'
#    batch_size=32
    """
    generator that takes in data of size [N, H, W], and yields data for our vxm model
    
    Note that we need to provide numpy data for each input, and each output
    
    inputs:  moving_image [bs, H, W, 1], fixed_image [bs, H, W, 1]
    outputs: moved_image  [bs, H, W, 1], zeros [bs, H, W, 2]
    """
    train_names_R = [os.path.basename(train_path) for train_path in glob(f'{data_root}/A/{mode}/*_R.tif')]
    random.shuffle(train_names_R)
    assert len(train_names_R) > 0, "Could not find any training data"
    
    phi_zeros = None
    while True:
        batch_names = np.random.choice(train_names_R, size=batch_size)
        X1s = []
        X2s = []
        if load_GT == True:
            X3s = []
        for batch_name_R in batch_names:
            batch_name_T = batch_name_R.replace('_R', '_T')
            X1 = load_image(f'{data_root}/A/{mode}/{batch_name_T}')
            X2 = load_image(f'{data_root}/B/{mode}/{batch_name_R}')
            if X1.ndim == 2:
                X1 = X1[np.newaxis, ..., np.newaxis]
                X2 = X2[np.newaxis, ..., np.newaxis]
            X1s.append(X1)
            X2s.append(X2)
            if load_GT == True:
                X3 = load_image(f'{data_root}/A/{mode}/{batch_name_R}')
                if X3.ndim == 2:
                    X3 = X3[np.newaxis, ..., np.newaxis]
                X3s.append(X3)
        
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
            else:
                X3s = X3s[0]
            X3s = X3s.astype('float') / 255
            outputs = [X3s, phi_zeros]
        else:
            outputs = [X2s, phi_zeros]
        
        yield inputs, outputs

# %%
supervised = False
# let's test it
train_generator = eliceiri_data_generator(
        data_root='./Datasets/Eliceiri_patches/patch_trans10_rot5',
        mode='train',
        batch_size=1,
        load_GT=supervised)
input_sample, output_sample = next(train_generator)

# visualize
slices_2d = [f[0,...,0] for f in input_sample + output_sample]
titles = ['input_moving', 'input_fixed', 'output_moved_ground_truth', 'zero']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

# %% Model
nb_enc_features = [16, 32, 32, 32]
nb_dec_features = [32, 32, 32, 32, 32, 16, 16]
# first, let's get a unet (before the final layer)
vol_shape = input_sample[0].shape[1:-1]
ndims = len(vol_shape)
unet = vxm.networks.unet_core(vol_shape, nb_enc_features, nb_dec_features);

# inputs
print('numer of inputs', len(unet.inputs))
moving_input_tensor = unet.inputs[0]
fixed_input_tensor = unet.inputs[1]
    
# output
print('output:', unet.output)

# transform the results into a flow field.
disp_tensor = keras.layers.Conv2D(ndims, kernel_size=3, padding='same', name='disp')(unet.output)
# check
print('displacement tensor:', disp_tensor)

# a cool aspect of keras is that we can easily form new models via tensor pointers:
def_model = keras.models.Model(unet.inputs, disp_tensor)
# def_model will now *share layers* with the UNet -- if we change layer weights 
# in the UNet, they change in the def_model 

spatial_transformer = neuron.layers.SpatialTransformer(name='spatial_transformer')

# warp the image
moved_image_tensor = spatial_transformer([moving_input_tensor, disp_tensor])

inputs = [moving_input_tensor, fixed_input_tensor]
outputs = [moved_image_tensor, disp_tensor]
vxm_model = keras.models.Model(inputs, outputs)

# losses. Keras recognizes the string 'mse' as mean squared error, so we don't have to code it
#losses = ['mse', vxm.losses.Grad('l2').loss]

# MI loss test
max_clip = 1
num_bins = 48
bin_centers = np.linspace(0, max_clip, num_bins*2+1)[1::2]
if supervised == True:
    loss_function = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
else:
    loss_function = [
            vxm.losses.mutualInformation(
                    bin_centers,
                    max_clip=max_clip,
                    local_mi=False), 
            vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter.
lambda_param = 5.0 # 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=loss_function, loss_weights=loss_weights)

# %% Train


nb_epochs = 10
steps_per_epoch = 100
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2);


def plot_history(hist, loss_name='loss'):
    """
    Quick function to plot the history 
    """
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
plot_history(hist)

# %% Registration
# let's get some data
val_generator = eliceiri_data_generator(
        data_root='./Datasets/Eliceiri_patches/patch_trans10_rot5',
        mode='test',
        batch_size=1,
        load_GT=True)
val_input, val_output = next(val_generator)

val_pred = vxm_model.predict(val_input)

# visualize
slices_2d = [f[0,...,0] for f in val_input + val_output[:1] + val_pred]
titles = ['input_moving', 'input_fixed', 'output_moved_ground_truth', 'predicted_moved', 'deformation_x']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

#neuron.plot.flow([val_pred[1].squeeze()], width=5);
