# -*- coding: utf-8 -*-
# imports
import os, sys

# third party imports
import numpy as np
import keras.layers
import matplotlib.pyplot as plt


# VoxelMorph
sys.path.append('./voxelmorph/ext/pynd-lib/')
sys.path.append('./voxelmorph/ext/pytools-lib/')
sys.path.append('./voxelmorph/ext/neuron/')
#sys.path.append('./voxelmorph/')
import voxelmorph.src as vxm
import neuron
from tensorflow.keras.datasets import mnist

# %%
def load_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    x_train = X_train[Y_train==5, ...]
    y_train = Y_train[Y_train==5]
    x_test = X_test[Y_test==5, ...]
    y_test = Y_test[Y_test==5]
    
    nb_val = 1000 # keep 10,000 subjects for validation
    x_val = x_train[-nb_val:, ...]  # this indexing means "the last nb_val entries" of the zeroth axis
    y_val = y_train[-nb_val:]
    x_train = x_train[:-nb_val, ...]
    y_train = y_train[:-nb_val]

    # fix data
    x_train = x_train.astype('float')/255
    x_val = x_val.astype('float')/255
    x_test = x_test.astype('float')/255
    
    pad_amount = ((0, 0), (2,2), (2,2))
    
    # fix data
    x_train = np.pad(x_train, pad_amount, 'constant')
    x_val = np.pad(x_val, pad_amount, 'constant')
    x_test = np.pad(x_test, pad_amount, 'constant')
    
    # choose nb_vis sample indexes
    nb_vis = 5
    idx = np.random.choice(x_train.shape[0], nb_vis, replace=False)
    example_digits = [f for f in x_train[idx, ...]]
    # plot
    neuron.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True);

    return x_train, x_val, x_test, y_train, y_val, y_test

x_train, x_val, x_test, y_train, y_val, y_test = load_mnist()

# %% Model
ndims = 2
vol_shape = x_train.shape[1:]
nb_enc_features = [32, 32, 32, 32]
nb_dec_features = [32, 32, 32, 32, 32, 16]
# first, let's get a unet (before the final layer)
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
losses = [
        vxm.losses.mutualInformation(
                bin_centers,
                max_clip=max_clip,
                local_mi=False), 
        vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter.
lambda_param = 1.0 # 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# %% Train
def vxm_data_generator(x_data, batch_size=32):
    """
    generator that takes in data of size [N, H, W], and yields data for our vxm model
    
    Note that we need to provide numpy data for each input, and each output
    
    inputs:  moving_image [bs, H, W, 1], fixed_image [bs, H, W, 1]
    outputs: moved_image  [bs, H, W, 1], zeros [bs, H, W, 2]
    """
    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation. We'll explain this below.
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs
        # inputs need to be of the size [batch_size, H, W, number_features]
        #   number_features at input is 1 for us
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # outputs
        # we need to prepare the "true" moved image.  
        # Of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield inputs, outputs

# let's test it
train_generator = vxm_data_generator(x_train)
input_sample, output_sample = next(train_generator)

# visualize
slices_2d = [f[0,...,0] for f in input_sample + output_sample]
titles = ['input_moving', 'input_fixed', 'output_moved_ground_truth', 'zero']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

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
#plot_history(hist)

# %% Registration
# let's get some data
val_generator = vxm_data_generator(x_val, batch_size = 1)
val_input, _ = next(val_generator)

val_pred = vxm_model.predict(val_input)

# visualize
slices_2d = [f[0,...,0] for f in val_input + val_pred]
titles = ['input_moving', 'input_fixed', 'predicted_moved', 'deformation_x']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

neuron.plot.flow([val_pred[1].squeeze()], width=5);
