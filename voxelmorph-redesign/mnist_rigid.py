"""
Example script to train an affine VoxelMorph model.

For the CVPR and MICCAI papers, we have data arranged in train, validate, and test folders. Inside each folder
are normalized T1 volumes and segmentations in npz (numpy) format. You will have to customize this script slightly
to accommodate your own data. All images should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed. Otherwise,
registration will be scan-to-scan.
"""

import os
import random
import argparse
import glob
import math
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


# %%
# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
#parser.add_argument('--datadir', default='./data/', help='base data directory')

parser.add_argument("--data_root", type=str,
                    dest="data_root", default='',
                    help="data folder")
parser.add_argument("--model_name", type=str,
                    dest="model_name", default='mnist',
                    help="models folder")
parser.add_argument("--supervised", dest="supervised", action="store_true")
parser.set_defaults(supervised=False)
parser.add_argument("--max_clip", type=float,
                    dest="max_clip", default=1,
                    help="maximum input value to calculate bins")
parser.add_argument("--num_bins", type=int,
                    dest="num_bins", default=48,
                    help="number of bins when calculating mutual information")
parser.add_argument("--a2b", type=int, default=0, choices=[0, 1], help='A:SHG, B:BF')

#parser.add_argument('--atlas', help='atlas filename')
#parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs (default: 1500)')
parser.add_argument('--steps_per_epoch', type=int, help='frequency of model saves (default: 100)')
parser.add_argument('--load_weights', help='optional weights file to initialize with')
parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.00001)')
parser.add_argument('--prob-same', type=float, default=0.3, help='likelihood that source/target training images will be the same (default: 0.3)')

# network architecture parameters
parser.add_argument('--rigid', action='store_true', help='force rigid registration')
parser.set_defaults(rigid=True)
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--blurs', type=float, nargs='+', default=[1], help='levels of gaussian blur kernel for each scale (default: 1)')
#parser.add_argument('--padding', type=int, nargs='+', default=[256, 256, 256], help='padded image target shape (default: 256 256 256')
#parser.add_argument('--resize', type=float, default=0.25, help='after-padding image resize factor (default: 0.25)')
args = parser.parse_args()

batch_size = args.batch_size
#mode = 'train'
#args.supervised = False
#args.steps_per_epoch = 100
args.bidir = True
#args.blurs = [5, 4, 3, 2, 1]

args.model_name = 'mnist_nonrigid_us_b2a'
RIGID = False

# %%
def load_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    x_train = X_train[Y_train==4, ...]
    y_train = Y_train[Y_train==4]
    x_test = X_test[Y_test==4, ...]
    y_test = Y_test[Y_test==4]
    
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
#    vxm.tf.neuron.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True);

    return x_train, x_val, x_test, y_train, y_val, y_test

x_train, x_val, x_test, y_train, y_val, y_test = load_mnist()

# %%
# let's test it
train_generator = vxm.generators.mnist_data_generator(
        x_train, 
        a2b=args.a2b,
        batch_size=batch_size,
        load_GT=args.supervised,
        rigid=RIGID,
        bidir=args.bidir)
input_sample, output_sample = next(train_generator)
steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch else math.ceil(len(x_train) / batch_size)

# visualize
slices_2d = [f[0,...,0] for f in input_sample + output_sample[:1]]
titles = ['input_moving', 'input_fixed', 'output_moved_ground_truth']
vxm.tf.neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

# %%
# extract shape from sampled input
inshape = input_sample[0].shape[1:-1]

# prepare model folder
model_dir = "./models/" + args.model_name
# prepare model folder
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# tensorflow gpu handling
device = '/gpu:' + args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
tf.keras.backend.set_session(tf.Session(config=config))

# ensure valid batch size given gpu count
nb_gpus = len(args.gpu.split(','))
assert np.mod(batch_size, nb_gpus) == 0, 'Batch size (%d) should be a multiple of the number of gpus (%d)' % (batch_size, nb_gpus)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

# transform type
transform_type = 'rigid' if args.rigid else 'affine'

# %%
with tf.device(device):

    # build the model
    if RIGID:
        model = vxm.networks.VxmAffine(
            inshape,
            enc_nf=enc_nf,
            transform_type=transform_type,
            bidir=args.bidir,
            blurs=args.blurs
        )
    else:
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        nfeats = input_sample[0].shape[-1]
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=args.bidir,
#            use_probs=args.use_probs,
#            int_steps=args.int_steps,
#            int_downsize=args.int_downsize,
            src_feats=nfeats,
            trg_feats=nfeats
        )

    # load initial weights (if provided)
    if args.load_weights:
        print('loading', args.load_weights)
        model.load_weights(args.load_weights)

    # multi-gpu support
    if nb_gpus > 1:
        save_callback = vxm.networks.ModelCheckpointParallel(save_filename, period=10, verbose=1)
        model = tf.keras.utils.multi_gpu_model(model, gpus=nb_gpus)
    else:
        save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, period=10, verbose=1)

    # configure loss
    if args.supervised == True:
        image_loss_func = vxm.losses.NCC().loss
    else:
        bin_centers = np.linspace(0, args.max_clip, args.num_bins*2+1)[1::2]
        image_loss_func = vxm.losses.NMI(
                bin_centers=bin_centers, 
                vol_size=inshape, 
                max_clip=args.max_clip, 
                local=False
                ).loss
        


    # need two image loss functions if bidirectional
    if args.bidir:
        losses  = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses  = [image_loss_func]
        weights = [1]
        
    if not RIGID:
        losses += [vxm.losses.Grad('l2').loss]
        weights += [0.01]

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

    # save starting weights
    model.save(save_filename.format(epoch=args.initial_epoch))

    hist = model.fit_generator(
            train_generator,
            initial_epoch=args.initial_epoch,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[save_callback],
            verbose=1);

    # save final model weights
    model.save(save_filename.format(epoch=args.epochs))

vxm.tf.utils.plot_history(hist=hist, model_name=args.model_name)
# %% Test
# load trained model
model_name = 'mnist_us'
model_idx = '0500.h5'
model_dir = "./models/" + model_name
load_model_file = os.path.join(model_dir, model_idx)

model = vxm.networks.VxmAffine.load(load_model_file)

# %% test
val_generator = vxm.generators.mnist_data_generator(
#        x_val, 
        x_test, 
        a2b=args.a2b,
        batch_size=batch_size,
        load_GT=True,
        rigid=RIGID,
        bidir=args.bidir)
# %%
val_input, val_output = next(val_generator)

val_pred = model.predict(val_input)

# visualize
slices_2d = [f[0,...,0] for f in val_input + val_output[:1] + val_pred[:1]]
titles = ['input moving $M$', 'input fixed $F$', 'output ground truth', 'predicted moved $M_\phi$']
vxm.tf.neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

# %%
def plot_samples(model,                     # trained model
                 val_generator,           # the 2D slices
                 cols=10,                # number of samples  
                 titles=None,         # list of titles
                 save=False,           # option to actually show the plot (plt.show())
                 save_dir='./',		# dir to save img
                 save_name='sample.png'):
#    rows=10
#    cols=4
    width=20
#    fig, axs = plt.subplots(rows, cols)
    for col in range(cols):
        # get result slices
        val_input, val_output = next(val_generator)
        val_pred = model.predict(val_input)
        slices_in = [f[0,...,0] for f in val_input + val_output[:1] + val_pred[:1]]
        
        if col == 0:
            rows = len(slices_in)
            fig, axs = plt.subplots(rows, cols)
        
        for i in range(rows):
            ax = axs[i][col]
            ax.axis('off')
            if col == 0:
                ax.set_title(titles[i], fontsize='medium', rotation='vertical', verticalalignment='center', x=-0.1, y=0.5)
            im_ax = ax.imshow(slices_in[i], cmap='gray', interpolation="nearest")
            
    # show the plots
    fig.set_size_inches(width, rows/cols*width)
#    fig.subplots_adjust(hspace=0.01,wspace=0.25)

    if save:
        plt.savefig(f'{save_dir}/{save_name}', format='png', dpi=300, bbox_inches='tight', transparent=False)
        plt.tight_layout()
    else:
        plt.show()
    
    return (fig, axs)

plot_samples(model, val_generator, cols=10, titles=titles, save=True,
             save_dir='./result_samples/',
             save_name=f'{model_name}_trail.png')