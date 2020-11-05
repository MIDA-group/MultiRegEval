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

# %%
# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
#parser.add_argument('--datadir', default='./data/', help='base data directory')

parser.add_argument("--data_root", type=str,
                    dest="data_root", default='../Datasets/Eliceiri_patches/patch_trans10_rot5',
                    help="data folder")
parser.add_argument("--model_name", type=str,
                    dest="model_name", default='test',
                    help="models folder")
parser.add_argument("--supervised", dest="supervised", action="store_true")
parser.set_defaults(supervised=False)
parser.add_argument("--max_clip", type=float,
                    dest="max_clip", default=1,
                    help="maximum input value to calculate bins")
parser.add_argument("--num_bins", type=int,
                    dest="num_bins", default=48,
                    help="number of bins when calculating mutual information")

#parser.add_argument('--atlas', help='atlas filename')
#parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 1)')
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
data_root = args.data_root
#mode = 'train'
supervised = args.supervised
#args.steps_per_epoch = 100
# %%
## load and prepare training data
#train_vol_names = glob.glob(os.path.join(args.datadir, '*.npz'))
#random.shuffle(train_vol_names)  # shuffle volume list
#assert len(train_vol_names) > 0, 'Could not find any training data'
#
#generator_args = dict(no_warp=True, batch_size=batch_size, pad_shape=args.padding, resize_factor=args.resize, bidir=args.bidir)
#
#generator = vxm.generators.scan_to_scan(train_vol_names, prob_same=args.prob_same, **generator_args)


# %%
# let's test it
train_generator = vxm.generators.eliceiri_data_generator(
        data_root=data_root,
        mode='train',
        batch_size=args.batch_size,
        load_GT=supervised)
input_sample, output_sample = next(train_generator)
steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch else math.ceil(len(glob.glob(f'{data_root}/A/train/*_R.tif')) / batch_size)

# visualize
slices_2d = [f[0,...,0] for f in input_sample + output_sample]
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
    model = vxm.networks.VxmAffine(
        inshape,
        enc_nf=enc_nf,
        transform_type=transform_type,
        bidir=args.bidir,
        blurs=args.blurs
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
    if supervised == True:
        image_loss_func = vxm.losses.MSE().loss
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
# %%

# =============================================================================
# # test
# val_generator = vxm.generators.eliceiri_data_generator(
#         data_root=data_root,
#         mode='test',
#         batch_size=args.batch_size,
#         load_GT=True)
# # %%
# val_input, val_output = next(val_generator)
# 
# val_pred = model.predict(val_input)
# 
# # visualize
# slices_2d = [f[0,...,0] for f in val_input + val_output[:1] + [val_pred]]
# titles = ['input_moving', 'input_fixed', 'output_moved_ground_truth', 'predicted_moved']
# vxm.tf.neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);
# =============================================================================

