"""
Trains a segmentation network in an unsupervised fashion, using a probabilistic
atlas and unlabeled scans.

Unsupervised deep learning for Bayesian brain MRI segmentation
A.V. Dalca, E. Yu, P. Golland, B. Fischl, M.R. Sabuncu, J.E. Iglesias
Under Review. arXiv https://arxiv.org/abs/1904.11319
"""

import os
import random
import argparse
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import voxelmorph as vxm


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('datadir', help='base data directory')
parser.add_argument('--atlas', required=True, help='atlas filename')
parser.add_argument('--mapping', help='atlas mapping filename')
parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.00001)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--no-warp-atlas', action='store_true', help='disable atlas warp method within network')
parser.add_argument('--stat-pre-warp', action='store_true', help='compute gaussian stats before applying warp to atlas')
parser.add_argument('--init-stat', help='npz file defining guesses for initial stats (with arrays init_mu and init_sigma)')

# loss hyperparameters
parser.add_argument('--grad-loss-weight', type=float, default=10.0, help='weight of gradient loss (lamba) (default: 10.0)')
args = parser.parse_args()


# load reference atlas (group labels in tissue types if necessary)
atlas_full = vxm.py.utils.load_volfile(args.atlas, add_batch_axis=True)
if args.mapping:
    mapping = np.load(args.mapping)['mapping'].astype('int').flatten()
    assert len(mapping) == atlas_full.shape[-1], \
        'mapping shape %d is inconsistent with atlas shape %d' % (len(mapping), atlas_full.shape[-1])
    nb_labels = int(1 + np.max(mapping))
    atlas = np.zeros([*atlas_full.shape[:-1], nb_labels])
    for i in range(np.max(mapping.shape)):
        atlas[0, ..., mapping[i]] = atlas[0, ..., mapping[i]] + atlas_full[0, ..., i]
else:
    atlas = atlas_full
    nb_labels = atlas.shape[-1]

# get input shape
inshape = atlas.shape[1:-1]

# load guesses for means and variances
init_mu = np.load(args.init_stat)['init_mu'] if args.init_stat else None
init_sigma = np.load(args.init_stat)['init_sigma'] if args.init_stat else None

# load and prepare training data
train_vol_names = glob.glob(os.path.join(args.datadir, '*.npz'))
random.shuffle(train_vol_names)  # shuffle volume list
assert len(train_vol_names) > 0, 'Could not find any training data'

# scan-to-atlas generator
generator = vxm.generators.scan_to_atlas(train_vol_names, atlas, batch_size=args.batch_size)

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# tensorflow gpu handling
device = '/gpu:' + args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
tf.keras.backend.set_session(tf.Session(config=config))

# ensure valid batch size given gpu count
nb_gpus = len(args.gpu.split(','))
assert np.mod(args.batch_size, nb_gpus) == 0, 'Batch size (%d) should be a multiple of the number of gpus (%d)' % (args.batch_size, nb_gpus)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

with tf.device(device):

    warp_atlas = not args.no_warp_atlas

    # build the model
    model = vxm.networks.ProbAtlasSegmentation(
        inshape,
        nb_unet_features=[enc_nf, dec_nf],
        nb_labels=nb_labels,
        stat_post_warp=(not args.stat_pre_warp),
        warp_atlas=warp_atlas,
        init_mu=init_mu,
        init_sigma=init_sigma
    )

    # load initial weights (if provided)
    if args.load_weights:
        model.load_weights(args.load_weights)

    # prepare loss
    def loss(_, yp):
        m = tf.cast(model.inputs[0] > 0, tf.float32)
        return -K.sum(yp * m) / K.sum(m)

    grad_weight = args.grad_loss_weight if warp_atlas else 0

    losses  = [loss, vxm.losses.Grad('l2').loss]
    weights = [1.0, grad_weight]

    # multi-gpu support
    if nb_gpus > 1:
        save_callback = vxm.networks.ModelCheckpointParallel(save_filename)
        model = tf.keras.utils.multi_gpu_model(model, gpus=nb_gpus)
    else:
        save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

    # save starting weights
    model.save(save_filename.format(epoch=args.initial_epoch))

    model.fit_generator(generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[save_callback],
        verbose=1
    )

    # save final model weights
    model.save(save_filename.format(epoch=args.epochs))
