# voxelmorph

Unsupervised Learning with CNNs for Image Registration  
This repository incorporates several variants, first presented at CVPR2018 (initial unsupervised learning) and then MICCAI2018  (probabilistic & diffeomorphic formulation)

keywords: machine learning, convolutional neural networks, alignment, mapping, registration

## RecentUpdates

See our [learning method to automatically build atlases](http://voxelmorph.mit.edu/atlas_creation/) using VoxelMorph, to appear at NeurIPS2019.


# Instructions

## Training

If you would like to train your own model, you will likely need to customize some of the data loading code in `voxelmorph/generators.py` for your own datasets and data formats. However, it is possible to run many of the example scripts out-of-the-box, assuming that you have a directory containing training data files in npz (numpy) format. It's assumed that each npz file in your data folder has a `vol` parameter, which points to the numpy image data to be registered, and an optional `seg` variable, which points to a corresponding discrete segmentation. It's also assumed that the shape of all image data in a directory is consistent.

For a given `/path/to/training/data`, the following script will train the dense network (described in MICCAI 2018 by default) using scan-to-scan registration. Model weights will be saved to a path specified by the `--model-dir` flag.

```
python scripts/train.py /path/to/training/data --model-dir /path/to/models/output --gpu 0
```

Scan-to-atlas registration can be enabled by providing an atlas file with the `--atlas atlas.npz` command line flag. If you'd like to train using the original dense CVPR network (no diffeomorphism), use the `--int-steps 0` flag to specify no flow integration steps. Use the `--help` flag to inspect all of the command line options that can be used to fine-tune network architecture and training.

## Registration

If you simply want to register two images, you can use the `register.py` script with the desired model file. For example, if we have a model `model.h5` trained to register a subject (moving) to an atlas (fixed), we could run:

```
python scripts/register.py moving.nii.gz atlas.nii.gz warped.nii.gz --model model.h5 --gpu 0
```

This will save the moved image to `warped.nii.gz`. To also save the predicted deformation field, use the `--save-warp` flag. Both npz or nifty files can be used as input/output in this script.

## Testing (measuring Dice scores)

To test the quality of a model by computing dice overlap between an atlas segmentation and warped test scan segmentations, run:

```
python scripts/test.py --atlas data/atlas.npz --scans data/test_scan.npz --labels data/labels.npz --model models/model.h5 --gpu 0
```

Just like for the training data, the atlas and test npz files include `vol` and `seg` parameters and the `labels.npz` file contains a list of corresponding anatomical labels to include in the computed dice score.

## Parameter choices

### CVPR version

For the CC loss function, we found a reg parameter of 1 to work best. For the MSE loss function, we found 0.01 to work best.

### MICCAI version

For our data, we found `image_sigma=0.01` and `prior_lambda=25` to work best.

In the original MICCAI code, the parameters were applied after the scaling of the velocity field. With the newest code, this has been "fixed", with different default parameters reflecting the change. We recommend running the updated code. However, if you'd like to run the very original MICCAI2018 mode, please use `xy` indexing and `use_miccai_int` network option, with MICCAI2018 parameters.

## Spatial Transforms and Integration

- The spatial transform code, found at [`neuron.layers.SpatialTransform`](https://github.com/adalca/neuron/blob/master/neuron/layers.py), accepts N-dimensional affine and dense transforms, including linear and nearest neighbor interpolation options. Note that original development of VoxelMorph used `xy` indexing, whereas we are now emphasizing `ij` indexing.

- For the MICCAI2018 version, we integrate the velocity field using [`neuron.layers.VecInt`]((https://github.com/adalca/neuron/blob/master/neuron/layers.py)). By default we integrate using scaling and squaring, which we found efficient.


# VoxelMorph Papers

If you use voxelmorph or some part of the code, please cite (see [bibtex](citations.bib)):

  * For the atlas formation model:  

    **Learning Conditional Deformable Templates with Convolutional Networks**  
  [Adrian V. Dalca](http://adalca.mit.edu), [Marianne Rakic](https://mariannerakic.github.io/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)
  NeurIPS 2019. [eprint arXiv:1908.02738](https://arxiv.org/abs/1908.02738)

  * For the diffeomorphic or probabilistic model:

    **Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MedIA: Medial Image Analysis. 2019. [eprint arXiv:1903.03545](https://arxiv.org/abs/1903.03545) 

    **Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)

  * For the original CNN model, MSE, CC, or segmentation-based losses:

    **VoxelMorph: A Learning Framework for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
IEEE TMI: Transactions on Medical Imaging. 2019. 
[eprint arXiv:1809.05231](https://arxiv.org/abs/1809.05231)

    **An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)


# Notes on Data
In our initial papers, we used publicly available data, but unfortunately we cannot redistribute it (due to the constraints of those datasets). We do a certain amount of pre-processing for the brain images we work with, to eliminate sources of variation and be able to compare algorithms on a level playing field. In particular, we perform FreeSurfer `recon-all` steps up to skull stripping and affine normalization to Talairach space, and crop the images via `((48, 48), (31, 33), (3, 29))`. 

We encourage users to download and process their own data. See [a list of medical imaging datasets here](https://github.com/adalca/medical-datasets). Note that you likely do not need to perform all of the preprocessing steps, and indeed VoxelMorph has been used in other work with other data.


# Creation of Deformable Templates

We present a template consturction method in this [preprint](https://arxiv.org/abs/1908.02738): 

  *  **Learning Conditional Deformable Templates with Convolutional Networks**  
  [Adrian V. Dalca](http://adalca.mit.edu), [Marianne Rakic](https://mariannerakic.github.io/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)
  NeurIPS 2019. [eprint arXiv:1908.02738](https://arxiv.org/abs/1908.02738)

To experiment with this method, please use `train_template.py` for unconditional templates and `train_cond_template.py` for conditional templates, which use the same conventions as voxelmorph (please note that these files are less polished than the rest of the voxelmorph library).

We've also provided an unconditional atlas in `data/generated_uncond_atlas.npz.npy`. 

Models in h5 format weights are provided for [unconditional atlas here](http://people.csail.mit.edu/adalca/voxelmorph/atlas_creation_uncond_NCC_1500.h5), and [conditional atlas here](http://people.csail.mit.edu/adalca/voxelmorph/.h5).

**Explore the atlases [interactively here](http://voxelmorph.mit.edu/atlas_creation/)** with tipiX!


# Unified Segmentation

We recently published a method on deep learning methods for unsupervised segmentation that makes use of voxelmorph infrastructure. See the [unified seg README for more information](unified_seg/README.md).


# Significant Updates
2019-11-28: Added a preliminary version of pytorch
2019-08-08: Added support for building templates
2019-04-27: Added support for unified segmentation  
2019-01-07: Added example register.py file
2018-11-10: Added support for multi-gpu training  
2018-10-12: Significant overhaul of code, especially training scripts and new model files.  
2018-09-15: Added MICCAI2018 support and py3 transition  
2018-05-14: Initial Repository for CVPR version, py2.7


# Contact:
For any problems or questions please [open an issue](https://github.com/voxelmorph/voxelmorph/issues/new?labels=voxelmorph) in github.  
