[![License](https://img.shields.io/badge/license-MIT-green?style=flat)](./LICENSE) [![](https://img.shields.io/badge/python-3.6+-blue.svg?style=flat)](https://www.python.org/download/releases/3.6.0/) [![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) 

# Evaluation of Multimodal Biomedical Image Registration Methods



## Dependencies

[`environment.yml`](./environment.yml) includes the **full** list of packages used to run most of the experiments. Some packages might be unnecessary. And here are some exceptions:

* For [StarGAN v2](https://github.com/clovaai/stargan-v2), please follow their [dependency list](https://github.com/clovaai/stargan-v2#software-installation).
* For [CoMIR](https://github.com/MIDA-group/CoMIR), to reduce GPU memory usage, the inference on GPU requires `pytorch>=1.6` to use the [Automatic Mixed Precision package](https://pytorch.org/docs/stable/amp.html), otherwise it uses *half-precision*.

## Execution order

### Balvan's Data

- Dataset preparation

```bash
# raw data cleaning
# data_skluzavky -> Balvan
python ./utils/extract_balvan_modalites.py

# prepare training data for pix2pix and others (3 folds)
# Balvan -> Balvan_temp
python ./utils/prepare_Balvan.py

# make evaluation patches
# Balvan -> Balvan_1to4tiles -> Balvan_patches (3 folds)
python ./utils/make_balvan_patches.py
```

- pix2pix and CycleGAN

```bash
# train and test 
cd pytorch-CycleGAN-and-pix2pix/
./commands_balvan.sh {fold} {gpu_id}

# modality mapping of evaluation data
# Balvan_patches -> Balvan_patches_fake
./predict_balvan.sh
```

- DRIT++

```bash
# train and test 
cd ../DRIT/src/
./commands_balvan.sh

# modality mapping of evaluation data
# Balvan_patches -> Balvan_patches_fake
./predict_balvan.sh
```

- VoxelMorph

```bash
# train affine model
cd voxelmorph-redesign/
./commands_balvan.sh

# evaluate
python test_affine.py
```

- Evaluate methods

```bash
python evaluate.py
```

------



### Zurich Data

- Dataset preparation

```bash
# raw data cleaning
# Zurich_dataset_v1.0 -> Zurich
python ./utils/extract_zurich_modalites.py

# prepare training data for pix2pix and others
# Zurich -> Zurich_temp (3 folds)
python ./utils/prepare_Zurich.py

# make evaluation patches
# Zurich -> Zurich_tiles -> Zurich_patches (3 folds)
python ./utils/make_zurich_patches.py

```

- pix2pix and CycleGAN

```bash
# train and test 
cd pytorch-CycleGAN-and-pix2pix/
./commands_zurich.sh

# modality mapping of evaluation data
# Zurich_patches -> Zurich_patches_fake
./predict_zurich.sh
```

- DRIT++

```bash
# train and test 
cd ../DRIT/src/
./commands_zurich.sh

# modality mapping of evaluation data
# Zurich_patches -> Zurich_patches_fake
./predict_zurich.sh
```

- Evaluate methods

```bash
python evaluate.py
```

------



### Eliceiri's Dada

- Dataset preparation

```bash
# prepare training data for pix2pix and others
# HighRes_Splits/CNN_Training/ -> Eliceiri_temp
python ./utils/prepare_Eliceiri.py

# make evaluation patches
# HighRes_Splits/WSI/ -> Eliceiri_patches
python ./utils/make_eliceiri_patches.py
```

- pix2pix and CycleGAN

```bash
# train and test 
cd pytorch-CycleGAN-and-pix2pix/
./commands_eliceiri.sh {gpu_id}

# modality mapping of evaluation data
# Eliceiri_patches -> Eliceiri_patches_fake
./predict_eliceiri.sh
```







## Code Reference

- [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [DRIT/DRIT++](https://github.com/HsinYingLee/DRIT) 
- [StarGAN v2](https://github.com/clovaai/stargan-v2)
- [CoMIR](https://github.com/MIDA-group/CoMIR)
- MUNIT
  - [PyTorch](https://github.com/NVlabs/MUNIT) (official)
  - [TensorFlow](https://github.com/taki0112/MUNIT-Tensorflow)

