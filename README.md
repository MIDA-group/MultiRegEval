[![License](https://img.shields.io/badge/license-MIT-green?style=flat)](./LICENSE) [![](https://img.shields.io/badge/python-3.6+-blue.svg?style=flat)](https://www.python.org/download/releases/3.6.0/) [![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) 

# Evaluation Framework for Multimodal Biomedical Image Registration Methods

Code of paper [Is Image-to-Image Translation the Panacea for Multimodal Image Registration? A Comparative Study](https://arxiv.org/abs/2103.16262)

Open-access data: [Datasets for Evaluation of Multimodal Image Registration](https://zenodo.org/record/4587903)

## Overview

This repository provides an open-source quantitative evaluation framework for multimodal biomedical registration, aiming to contribute to the openness and reproducibility of future research.

- [`evaluate.py`](./evaluate.py) is the main script to call the registration methods and calculate their performance.

- [`data_description.md`](./Datasets/data_description.md) contains detailed descriptions of the evaluation datasets, and instructions to customise them.
- The `*.sh` scripts provide examples to set large-scale evaluations.
- [`plot.py`](./plot.py) and [`show_samples.py`](show_samples.py) can be used to plot the registration performance and visualise the modality-translation results (see [paper](https://arxiv.org/abs/2103.16262) for examples).
- Each folder contains the modified implementation of a method, whose compatibility with this evaluation framework is tested (see [paper](https://arxiv.org/abs/2103.16262) for details).
- Other files should be self-explanatory, otherwise, please open an issue.

## Usage

- pix2pix and CycleGAN: run `commands_*.sh` to train and `predict_*.sh` to translate

```bash
# train and test 
cd pytorch-CycleGAN-and-pix2pix/
./commands_{dataset}.sh {fold} {gpu_id} # no {fold} for Histological data

# modality mapping of evaluation data
# {Dataset}_patches -> {Dataset}_patches_fake
./predict_{dataset}.sh
```

- CoMIR: run [`commands_train.sh`](./COMIR/commands_train.sh) and [`predict_all.sh`](./CoMIR/predict_all.sh)

```bash
# train and test (for all datasets)
cd ../CoMIR/
./commands_train.sh

# modality mapping of evaluation data
# {Dataset}_patches -> {Dataset}_patches_fake
./predict_all.sh {gpu_id}
```

- DRIT++: run `commands_*.sh` to train and [`predict_all.sh`](./DRIT/src/predict_all.sh) to translate

```bash
# train and test 
cd ../DRIT/src/
./commands_{dataset}.sh

# modality mapping of evaluation data
# {Dataset}_patches -> {Dataset}_patches_fake
./predict_{dataset}.sh
```

- StarGANv2: 



## Dependencies

[`environment.yml`](./environment.yml) includes the **full** list of packages used to run most of the experiments. Some packages might be unnecessary. And here are some exceptions:

* [SimpleElastix](https://simpleelastix.github.io/) is required to compute the Mutual Information baseline performance. 
* For [CoMIR](https://github.com/MIDA-group/CoMIR), to reduce GPU memory usage, the inference on GPU requires `pytorch>=1.6` to use the [Automatic Mixed Precision package](https://pytorch.org/docs/stable/amp.html), otherwise it uses *half-precision*.


## Citation

Please consider citing our paper and dataset if you find the code useful for your research.
```
@article{luImagetoImageTranslationPanacea2021,
  title = {Is {{Image}}-to-{{Image Translation}} the {{Panacea}} for {{Multimodal Image Registration}}? {{A Comparative Study}}},
  shorttitle = {Is {{Image}}-to-{{Image Translation}} the {{Panacea}} for {{Multimodal Image Registration}}?},
  author = {Lu, Jiahao and {\"O}fverstedt, Johan and Lindblad, Joakim and Sladoje, Nata{\v s}a},
  year = {2021},
  month = mar,
  archiveprefix = {arXiv},
  eprint = {2103.16262},
  eprinttype = {arxiv},
  journal = {arXiv:2103.16262 [cs, eess]},
  primaryclass = {cs, eess}
}

@datasettype{luDatasetsEvaluationMultimodal2021,
  title = {Datasets for {{Evaluation}} of {{Multimodal Image Registration}}},
  author = {Lu, Jiahao and {\"O}fverstedt, Johan and Lindblad, Joakim and Sladoje, Nata{\v s}a},
  year = {2021},
  month = apr,
  publisher = {{Zenodo}},
  doi = {10.5281/zenodo.4587903},
  language = {eng}
}
```

## Code Reference

- [pix2pix & CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [DRIT++](https://github.com/HsinYingLee/DRIT) 
- [StarGANv2](https://github.com/clovaai/stargan-v2)
- [CoMIR](https://github.com/MIDA-group/CoMIR)
- [alpha-AMD](https://github.com/MIDA-group/py_alpha_amd_release)

