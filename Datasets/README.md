[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5557568.svg)](https://doi.org/10.5281/zenodo.5557568)

# Datasets

Open-access evaluation data: [Datasets for Evaluation of Multimodal Image Registration](https://zenodo.org/record/5557568)



## Description

### Zurich data

The Zurich dataset is divided into 3 sub-groups by IDs: \{7, 9, 20, 3, 15, 18\}, \{10, 1, 13, 4, 11, 6, 16\}, \{14, 8, 17, 5, 19, 12, 2\}. Since the images vary in size, each image is subdivided into the maximal number of equal-sized non-overlapping regions such that each region can contain exactly one 300x300 px image patch. Then one 300x300 px image patch is extracted from the centre of each region. The particular 3-folded grouping followed by splitting leads to that each evaluation fold contains 72 test samples.

- Modality A: Near-Infrared (NIR)
- Modality B: three colour channels (in B-G-R order)

### Cytological data

The Cytological data contains images from 3 different cell lines; all images from one cell line is treated as one fold in 3-folded cross-validation.
Each image in the dataset is subdivided from 600x600 px into 2x2 patches of size 300x300 px, so that there are 420 test samples in each evaluation fold.

- Modality A: Fluorescence Images
- Modality B: Quantitative Phase Images (QPI)

### Histological dataset

For the Histological data, to avoid too easy registration relying on the circular border of the TMA cores, the evaluation images are created by cutting  834x834 px patches from the centres of the original 134 TMA image pairs.

- Modality A: Second Harmonic Generation (SHG)
- Modality B: Bright-Field (BF)


The evaluation set created from the above [three publicly available 2D datasets](#data-sources) consists of images undergone 4 levels of (rigid) transformations of increasing size of displacement. The level of transformations is determined by the size of the rotation angle θ and the displacement tx & ty, detailed in the table. Each image sample is transformed exactly once at each transformation level so that all levels have the same number of samples. 

<table align="center">
  <caption style="text-align:center">Transformation levels in evaluation sets</caption>
<thead>
  <tr>
    <th rowspan="2" style="text-align:center">Transformation level</th>
    <th style="text-align:center">Range of rotation&nbsp;&nbsp;θ [deg]</th>
    <th colspan="2" style="text-align:center">Range of translation&nbsp;&nbsp;tx &amp; ty [px]</th>
  </tr>
  <tr>
    <th style="text-align:center">All datasets</th>
    <th style="text-align:center">Zurich &amp; Cytological data </th>
    <th style="text-align:center">Histological data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align:center">1</td>
    <td style="text-align:center">[0, 5) ∪ [-5, 0)</td>
    <td style="text-align:center">[0, 7) ∪ [-7, 0)</td>
    <td style="text-align:center">[0, 20) ∪ [-20, 0)</td>
  </tr>
  <tr>
    <td style="text-align:center">2</td>
    <td style="text-align:center">[5, 10) ∪ [-10, -5)</td>
    <td style="text-align:center">[7, 14) ∪ [-14, -7)</td>
    <td style="text-align:center">[20, 40) ∪ [-40, -20)</td>
  </tr>
  <tr>
    <td style="text-align:center">3</td>
    <td style="text-align:center">[10, 15) ∪ [-15, -10)</td>
    <td style="text-align:center">[14, 21) ∪ [-21, -14)</td>
    <td style="text-align:center">[40, 60) ∪ [-60, -40)</td>
  </tr>
  <tr>
    <td style="text-align:center">4</td>
    <td style="text-align:center">[15, 20) ∪ [-20, -15)</td>
    <td style="text-align:center">[21, 28) ∪ [-28, -21)</td>
    <td style="text-align:center">[60, 80) ∪ [-80, -60)</td>
  </tr>
</tbody>
</table>

------

### Radiological dataset

The Radiological dataset is divided into 3 sub-groups by patient IDs: \{109, 106, 003, 006\}, \{108, 105, 007, 001\}, \{107, 102, 005, 009\}. Since the Radiological dataset is non-isotropic (and also of varying resolution), it is resampled using B-spline interpolation to 1 mm<sup>3</sup> cubic voxels, taking explicit care to not resample twice; displaced volumes are transformed and resampled in one step. 

- Modality A: T1-weighted MRI
- Modality B: T2-weighted MRI

(Run [`make_rire_patches.py`](../utils/make_rire_patches.py) to generate the sub-volumes.)

Reference sub-volumes of size 210x210x70 voxels are cropped directly from centres of the (non-displaced) resampled volumes. 
Similarly as for the aforementioned 2D datasets, random (uniformly-distributed) transformations are composed of rotations θx, θy ∈ [-4, 4] degrees around the x- and y-axes, rotation θz ∈ [-20, 20] degrees around the z-axis, translations tx, ty ∈ [-19.6, 19.6]  voxels in x and y directions and translation tz ∈ [-6.5, 6.5]  voxels in z direction. 
40 rigid transformations of increasing sizes of displacement are applied to each volume. Transformed sub-volumes, of size 210x210x70 voxels, are cropped from centres of the transformed and resampled volumes. 

------

In total, it contains 864 image pairs created from Zurich dataset, 5040 image pairs created from the Cytological dataset, 536 image pairs created from the Histological dataset, and metadata with scripts to create the 480 volume pairs from the Radiological dataset. Each image pair consists of a reference patch I<sub>Ref</sub> and its corresponding initial transformed patch I<sub>Init</sub> in both modalities (see [paper](https://arxiv.org/abs/2103.16262) for examples), along with the ground-truth transformation parameters to recover it.



## Metadata

In the `*.zip` files, each row in `{Zurich,Balvan}_patches/fold[1-3]/patch_tlevel[1-4]/info_test.csv` or `Eliceiri_patches/patch_tlevel[1-4]/info_test.csv` provides the information of an image pair as follow:

- Filename: identifier(ID) of the image pair
- X1_Ref: x-coordinate of upper left corner of reference patch I<sub>Ref</sub>
- Y1_Ref: y-coordinate of upper left corner of reference patch I<sub>Ref</sub>
- X2_Ref: x-coordinate of lower left corner of reference patch I<sub>Ref</sub>
- Y2_Ref: y-coordinate of lower left corner of reference patch I<sub>Ref</sub>
- X3_Ref: x-coordinate of lower right corner of reference patch I<sub>Ref</sub>
- Y3_Ref: y-coordinate of lower right corner of reference patch I<sub>Ref</sub>
- X4_Ref: x-coordinate of upper right corner of reference patch I<sub>Ref</sub>
- Y4_Ref: y-coordinate of upper right corner of reference patch I<sub>Ref</sub>
- X1_Trans: x-coordinate of upper left corner of transformed patch I<sub>Init</sub>
- Y1_Trans: y-coordinate of upper left corner of transformed patch I<sub>Init</sub>
- X2_Trans: x-coordinate of lower left corner of transformed patch I<sub>Init</sub>
- Y2_Trans: y-coordinate of lower left corner of transformed patch I<sub>Init</sub>
- X3_Trans: x-coordinate of lower right corner of transformed patch I<sub>Init</sub>
- Y3_Trans: y-coordinate of lower right corner of transformed patch I<sub>Init</sub>
- X4_Trans: x-coordinate of upper right corner of transformed patch I<sub>Init</sub>
- Y4_Trans: y-coordinate of upper right corner of transformed patch I<sub>Init</sub>
- Displacement: mean Euclidean distance between reference corner points and transformed corner points 
- RelativeDisplacement: the ratio of displacement to the width/height of image patch
- Tx: randomly generated translation in x direction to synthesise the transformed patch I<sub>Init</sub>
- Ty: randomly generated translation in y direction to synthesise the transformed patch I<sub>Init</sub>
- AngleDegree: randomly generated rotation in degrees to synthesise the transformed patch I<sub>Init</sub>
- AngleRad: randomly generated rotation in radian to synthesise the transformed patch I<sub>Init</sub>

In addition, each row in `RIRE_patches/fold[1-3]/patch_tlevel[1-4]/info_test.csv` has following columns:

- Z1_Ref: z-coordinate of upper left corner of reference patch I<sub>Ref</sub>
- Z2_Ref: z-coordinate of lower left corner of reference patch I<sub>Ref</sub>
- Z3_Ref: z-coordinate of lower right corner of reference patch I<sub>Ref</sub>
- Z4_Ref: z-coordinate of upper right corner of reference patch I<sub>Ref</sub>
- Z1_Trans: z-coordinate of upper left corner of transformed patch I<sub>Init</sub>
- Z2_Trans: z-coordinate of lower left corner of transformed patch I<sub>Init</sub>
- Z3_Trans: z-coordinate of lower right corner of transformed patch I<sub>Init</sub>
- Z4_Trans: z-coordinate of upper right corner of transformed patch I<sub>Init</sub>
- (...and similarly, coordinates of the 5th-8th corners)
- Tz: randomly generated translation in z direction to synthesise the transformed patch I<sub>Init</sub>
- AngleDegreeX: randomly generated rotation around X-axis in degrees to synthesise the transformed patch I<sub>Init</sub>
- AngleRadX: randomly generated rotation around X-axis in radian to synthesise the transformed patch I<sub>Init</sub>
- AngleDegreeY: randomly generated rotation around Y-axis in degrees to synthesise the transformed patch I<sub>Init</sub>
- AngleRadY: randomly generated rotation around Y-axis in radian to synthesise the transformed patch I<sub>Init</sub>
- AngleDegreeZ: randomly generated rotation around Z-axis in degrees to synthesise the transformed patch I<sub>Init</sub>
- AngleRadZ: randomly generated rotation around Z-axis in radian to synthesise the transformed patch I<sub>Init</sub>


## Naming convention

### Zurich Data

```
zh{ID}_{iRow}_{iCol}_{ReferenceOrTransformed}.png
```

Example: `zh5_03_02_R.png` indicates the **Reference** patch of the **3rd row** and **2nd column** cut from the image with ID `zh5`.

### Cytological data

```
{{cellline}_{treatment}_{fieldofview}_{iFrame}}_{iRow}_{iCol}_{ReferenceOrTransformed}.png
```

Example: `PNT1A_do_1_f15_02_01_T.png` indicates the **Transformed** patch of the **2nd row** and **1st column** cut from the image with ID `PNT1A_do_1_f15`.

### Histological data

```
{ID}_{ReferenceOrTransformed}.tif
```

Example: `1B_A4_T.tif` indicates the **Transformed** patch cut from the image with ID `1B_A4`.

### Radiological data

```
patient_{ID}_{iTransform}_T.mhd
patient_{ID}_R.mhd
```

Example: `patient_003_8_T.mhd` indicates the sub-volume **Transformed** with the **8th random transformation** cut from the volume with patient ID `003`; `patient_003_R.mhd` indicates the **Reference** sub-volume the volume with patient ID `003`.




## Instructions for customising evaluation data

If you want to generate more evaluation data with different settings, please modify the scripts in [`../utils/`](../utils/) and follow the instructions below. (Run scripts under the root directory. Paths might need changing.)

### Zurich Data

```bash
# raw data cleaning
# Zurich_dataset_v1.0 -> Zurich
python ./utils/extract_zurich_modalites.py

# prepare training data for I2I translation
# Zurich -> Zurich_temp (3 folds)
python ./utils/prepare_Zurich.py

# make evaluation patches
# Zurich -> Zurich_tiles -> Zurich_patches (3 folds)
python ./utils/make_zurich_patches.py
```

### Cytological data

```bash
# raw data cleaning
# -> Balvan
python ./utils/extract_balvan_modalites.py

# prepare training data for I2I translation
# Balvan -> Balvan_temp (3 folds)
python ./utils/prepare_Balvan.py

# make evaluation patches
# Balvan -> Balvan_1to4tiles -> Balvan_patches (3 folds)
python ./utils/make_balvan_patches.py
```

### Histological data

```bash
# prepare training data for I2I translation
# -> Eliceiri_temp
python ./utils/prepare_Eliceiri.py

# make evaluation patches
# -> Eliceiri_patches
python ./utils/make_eliceiri_patches.py
```

### Radiological data

```bash
# prepare training data for I2I translation
# RIRE -> RIRE_temp
python ./utils/prepare_RIRE.py

# make evaluation patches
# RIRE -> RIRE_patches
python ./utils/make_rire_patches.py

# stack I2I translated slices to volumes and generate cutouts
# RIRE_slices_fake -> RIRE_patches_fake
python ./utils/make_rire_patches_fake.py
```



## Citation

Please consider citing our paper if you find this dataset helpful.
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

## Data sources

- Zurich data: [Near-Infrared and RGB images](https://sites.google.com/site/michelevolpiresearch/data/zurich-dataset)
- Cytological data: [Quantitative Phase Images (QPI)](https://zenodo.org/record/2601562) and [Fluorescence Images](https://zenodo.org/record/4531900)
- Histological data: [SHG and BF images](https://zenodo.org/record/4550300)
- Radiological data: [RIRE Dataset](https://www.insight-journal.org/rire/download_data.php)

