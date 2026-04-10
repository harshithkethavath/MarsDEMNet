---
license: cc0-1.0
task_categories:
- depth-estimation
- image-to-image
language:
- en
pretty_name: MCTED
size_categories:
- 10K<n<100K
tags:
- Mars
- remote-sensing
- image
- depth-estimation
- DEM
---

# MCTED - Mars CTX Terrain-Elevation Dataset

[**Dataset repository**](https://github.com/ESA/MCTED) | [**arXiv article**](https://arxiv.org/abs/2509.08027)

<img src="./.images/dataset_samples.png" alt="Dataset samples" style="width:100%;"/>

## Overview
**MCTED** is a machine-learning-ready dataset of optical images of the surface of Mars, paired with their corresponding digital elevation models.
It was created using an extensive repository of orthoimage-DEM pairs with the NASA Ames Stereo Pipeline using the Mars Reconneissance Orbiter's CTX instrument imagery by
[Day et al. 2023](https://github.com/GALE-Lab/Mars_DEMs). We process the samples from the repository using a developed pipeline aimed at eliminating elevation artifacts, 
imputing missing data points and sample selection. The dataset is provided in the form of 518x518 patches.

This dataset is fully open-source, with all data and code used for it's generation available publicly.

## Dataset contents
The dataset contains in total **80,898** samples, divided into two splits:
|Training|Validation|
|---|---|
|65,090|15,808|

Each sample consists of 4 different files:
|Type|Description|
|---|---|
|***optical.png***|The monochromatic optical image patch. Despite being monochromatic, the image still has 3 channels, with all channels being the same|
|***elevation.tiff***|The elevation data patch in meters w.r.t. the Martian datum|
|***deviation_mask.png***|Binary mask with locations that were identified as elevation artifacts during dataset generation and were replaced with interpolated values|
|***initial_nan_mask.png***|Binary mask with locations that contained missing values in the Day et al. data samples and were imputed during processing|

### Sample naming
Each sample follows the following naming convention:

<img src="./.images/sample_naming.png" alt="Naming convention of each sample" style="border-radius:15px; width: 70%;">

## Data source

The dataset has been generated using a orthoimage-DEM pair repository generated from MROs CTX imagery using the [NASA Ames Stereo Pipeline](https://github.com/NeoGeographyToolkit/StereoPipeline) 
by [Day et al. 2023](https://faculty.epss.ucla.edu/~mday/index.php/mars-dems/). We pass the samples through an extensive processing and selection pipeline, using approximately **47%** of the available data.

<img src="./.images/sankey_processing.png" alt="Sankey diagram of processed samples" style="width:50%;"/>

## Typical usage
The simplest way to use MCTED is by using the `load_dataset` function from HuggingFace's `datasets` python package:
```python
from datasets import load_dataset

# Download and load the dataset 
mcted = load_dataset("ESA-Datalabs/MCTED", num_proc=8)
```

## Example of accessing sample data
```python
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

mcted = load_dataset("ESA-Datalabs/MCTED", num_proc=8)

# Load one sample from the validation split
sample = mcted["validation"][0]

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(sample["optical.png"])
plt.title("Optical image")

plt.subplot(1, 4, 2)
plt.imshow(np.array(sample["elevation.tif"]), cmap="terrain")
plt.title("DEM")

plt.subplot(1, 4, 3)
plt.imshow(sample["deviation_mask.png"], cmap="gray")
plt.title("Elevation outlier mask")

plt.subplot(1, 4, 4)
plt.imshow(sample["initial_nan_mask.png"], cmap="gray")
plt.title("Initial invalid values mask")
```

## Citation
```bibtex
@misc{osadnik2025,
      title={MCTED: A Machine-Learning-Ready Dataset for Digital Elevation Model Generation From Mars Imagery}, 
      author={Rafał Osadnik and Pablo Gómez and Eleni Bohacek and Rickbir Bahia},
      year={2025},
      eprint={2509.08027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.08027}, 
}
```