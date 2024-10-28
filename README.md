# Segmentation and Manifold UnRolling Framework (SMURF)

Welcome to the SMURF repository!

We introduce SMURF (Segmentation and Manifold UnRolling Framework) to leverage soft segmentation with [VisiumHD data](https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression), facilitating the creation of a cells*genes `anndata` object. SMURF uses high-resolution images from VisiumHD for nuclei segmentation.


## Contents


- [Segmentation and Manifold UnRolling Framework (SMURF)](#segmentation-and-manifold-unrolling-framework-smurf)
  - [Contents](#contents)
- [Installation ](#installation-)
  - [Create conda ](#create-conda-)
  - [Lite version ](#lite-version-)
  - [Full version ](#full-version-)
- [Tutorial ](#tutorial-)
  - [PRE: Nuclei segmentation (skip if you had your perferred nuclei segmentation method). ](#pre-nuclei-segmentation-skip-if-you-had-your-perferred-nuclei-segmentation-method-)
  - [Lite version. ](#lite-version--1)
  - [Full version (GPU needed). ](#full-version-gpu-needed-)
- [Introduction for fmportant functions and object.  ](#introduction-for-fmportant-functions-and-object--)
  - [`SO` Important Features:](#so-important-features)
  - [`SO` Greate Results:](#so-greate-results)
  - [`SO` Beautiful Visualization:](#so-beautiful-visualization)
    - [1. `plot_cellcluster_position`](#1-plot_cellcluster_position)
    - [2. `plot_results`](#2-plot_results)
- [Citation  ](#citation--)



# Installation <a name="installation"></a>

You can install SMURF directly from its GitHub repository. Here’s how you can do it:

## Create conda <a name="Createconda"></a>

It is recommended to create a Conda environment, especially for users who want to use the full version:

```bash
conda create -n smurf python=3.10
conda activate smurf
```

## Lite version <a name="Lite"></a>

To install the lite version of smurf, run the following command:
```bash
pip install git+https://github.com/The-Mitra-Lab/SMURF.git
```

## Full version <a name="Full"></a>

To install the full version of smurf, run the following command:
```bash
pip install "git+https://github.com/The-Mitra-Lab/SMURF.git#egg=SMURF[full]"
```


# Tutorial <a name="Tutorial"></a>

SMURF relies on external packages for nuclei segmentation and create an so (spatial object) for further analysis. Researchers are encouraged to use their preferred segmentation methods. But if you are new to it. Please take [StarDist](https://qupath.readthedocs.io/en/0.3/docs/advanced/stardist.html) as a trial. Here we use [mouse brain](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he) data from 10x as example:

## PRE: Nuclei segmentation (skip if you had your perferred nuclei segmentation method). <a name="pre"></a>

[Link to view](https://nbviewer.org/github/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_cell_segmentation.ipynb)

[Link to file](https://github.com/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_cell_segmentation.ipynb)

## Lite version. <a name="lv"></a>

[Link to view](https://nbviewer.org/github/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_Mousebrian.ipynb)

[Link to file](https://github.com/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_Mousebrian.ipynb)


## Full version (GPU needed). <a name="fv"></a>

[Link to view](https://nbviewer.org/github/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_Mousebrian_full.ipynb)

[Link to file](https://github.com/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_Mousebrian_full.ipynb)


# Introduction for fmportant functions and object.  <a name="Introduction"></a>

Here are some tips for using `so`.

## `SO` Important Features:

- **`so.image_temp()`**
  - Provides the image portion covered by spots.

- **`So.segmentation_final`**
  - Gives the nuclei segmentation plot.

- **`So.pixels_cells`**
  - Represents the final results of cells with the same format as `So.segmentation_final`.

- **`So.final_nuclei`**
  - Stores the nuclei * genes `adata` matrix.

## `SO` Greate Results:

- **`adata_sc_final`**
  - `obs`:
    - `cell_cluster`: Denotes the cluster assignment from the iteration.
    - `cos_simularity`: The cosine similarity of the cell's gene expression with the average expression of its cluster.
    - `cell_size`: The number of `2um` spots occupied by this cell.
    - `x` and `y`: The absolute coordinates of the cell, matching the spot locations provided by 10x.
  - `var`:
    - The same as 10x provide for spots*genes array.


##  `SO` Beautiful Visualization:


### 1. `plot_cellcluster_position`

`plot_cellcluster_position(cell_cluster_final, col_num=5)`

This function is often used after `return_celltype_plot` as:

**Example:**

```python
import smurf as su
cell_cluster_final = su.return_celltype_plot(adata_sc, so, cluster_name='leiden')
su.plot_cellcluster_position(cell_cluster_final, col_num=5)
```

It plots the probability of each cell cluster. The first plot will contain all clusters, followed by one plot per cluster.

### 2. `plot_results`

`plot_results(original_image, result_image, transparency=0.6, transparent_background=False, include_label=None, colors=None, dpi=1500, figsize=(20, 20), save=None)`

This function allows you to plot results with the same format as So.segmentation_final mapped onto so.image_temp().

**Parameters:**

- **`original_image`**
  - Normally input `so.image_temp()`.

- **`result_image`**
  - Represents the final results of cells with the same format as `So.segmentation_final`.

- **`transparency`** *(float, optional)*:
  -  The transparency of the result_image. Should be in the range (0, 1].

- **`ransparent_background`** *(bool, optional)*:
  - Whether the zero parts of `result_image` should be transparent.

- **`**include_label`** *(bool, optional)*:
  -  Whether to include a label indicating which color refers to which cluster. Useful for cell cluster results and only valid when the number of unique clusters is ≤ 50.

- **`colors`** *(list of tuples, optional)*:
  -  Provide your own colors in the format like`[(255, 0, 0), (0, 255, 0), ...]`. Will use the default colors if `None`.

- **`dpi`** *(int, optional)*:
  - The resolution of the plot. A higher value may decrease speed; a lower value may reduce quality.

- **`figsize`** *(tuple, optional)*:
  - Size of the figure in inches `(width, height)`. Default is `(20, 20)`.

- **`save`** *(str or None, optional)*:
  - The filename to save the plot. If `False` or `None`, the plot will be shown but not saved.

**Example:**

```python
import smurf as su
su.plot_results(so.image_temp(), so.pixels_cells)
```

# Citation  <a name="Citation"></a>

Paper on the way...

```latex

@misc{smurf,
  author = {Juanru Guo and The Mitra Lab},
  title = {SMURF: Segmentation and Manifold UnRolling Framework},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/The-Mitra-Lab/SMURF}},
}

```
