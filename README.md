# Segmentation and Manifold UnRolling Framework (SMURF)

Welcome to the SMURF repository!

We introduce SMURF (Segmentation and Manifold UnRolling Framework) to perform soft segmentation of [VisiumHD data](https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression), facilitating the creation of a cells*genes anndata object. SMURF uses high-resolution images from VisiumHD for nuclei segmentation and then assigns the transcripts recovered from each capture ‘spot’ to a nearby cell. See xxx for further details.


## Contents


- [Segmentation and Manifold UnRolling Framework (SMURF)](#segmentation-and-manifold-unrolling-framework-smurf)
  - [Contents](#contents)
- [Installation ](#installation-)
  - [Create conda ](#create-conda-)
  - [Lite version ](#lite-version-)
  - [Full version (GPU needed)](#full-version-gpu-needed)
  - [Developer version](#developer-version)
- [Tutorial ](#tutorial-)
  - [Lite version. ](#lite-version--1)
  - [Full version (GPU needed). ](#full-version-gpu-needed-)
  - [External reference: Nuclei segmentation (skip this section if you have already segmented your data with your preferred segmentation method). ](#external-reference-nuclei-segmentation-skip-this-section-if-you-have-already-segmented-your-data-with-your-preferred-segmentation-method-)
- [Introduction for important functions and object.  ](#introduction-for-important-functions-and-object--)
  - [`SO` Important Features:](#so-important-features)
  - [`SO` Greate Results:](#so-greate-results)
- [Citation  ](#citation--)


# Installation <a name="installation"></a>

You can install SMURF directly from its GitHub repository. Here’s how you can do it:

## Create conda <a name="Createconda"></a>

It is recommended you create a Conda environment, especially if you want to use the full version of SMURF (as opposed to the “lite” version).

```bash
conda create -n smurf python=3.10
conda activate smurf
```

## Lite version <a name="Lite"></a>

To install the lite version of smurf, run the following command:

```bash
pip install pysmurf
```

## Full version (GPU needed)<a name="Full"></a>

To install the full version of smurf, run the following command:

```bash
pip install "pysmurf[full]"
```

The only difference between the full version and the lite version is that the full version ensures that the required version of PyTorch (and related packages) are correctly installed.

## Developer version

To install the Developer version, run the following command:

```bash
pip install git+https://github.com/The-Mitra-Lab/SMURF.git
```

or

```bash
pip install git+https://github.com/The-Mitra-Lab/SMURF[full].git
```


# Tutorial <a name="Tutorial"></a>

SMURF relies on external packages for nuclei segmentation to create a so (a spatial object datastructure) for further analysis. Researchers are encouraged to use their own preferred segmentation methods. For this tutorial, we use [StarDist](https://qupath.readthedocs.io/en/0.3/docs/advanced/stardist.html), which works well for many datasets. Here we use [mouse brain](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he) data from 10x as example:

## Lite version. <a name="lv"></a>

[Link to view](https://nbviewer.org/github/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_Mousebrian.ipynb)

[Link to file](https://github.com/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_Mousebrian.ipynb)


## Full version (GPU needed). <a name="fv"></a>

[Link to view](https://nbviewer.org/github/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_Mousebrian_full.ipynb)

[Link to file](https://github.com/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_Mousebrian_full.ipynb)

## External reference: Nuclei segmentation (skip this section if you have already segmented your data with your preferred segmentation method). <a name="pre"></a>

[Link to view](https://nbviewer.org/github/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_cell_segmentation.ipynb)

[Link to file](https://github.com/The-Mitra-Lab/SMURF/blob/main/test/Tutorial_cell_segmentation.ipynb)


# Introduction for important functions and object.  <a name="Introduction"></a>

Here are some tips for using `so` so (a spatial object datastructure).

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
