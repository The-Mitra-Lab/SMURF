# Segmentation and Manifold UnRolling Framework (SMURF)

Welcome to the SMURF repository!

We introduce SMURF (Segmentation and Manifold UnRolling Framework) to perform soft segmentation of [VisiumHD data](https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression), facilitating the creation of a cells*genes anndata object. SMURF uses high-resolution images from VisiumHD for nuclei segmentation and then assigns the transcripts recovered from each capture ‘spot’ to a nearby cell. See [Tutorial](https://the-mitra-lab.github.io/SMURF/) and [preprint paper](https://www.biorxiv.org/content/10.1101/2025.05.28.656357v1) for further details.


## Contents

- [Segmentation and Manifold UnRolling Framework (SMURF)](#segmentation-and-manifold-unrolling-framework-smurf)
  - [Contents](#contents)
- [Installation ](#installation-)
  - [Create conda ](#create-conda-)
  - [Lite version ](#lite-version-)
  - [Full version (GPU needed)](#full-version-gpu-needed)
  - [Developer version](#developer-version)
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
pip install "git+https://github.com/The-Mitra-Lab/SMURF.git#egg=pysmurf[full]"
```

# Citation  <a name="Citation"></a>

```latex

@article{guo2025smurf,
  title={SMURF Reconstructs Single-Cells from Visium HD Data to Reveal Zonation of Transcriptional Programs in the Intestine},
  author={Guo, Juanru and Sarafinovska, Simona and Hagenson, Ryan and Valentine, Mark and Dougherty, Joseph and Mitra, Robi David and Muegge, Brian D},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}


```
