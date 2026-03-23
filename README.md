# Segmentation and Manifold Unrolling Framework (SMURF)

Welcome to the SMURF repository!

SMURF (Segmentation and Manifold Unrolling Framework) is a framework for high-resolution spatial transcriptomics that supports both soft segmentation / transcript assignment and tissue unrolling / flattening. In Visium HD examples, SMURF uses high-resolution histology images for nuclei segmentation and assigns transcripts from each capture spot to nearby cells, enabling reconstruction of a cell-by-gene AnnData object. SMURF can also map complex tissue architectures into Cartesian coordinates for downstream spatial analysis. See the [tutorial](https://the-mitra-lab.github.io/SMURF/) and [preprint](https://www.biorxiv.org/content/10.1101/2025.05.28.656357v3) for further details.

## Contents

- [Segmentation and Manifold Unrolling Framework (SMURF)](#segmentation-and-manifold-unrolling-framework-smurf)
  - [Contents](#contents)
  - [Installation](#installation)
    - [Create a Conda environment](#create-a-conda-environment)
  - [Choose between SMURF-lite and SMURF-full](#choose-between-smurf-lite-and-smurf-full)
    - [SMURF-lite](#smurf-lite)
    - [SMURF-full](#smurf-full)
    - [Developer install](#developer-install)
  - [Computational resources](#computational-resources)
  - [Tutorials](#tutorials)
  - [Citation](#citation)

## Installation

We recommend creating a fresh Conda environment before installing SMURF.

### Create a Conda environment

```bash
conda create -n smurf python=3.10
conda activate smurf
```

## Choose between SMURF-lite and SMURF-full

SMURF supports two operating modes for transcript assignment in shared spots.

### SMURF-lite

Install with:

```bash
pip install pysmurf
```

SMURF-lite is the non-deep-learning mode. In the shared-spot transcript-assignment step, it uses a spatial-distance-weighted multinomial strategy and can be run on a standard workstation. Use this mode when a GPU is not available.

### SMURF-full

Install with:

```bash
pip install "pysmurf[full]"
```

SMURF-full uses the same overall workflow as SMURF-lite, but adds a deep-learning refinement for shared spots involving cells of the same type. A GPU is strongly recommended for practical runtimes.

### Developer install

Lite:

```bash
pip install "pysmurf @ git+https://github.com/The-Mitra-Lab/SMURF.git"
```

Full:

```bash
pip install "pysmurf[full] @ git+https://github.com/The-Mitra-Lab/SMURF.git"
```

## Computational resources

Runtime and storage requirements depend on dataset size and analysis settings. On our benchmark workstation (Ubuntu 20.04.6 LTS; Intel Core i9-10920X CPU, 251 GiB RAM, NVIDIA GeForce RTX 3090 GPU), the mouse brain dataset required 1.69 h with SMURF-lite and 2.92 h with SMURF-full. Storage requirements depend on the size of the 2 μm outputs, associated histology images, and whether high-resolution visualizations and intermediate files are retained. In our experience, a typical mouse brain analysis can be run with approximately 25 GB of available storage, whereas larger datasets or analyses that retain intermediate files and high-resolution outputs may require on the order of 100 GB.

## Tutorials

The main tutorials are available on the documentation site:

- [SMURF-lite segmentation tutorial](https://the-mitra-lab.github.io/SMURF/)
- [SMURF-full segmentation tutorial](https://the-mitra-lab.github.io/SMURF/)
- [Unrolling tutorial](https://the-mitra-lab.github.io/SMURF/)

These tutorials illustrate the core segmentation and unrolling workflows used in the manuscript.

## Citation

Please cite SMURF as follows:

```latex
@article{guo2025smurf,
  title={SMURF: soft-segmentation for single-cell reconstruction and topological analysis of spatial transcriptomic data},
  author={Guo, Juanru and Sarafinovska, Simona and Hagenson, Ryan A and Valentine, Mark C and Chen, David Y and McCoy, William H and Dougherty, Joseph D and Mitra, Robi D and Muegge, Brian D},
  journal={bioRxiv},
  year={2025}
}
```
