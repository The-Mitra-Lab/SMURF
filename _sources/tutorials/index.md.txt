# Tutorials

SMURF relies on external packages for nuclei segmentation to create a so (a spatial object datastructure) for further analysis. Researchers are encouraged to use their own preferred segmentation methods. For this tutorial, we use [StarDist](https://qupath.readthedocs.io/en/0.3/docs/advanced/stardist.html), which works well for many datasets. Here we use [mouse brain](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he) data from 10x as example:

## Lite version

```{toctree}
:maxdepth: 1

notebooks/Tutorial_Mousebrian
```

## Full version (GPU needed)

```{toctree}
:maxdepth: 1

notebooks/Tutorial_Mousebrian_full
```

## External reference: Nuclei segmentation (skip this section if you have already segmented your data with your preferred segmentation method)

```{toctree}
:maxdepth: 1

notebooks/Tutorial_cell_segmentation
```


## Unrolling Example

```{toctree}
:maxdepth: 1

notebooks/unroll_example
```
