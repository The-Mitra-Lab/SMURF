# SMURF: Segmentation and Manifold UnRolling Framework

Welcome to **SMURF**!

SMURF performs soft segmentation of [VisiumHD](https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression) data, facilitating the creation of a `cells*genes` `anndata` object. SMURF uses high-resolution images from VisiumHD for nuclei segmentation and then assigns the transcripts recovered from each capture 'spot' to a nearby cell.

If you find a model useful for your research, please cite the following:

```
@article{guo2025smurf,
  author  = {Guo, Juanru and Sarafinovska, Simona and Hagenson, Ryan A. and Valentine, Mark C. and Dougherty, Joseph D. and Mitra, Robi D. and Muegge, Brian D.},
  title   = {SMURF Reconstructs Single-Cells from Visium HD Data to Reveal Zonation of Transcriptional Programs in the Intestine},
  year    = {2025},
  journal = {bioRxiv},
}
```

```{eval-rst}
.. card:: Installation :octicon:`plug;1em;`
    :link: installation
    :link-type: doc

    Click here to view a brief *SMURF* installation guide and prerequisites.
```

```{eval-rst}
.. card:: Tutorials :octicon:`play;1em;`
    :link: tutorials/index
    :link-type: doc

    End-to-end tutorials showcasing key features in the package.
```

```{eval-rst}
.. card:: User guide :octicon:`info;1em;`
    :link: user_guide/index
    :link-type: doc

    User guide provides some detail information of *SMURF*.
```

```{eval-rst}
.. card:: API reference :octicon:`book;1em;`
    :link: api/index
    :link-type: doc

    Detailed descriptions of *SMURF* API and internals.
```

```{eval-rst}
.. card:: GitHub :octicon:`mark-github;1em;`
    :link: https://github.com/The-Mitra-Lab/SMURF

    Ask questions, report bugs, and contribute to *SMURF* at our GitHub repository.
```

```{toctree}
:hidden: true
:maxdepth: 3
:titlesonly: true

installation
tutorials/index
Data_structure/index
api/index
release_notes/index
references
GitHub <https://github.com/The-Mitra-Lab/SMURF>
```
