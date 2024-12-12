# SMURF: Segmentation and Manifold UnRolling Framework

Welcome to **SMURF**!

SMURF performs soft segmentation of [VisiumHD](https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression) data, facilitating the creation of a `cells*genes` `anndata` object. SMURF uses high-resolution images from VisiumHD for nuclei segmentation and then assigns the transcripts recovered from each capture 'spot' to a nearby cell. See xxx for further details.

SMURF was developed and maintained by [Juanru Guo](https://github.com/JuanruMaryGuo) and [The Mitra Lab](http://genetics.wustl.edu/rmlab/) at Washington University in St. Louis.

If you find a model useful for your research, please cite the following (paper on the way):

```
@misc{smurf,
  author = {Juanru Guo and The Mitra Lab},
  title = {SMURF: Segmentation and Manifold UnRolling Framework},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/The-Mitra-Lab/SMURF}},
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
    :link: https://github.com/rhagenson/SMURF

    Ask questions, report bugs, and contribute to *Pycallingcards* at our GitHub repository.
```

```{toctree}
:hidden: true
:maxdepth: 3
:titlesonly: true

installation
tutorials/index
user_guide/index
api/index
release_notes/index
references
GitHub <https://github.com/rhagenson/SMURF>
```
