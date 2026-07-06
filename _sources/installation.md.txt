# Installation

You can install SMURF directly from its GitHub repository.

## Create conda environment

It is recommended you create a conda environment, especially if you want to use the full version of SMURF (as opposed to the “lite” version).

```bash
conda create -n smurf python=3.10
conda activate smurf
```

### Lite version

To install the lite version of SMURF, run the following command:

```bash
pip install pysmurf
```

### Full version (GPU needed)

To install the full version of SMURF, run the following command:

```bash
pip install "pysmurf[full]"
```

The only difference between the full version and the lite version is that the full version ensures that the required version of PyTorch (and related packages) are correctly installed.

### Developer version

To install the Developer version, run the following command:

```bash
pip install git+https://github.com/The-Mitra-Lab/SMURF.git
```

or

```bash
pip install "git+https://github.com/The-Mitra-Lab/SMURF.git#egg=pysmurf[full]"
```
