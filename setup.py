import pathlib

from setuptools import find_packages, setup

# Get the current directory
HERE = pathlib.Path(__file__).parent

# Read the README file
README = (HERE / "README.md").read_text()

setup(
    name="pysmurf",
    version="1.0.2",
    author="Juanru Guo",
    author_email="g.juanru@wustl.edu",
    description="Segmentation and Manifold UnRolling Framework (SMURF)",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.26.4",
        "pandas >= 1.5.3",
        "tqdm",
        "scanpy >= 1.10.0",
        "scipy >= 1.12.0",
        "matplotlib >= 3.8.3",
        "numba >=  0.59.0",
        "scikit-learn >= 1.4.1",
        "Pillow >= 10.2.0",
        "anndata >= 0.10.6",
        "h5py >= 3.10.0",
        "pyarrow >= 16.1.0",
        "igraph >= 0.11.5",
    ],
    extras_require={"full": ["torch  >= 2.2.1", "py3nvml >= 0.2.7"]},
    python_requires=">=3.8",
)
