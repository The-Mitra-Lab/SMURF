from setuptools import setup, find_packages


setup(
    name='spatialcell',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'scanpy',
        'scipy',  
        'matplotlib',  
        'numba', 
        'scikit-learn', 
        'Pillow',
        'anndata' 
    ],
    extras_require={
        'advanced': ['torch','py3nvml'] 
    },
    python_requires='>=3.8',
)
