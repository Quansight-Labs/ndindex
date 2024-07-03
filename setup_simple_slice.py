from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "simple_slice_cython",
        ["simple_slice_cython.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="simple_slice_extension",
    ext_modules=cythonize(extensions),
    install_requires=['numpy'],
)
