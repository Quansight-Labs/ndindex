import setuptools
import versioneer

import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()

from Cython.Build import cythonize
ext_modules = cythonize(["ndindex/*.pyx"],
                        language_level="3",
)

setuptools.setup(
    name="ndindex",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Quansight Labs",
    description="A Python library for manipulating indices of ndarrays.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://quansight-labs.github.io/ndindex/",
    packages=['ndindex', 'ndindex.tests'],
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
    license="MIT",
    # NumPy is only required when using array indices
    extras_require={
        "arrays": "numpy",
    },
    tests_require=[
        'numpy',
        'pytest',
        'hypothesis',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
