import os
from setuptools import setup, Extension
import versioneer

import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()

from Cython.Build import cythonize

CYTHON_COVERAGE = os.environ.get("CYTHON_COVERAGE", False)
define_macros = []
compiler_directives = {}
if CYTHON_COVERAGE:
    define_macros.append(("CYTHON_TRACE_NOGIL", "1"))
    compiler_directives["linetrace"] = True

ext_modules = cythonize([
    Extension(
        "ndindex._slice", ["ndindex/_slice.pyx"],
        define_macros=define_macros,
    ),
    Extension(
        "ndindex._tuple", ["ndindex/_tuple.pyx"],
        define_macros=define_macros,
    )],
    language_level="3",
    compiler_directives=compiler_directives,
)

setup(
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
