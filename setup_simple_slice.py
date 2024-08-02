from setuptools import setup
from Cython.Build import cythonize
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "simple_slice_pybind11",
        ["simple_slice_pybind11.cpp"],
        cxx_std=11,
    ),
    *cythonize("simple_slice_cython.pyx"),
]

setup(
    name="simple_slice_pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False
)
