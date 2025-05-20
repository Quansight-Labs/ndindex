import importlib
import os
from setuptools import setup, Extension, Command
import shutil
import glob

from Cython.Build import cythonize
from Cython.Compiler.Version import version as cython_version
from packaging.version import Version


def import_versioneer():
    # Needed because the non-legacy build backend doesn't run in the local
    # dir, and hence an absolute import of `./versioneer.py` won't work.
    versioneer_path = os.path.join(os.path.dirname(__file__), 'versioneer.py')
    spec = importlib.util.spec_from_file_location(
        'versioneer', versioneer_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


versioneer = import_versioneer()


with open("README.md", "r") as fh:
    long_description = fh.read()

CYTHON_COVERAGE = os.environ.get("CYTHON_COVERAGE", False)
define_macros = []
compiler_directives = dict()
compiler_tenv = dict()
if CYTHON_COVERAGE:
    print("CYTHON_COVERAGE is set. Enabling Cython coverage support.")
    define_macros.append(("CYTHON_TRACE_NOGIL", "1"))
    compiler_directives["linetrace"] = True
if Version(cython_version) >= Version("3.1.0"):
    compiler_directives["freethreading_compatible"] = True
    compiler_tenv["CYTHON_FREE_THREADING"] = True
else:
    compiler_tenv["CYTHON_FREE_THREADING"] = False
ext_modules = cythonize([
    Extension(
        "ndindex._slice", ["ndindex/_slice.pyx"],
        define_macros=define_macros,
    ),
    Extension(
        "ndindex._tuple", ["ndindex/_tuple.pyx"],
        define_macros=define_macros,
    )
],
    language_level="3",
    compiler_directives=compiler_directives,
    compile_time_env=compiler_tenv,
)

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        paths = [
            './build',
            './dist',
            './*.egg-info',
            './**/*.so',
            './**/*.c',
            './**/*.cpp',
        ]
        for path in paths:
            for item in glob.glob(path, recursive=True):
                print(f"Removing {item}")
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)

setup(
    name="ndindex",
    version=versioneer.get_version(),
    cmdclass={
        **versioneer.get_cmdclass(),
        'clean': CleanCommand
    },
    author="Quansight Labs",
    description="A Python library for manipulating indices of ndarrays.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://quansight-labs.github.io/ndindex/",
    packages=['ndindex', 'ndindex.tests'],
    ext_modules=ext_modules,
    license="MIT",
    # NumPy is only required when using array indices
    extras_require={
        "arrays": "numpy",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
