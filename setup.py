import setuptools
import _versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ndindex",
    version=_versioneer.get_version(),
    cmdclass=_versioneer.get_cmdclass(),
    author="Quansight",
    description="A Python library for manipulating indices of ndarrays.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quansight/ndindex",
    packages=['ndindex', 'ndindex.tests'],
    license="MIT",
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
