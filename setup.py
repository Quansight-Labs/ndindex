import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ndindex",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Quansight",
    description="A Python library for manipulating indices of ndarrays.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://quansight.github.io/ndindex/",
    packages=['ndindex', 'ndindex.tests'],
    license="MIT",
    install_requires=[
        "numpy",
    ],
    tests_require=[
        'pytest',
        'hypothesis',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
