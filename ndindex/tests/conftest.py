# Make sure the doctests in the docs can import the module
import sys
sys.path.insert(0, '.')
from distutils.version import LooseVersion

# Make sure a new enough version of NumPy is installed for the tests
import numpy
if LooseVersion(numpy.__version__) < '1.20':
    raise RuntimeError("NumPy 1.20 (development version) or greater is required to run the ndindex tests")
