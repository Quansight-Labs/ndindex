from .ndindex import (ndindex, Slice, Integer, Tuple)

__all__ = ['ndindex', 'Slice', 'Integer', 'Tuple']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
