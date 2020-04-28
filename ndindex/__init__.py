from .ndindex import (ndindex, Slice, Integer, Tuple, ellipsis)

__all__ = ['ndindex', 'Slice', 'Integer', 'Tuple', 'ellipsis']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
