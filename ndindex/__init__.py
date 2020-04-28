__all__ = []

from .ndindex import ndindex

__all__ += ['ndindex']

from .slice import Slice

__all__ += ['Slice']

from .integer import Integer

__all__ += ['Integer']

from .tuple import Tuple

__all__ += ['Tuple']

from .ellipsis import ellipsis

__all__ += ['ellipsis']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
