from pytest import raises

from ..array import ArrayIndex

# Everything else is testsed in the subclasses

def test_ArrayIndex():
    raises(TypeError, lambda: ArrayIndex([]))
