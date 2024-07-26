cimport cython
import numpy as np
cimport numpy as np

from ._crt_cython cimport _crt2, ilcm

cdef inline ceiling(a, b):
    """
    Returns ceil(a/b)
    """
    return -(-a//b)


cdef inline _smallest(x, a, m):
    """
    Gives the smallest integer >= x that equals a (mod m)

    Assumes x >= 0, m >= 1, and 0 <= a < m.
    """
    n = ceiling(x - a, m)
    return a + n*m

cdef inline _where(cond, x, y):
    """
    Returns x if cond is True, y otherwise
    """
    return x if cond else y

@cython.ufunc
cdef (long, long, long) subindex_slice(
    long s_start,
    long s_stop,
    long s_step,
    long i_start,
    long i_stop,
    long i_step):

    cdef long long common
    # Chinese Remainder Theorem. We are looking for a solution to
    #
    # x = s.start (mod s.step)
    # x = index.start (mod index.step)
    #
    # If crt() returns None, then there are no solutions (the slices do
    # not overlap).
    if _crt2(s_step, i_step, s_start, i_start, &common) == -1:
        return (0, 0, 1)
    lcm = ilcm(s_step, i_step)
    start = max(s_start, i_start)

    # Get the smallest lcm multiple of common that is >= start
    start = _smallest(start, common, lcm)
    # Finally, we need to shift start so that it is relative to index
    start = (start - i_start)//i_step

    stop = ceiling((min(s_stop, i_stop) - i_start), i_step)
    stop = _where(stop < 0, 0, stop)

    step = lcm//i_step # = s_step//igcd(s_step, i_step)

    return (start, stop, step)
