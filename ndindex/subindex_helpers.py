"""
This file has the main algorithm for Slice.as_subindex(Slice)

Since Integer can use the same algorithm via Slice(i, i+1), and IntegerArray
needs to do this but in a way that only uses array friendly operations, we
need to have this factored out into a separately callable function.

TODO: we could remove the dependency on SymPy if we wanted to, by implementing
the special cases for ilcm(a, b) and the Chinese Remainder Theorem for 2
equations. It wouldn't be too bad (it just requires the extended gcd
algorithm), but depending on SymPy also isn't a big deal for the time being.

"""
from numpy import broadcast_arrays, amin, amax, where

def _crt(m1, m2, v1, v2):
    """
    Chinese Remainder Theorem

    Returns x such that x = v1 (mod m1) and x = v2 (mod m2), or None if no
    such solution exists.

    """
    # Avoid calling sympy_crt in the cases where the inputs would be arrays.
    if m1 == 1:
        return v2 % m2
    if m2 == 1:
        return v1 % m1

    # Only import SymPy when necessary
    from sympy.ntheory.modular import crt as sympy_crt

    res = sympy_crt([m1, m2], [v1, v2])
    if res is None:
        return res
    # Make sure the result isn't a gmpy2.mpz
    return int(res[0])

def _ilcm(a, b):
    # Avoid calling sympy_ilcm in the cases where the inputs would be arrays.
    if a == 1:
        return b
    if b == 1:
        return a

    # Only import SymPy when necessary
    from sympy import ilcm as sympy_ilcm

    return sympy_ilcm(a, b)

def ceiling(a, b):
    """
    Returns ceil(a/b)
    """
    return -(-a//b)

def _max(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    return amax(broadcast_arrays(a, b), axis=0)

def _min(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return amin(broadcast_arrays(a, b), axis=0)

def _smallest(x, a, m):
    """
    Gives the smallest integer >= x that equals a (mod m)

    Assumes x >= 0, m >= 1, and 0 <= a < m.
    """
    n = ceiling(x - a, m)
    return a + n*m

def subindex_slice(s_start, s_stop, s_step, i_start, i_stop, i_step):
    """
    Computes s.as_subindex(i) for slices s and i in a way that is (mostly)
    compatible with NumPy arrays.

    Returns (start, stop, step).

    """
    # Chinese Remainder Theorem. We are looking for a solution to
    #
    # x = s.start (mod s.step)
    # x = index.start (mod index.step)
    #
    # If crt() returns None, then there are no solutions (the slices do
    # not overlap).
    common = _crt(s_step, i_step, s_start, i_start)

    if common is None:
        return (0, 0, 1)
    lcm = _ilcm(s_step, i_step)
    start = _max(s_start, i_start)

    # Get the smallest lcm multiple of common that is >= start
    start = _smallest(start, common, lcm)
    # Finally, we need to shift start so that it is relative to index
    start = (start - i_start)//i_step

    stop = ceiling((_min(s_stop, i_stop) - i_start), i_step)
    stop = where(stop < 0, 0, stop)

    step = lcm//i_step # = s_step//igcd(s_step, i_step)

    return (start, stop, step)
