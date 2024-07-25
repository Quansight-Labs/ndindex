from math import gcd

cpdef (long long, long long, long long) gcdex(long a, long b):
    cdef long long x = 0, y = 1, u = 1, v = 0
    cdef long long q, r, m, n
    cdef int x_sign = 1, y_sign = 1

    if a == 0 and b == 0:
        return 0, 1, 0

    if a == 0:
        return 0, 1 if b > 0 else -1, abs(b)
    if b == 0:
        return 1 if a > 0 else -1, 0, abs(a)

    if a < 0:
        a, x_sign = -a, -1
    if b < 0:
        b, y_sign = -b, -1

    while a:
        q = b // a
        r = b % a
        m = x - u * q
        n = y - v * q
        b, a, x, y, u, v = a, r, u, v, m, n

    return x * x_sign, y * y_sign, b

cdef long long gcdex_(long long a, long long b, long long *x, long long *y):
    if not a and not b:
        x[0], y[0] = 0, 1
        return 0
    if not a:
        x[0], y[0] = 0, b // abs(b)
        return abs(b)
    if not b:
        x[0], y[0] = a // abs(a), 0
        return abs(a)

    x_sign, y_sign = 1, 1
    if a < 0:
        a, x_sign = -a, -1
    if b < 0:
        b, y_sign = -b, -1

    x[0], y[0], r, s = 1, 0, 0, 1

    while b:
        c, q = a % b, a // b
        a, b, r, s, x[0], y[0] = b, c, x[0] - q * r, y[0] - q * s, r, s

    x[0] *= x_sign
    y[0] *= y_sign

    return a

cdef int _combine(long long *a1, long long *m1, long long a2, long long m2):
    cdef long long inv_a, temp
    x, y, z = m1[0], a2 - a1[0], m2
    g = gcd(x, y, z)
    x, y, z = x//g, y//g, z//g
    if x != 1:
        g = gcdex_(x, z, &inv_a, &temp)
        if g != 1:
            return -1
        y *= inv_a
    a1[0], m1[0] = a1[0] + m1[0]*y, m1[0]*z
    return 0

cdef int _solve_congruence(list[long long] V, list[long long] M, long long *n):
    cdef long long k = 1
    n[0] = 0
    for v, m in zip(V, M):
        if _combine(n, &k, v, m) == -1:
            break
    else:
        n[0] = n[0] % k
        return 0
    return -1

cpdef solve_congruence(list[long long] V, list[long long] M):
    cdef long long n
    if _solve_congruence(V, M, &n) == -1:
        return None
    return n

cpdef long long ilcm(long a, long b):
    if a == 0 or b == 0:
        return 0
    return a // gcd(a, b) * b


cdef long long _crt2(long long m1, long long m2, long long v1, long long v2,
                    long long *n):
    p = m1*m2
    v = 0

    e = p // m1
    s, _, _ = gcdex(e, 1)
    v += e*(v1*s % m1)

    e = p // m2
    s, _, _ = gcdex(e, m2)
    v += e*(v2*s % m2)
    n[0] = v % p

    if v1 % m1 != n[0] % m1:
        result = _solve_congruence([v1, v2], [m1, m2], n)
        return result

    if v2 % m2 != n[0] % m2:
        result = _solve_congruence([v1, v2], [m1, m2], n)
        return result

    return 0
