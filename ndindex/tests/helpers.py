from numpy.testing import assert_equal

from pytest import fail

def check_same(a, index, same_exception=True):
    exception = None
    try:
        a_raw = a[index]
    except Exception as e:
        exception = e

    try:
        idx = ndindex(index)
        a_idx = a[idx.raw]
    except Exception as e:
        if not exception:
            fail(f"Raw form does not raise but ndindex form does ({e!r}): {index})")
        if same_exception:
            assert type(e) == type(exception), (e, exception)
            assert e.args == exception.args, (e.args, exception.args)
    else:
        if exception:
            fail(f"ndindex form did not raise but raw form does ({exception!r}): {index})")

    if not exception:
        assert_equal(a_raw, a_idx)
