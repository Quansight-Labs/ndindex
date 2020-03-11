from numpy.testing import assert_equal

from pytest import fail

def check_same(a, raw_type, raw_args, idx_type, idx_args, raises=False):
    exception = None
    try:
        raw = raw_type(*raw_args)
        a_raw = a[raw]
    except Exception as e:
        exception = e

    try:
        idx = idx_type(*idx_args)
        a_idx = a[idx.raw]
    except Exception as e:
        if not exception:
            fail(f"Raw form does not raise but ndindex form does ({raw_type}(*{raw_args}))")
        assert type(e) == type(exception)
        assert e.args == exception.args, (e.args, exception.args)
    else:
        if exception:
            fail(f"ndindex form did not raise but raw form does ({idx_type}(*{idx_args}))")

    if not exception:
        assert_equal(a_raw, a_idx)
