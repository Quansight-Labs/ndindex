from numpy.testing import assert_equal

from pytest import fail

def raises_same(a, raw_type, raw_args, idx_type, idx_args):
    try:
        raw = raw_type(*raw_args)
        a[raw]
    except Exception as e:
        exception = e
    else:
        fail(f"Raw form does not raise ({raw_type}(*{raw_args}))")

    try:
        idx = idx_type(*idx_args)
        a[idx.raw]
    except Exception as e:
        assert type(e) == type(exception)
        assert e.args == exception.args, (e.args, exception.args)
    else:
        fail(f"ndindex form did not raise ({idx_type}(*{idx_args}))")

def check_same(a, raw_type, raw_args, idx_type, idx_args, raises=False):
    if raises:
        return raises_same(a, raw_type, raw_args, idx_type, idx_args)

    raw = raw_type(*raw_args)
    idx = idx_type(*idx_args)
    assert_equal(a[raw], a[idx.raw])
