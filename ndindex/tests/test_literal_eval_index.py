import ast
from hypothesis import given, example
from hypothesis.strategies import one_of
import pytest

from ..literal_eval_index import literal_eval_index
from .helpers import ellipses, ints, slices, tuples, _doesnt_raise

Tuples = tuples(one_of(
    ellipses(),
    ints(),
    slices(),
)).filter(_doesnt_raise)

ndindexStrs = one_of(
    ellipses(),
    ints(),
    slices(),
    Tuples,
).map(lambda x: f'{x}')

class _Dummy:
    def __getitem__(self, x):
        return x
_dummy = _Dummy()

@example('3')
@example('-3')
@example('...')
@example('Ellipsis')
@example('+3')
@example('3:4')
@example('3:-4')
@example('3, 5, 14, 1')
@example('3, -5, 14, -1')
@example('3:15, 5, 14:99, 1')
@example('3:15, -5, 14:-99, 1')
@example(':15, -5, 14:-99:3, 1')
@example('3:15, -5, :, [1,2,3]')
@example('slice(None)')
@example('slice(None, None)')
@example('slice(None, None, None)')
@example('slice(14)')
@example('slice(12, 14)')
@example('slice(12, 72, 14)')
@example('slice(-12, -72, 14)')
@example('3:15, -5, slice(12, -14), (1,2,3)')
@example('..., -5, slice(12, -14), (1,2,3)')
@example('3:15, -5, slice(12, -14), (1,2,3), Ellipsis')
@given(ndindexStrs)
def test_literal_eval_index_hypothesis(ixStr):
    assert eval(f'_dummy[{ixStr}]') == literal_eval_index(ixStr)

def test_literal_eval_index_malformed_raise():
    with pytest.raises(ValueError):
        # we don't allow the bitwise not unary op
        ixStr = '~3'
        literal_eval_index(ixStr)

def test_literal_eval_index_ensure_coverage():
    # ensure full coverage, regarless of cpy version and accompanying changes to the ast grammar
    for node in (
        ast.Constant(7),
        ast.Num(7),
        ast.Index(ast.Constant(7)),
    ):
        assert literal_eval_index(node) == 7

    assert literal_eval_index(ast.ExtSlice((ast.Constant(7), ast.Constant(7), ast.Constant(7)))) == (7, 7, 7)
