import ast

def literal_eval_index(node_or_string):
    """
    "Safely" (needs validation) evaluate an expression node or a string containing
    a (limited subset) of valid numpy index or slice expressions.
    """
    if isinstance(node_or_string, str):
        node_or_string = ast.parse('dummy[{}]'.format(node_or_string.lstrip(" \t")) , mode='eval')
    if isinstance(node_or_string, ast.Expression):
        node_or_string = node_or_string.body
    if isinstance(node_or_string, ast.Subscript):
        node_or_string = node_or_string.slice

    def _raise_malformed_node(node):
        raise ValueError(f'malformed node or string: {node!r}')

    # from cpy37, should work until they remove ast.Num (not until cpy310)
    def _convert_num(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return node.value
        elif isinstance(node, ast.Num):
            # ast.Num was removed from ast grammar in cpy38
            return node.n
        _raise_malformed_node(node)
    def _convert_signed_num(node):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _convert_num(node.operand)
            if isinstance(node.op, ast.UAdd):
                return + operand
            else:
                return - operand
        return _convert_num(node)

    def _convert(node):
        if isinstance(node, ast.Tuple):
            return tuple(map(_convert, node.elts))
        elif isinstance(node, ast.List):
            return list(map(_convert, node.elts))
        elif isinstance(node, ast.Slice):
            return slice(
                _convert(node.lower) if node.lower is not None else None,
                _convert(node.upper) if node.upper is not None else None,
                _convert(node.step) if node.step is not None else None,
            )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'slice' and node.keywords == []:
            # support for parsing slices written out as 'slice(...)' objects
            return slice(*map(_convert, node.args))
        elif isinstance(node, ast.NameConstant) and node.value is None:
            # support for literal None in slices, eg 'slice(None, ...)'
            return None
        elif isinstance(node, ast.Ellipsis):
            # support for three dot '...' ellipsis syntax
            return ...
        elif isinstance(node, ast.Name) and node.id == 'Ellipsis':
            # support for 'Ellipsis' ellipsis syntax
            return ...
        elif isinstance(node, ast.Index):
            # ast.Index was removed from ast grammar in cpy39
            return _convert(node.value)
        elif isinstance(node, ast.ExtSlice):
            # ast.ExtSlice was removed from ast grammar in cpy39
            return tuple(map(_convert, node.dims))

        return _convert_signed_num(node)
    return _convert(node_or_string)
