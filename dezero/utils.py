import os
import subprocess
from collections.abc import Iterable


def _dot_var(v, verbose=False):
    dot_var = '{id} [label="{name}"], color=orange, style=filled\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id=id(v), name=name)


def _dot_func(f):
    dot_func = '{} [label="{}"], color=lightblue, style=filled, shape=box\n'
    txt = dot_func.format(id(f), f.__class__.__name__)
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


def get_dot_graph(output, verbose=True):
    from .core import PriorityItem, PrioritySet

    txt = ''
    # function_list = []
    # seen_set = set()
    
    # def add_function(f):
    #     if f not in seen_set:
    #         function_list.append(f)
    #         # funcs.sort(key=lambda x: x.generation)
    #         seen_set.add(f)

    def priority_set(iterable_queue: Iterable):
        return PrioritySet()(iterable_queue)

    function_list = priority_set([output.creator])

    # add_function(output.creator)
    txt += _dot_var(output, verbose)

    while function_list:
        function = function_list.pop()
        txt += _dot_func(function)
        for x in function.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                # add_function(x.creator)
                function_list.add(PriorityItem(x.creator))
    return 'digraph g{\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # store dot file
    tmp_dir = os.path.join(os.path.expanduser("~"), '.test')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # execute dot command
    extension = os.path.splitext(to_file)[1][1:]
    cmd = f'dot {graph_path} -T {extension} -o {to_file}'
    subprocess.run(cmd, shell=True)


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy