import os
import subprocess


def _dot_var(v, verbose=False):
    dot_var = '{var_id} [label="{var_name}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)
    return dot_var.format(var_id=id(v), var_name=name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        # y is weakref
        txt += dot_edge.format(id(f), id(y()))  
    return txt


def get_dot_graph(output, verbose=True):
    txt = ""
    funcs = []
    seen_set = set()
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    
    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    filename, dot_extension = os.path.splitext(to_file)
    
    # Stores as a .dot file
    utils_path = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(utils_path, '../images')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    graph_path = os.path.join(tmp_dir, f"{filename}.dot")

    with open(graph_path, mode='w') as file:
        file.write(dot_graph)

    # Calls dot command to visualize .dot file
    extension = dot_extension[1:]
    dot_path = os.path.relpath(graph_path, os.getcwd())
    file_path = os.path.join(os.path.dirname(dot_path), to_file)
    cmd = f"dot {dot_path} -T {extension} -o {file_path}"
    subprocess.run(cmd, shell=True)


def reshape_sum_backward(gy, shape, axis, keepdims):
    pivot_shape: tuple = (shape[0], 1)

    if not gy.shape:
        gy = gy.reshape((1,))

    if axis == 0:
        return gy.tile(pivot_shape)
    elif axis == 1:
        return gy.reshape(pivot_shape)
    else: # when axis is None
        return gy

# # Chainer's version
# def sum_to(x, shape):  
#     """Sum element along to output an array pf a given shape.

#     Args:
#         x (ndarray): Input array.
#         shape (tuple): Shape of the input array.
    
#     Returns:
#         ndarray: Output array of the shape.
#     """
#     ndim = len(shape)
#     lead = x.ndim - ndim
#     lead_axis = tuple(range(lead))

#     axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
#     y = x.sum(lead_axis + axis, keepdims=True)
#     if lead > 0:
#         y = y.squeeze(lead_axis)

#     return y

# Refactored version
def sum_to(x, shape):
    """Sum element along to output an array pf a given shape.

    Args:
        x (ndarray): Input array.
        shape (tuple): Shape of the input array.
    
    Returns:
        ndarray: Output array of the shape.
    """
    dims = {k: v for v, k in enumerate(shape)}
    axis = dims.get(1) if dims.get(1) else 0
    y = x.sum(axis=axis, keepdims=True)
    while len(y.shape) != len(shape):
        if 1 in y.shape:
            y = y.squeeze()
        else:
            raise ValueError(f"{x} can't be summed up into the shape {shape}.")
    return y