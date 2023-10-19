if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import utils
from dezero.core import Function, as_variable

# ---------------------------------------------------------
# basic functions: sin / cos / tanh / exp
# ---------------------------------------------------------
class Sin(Function):
    def forward(self, x):
        return  np.sin(x)
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * (-sin(x))
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = y * gy
        return gx


def exp(x):
    return Exp()(x)


# ---------------------------------------------------------
# tensor operations: reshape / transpose / get_item
# ---------------------------------------------------------
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)  # x is the instance of ndarray
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx
    

def transpose(x):
    return Transpose()(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
    
    def forward(self, x):
        return x[self.slices]
    
    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)
    

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape
    
    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    return GetItem(slices)(x)


# ---------------------------------------------------------
# aggregation functions: sum / broadcast_to / sum_to / tile / matmul
# ---------------------------------------------------------
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class Tile(Function):
    def __init__(self, reps):
        self.reps = reps

    def forward(self, x):
        self.x_shape = x.shape
        self.x_ndim = x.ndim
        return np.tile(x, self.reps)
    
    def backward(self, gy):
        if self.x_ndim == 1:
            gx = gy[0, :self.shape[0]]
        else:
            gx = gy[:self.shape[0], :self.shape[1]]
        return gx


def tile(x, reps):
    return Tile(reps)(x)


class MatMul(Function):
    def forward(self, x, W):
        y = np.matmul(x, W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW= matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


# ---------------------------------------------------------
# loss functions: mse
# ---------------------------------------------------------
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        return (diff ** 2).sum() / len(diff)
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def mean_squared_error_simple(x0, x1):
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)


# ---------------------------------------------------------
# transformation functions: linear
# ---------------------------------------------------------
class Linear(Function):
    def forward(self, x, W, b):
        y = matmul(x, W)
        if b is not None:
            y += b
        return y
    
    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW= matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


# Manually release t.data for memory efficiency
def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    
    y = t + b
    t.data = None
    return y

# ---------------------------------------------------------
# activation functions: sigmoid / softmax
# ---------------------------------------------------------
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y
