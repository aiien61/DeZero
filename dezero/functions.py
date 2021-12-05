from numbers import Number

import numpy as np

from dezero.core import Function


class Square(Function):
    def forward(self, x: Number) -> Number:
        return  x ** 2

    def backward(self, gy: Number) -> Number:
        x, *_ = self.inputs
        gx = 2 * x * gy
        return gx


def square(x: Number) -> Number:
    return Square()(x)


class Cube(Function):
    def forward(self, x: Number) -> Number:
        return x ** 3

    def backward(self, gy: Number) -> Number:
        x, *_ = self.inputs
        gx = 3 * (x ** 2) * gy
        return gx


def cube(x: Number) -> Number:
    return Cube()(x)


class Exp(Function):
    def forward(self, x: Number) -> Number:
        return np.exp(x)

    def backward(self, gy: Number) -> Number:
        x, *_ = self.inputs
        gx = np.exp(x) * gy
        return gx


def exp(x: Number) -> Number:
    return Exp()(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x, *_ = self.inputs
        return cos(x) * gy


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x, *_ = self.inputs
        return -sin(x) * gy


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y, *_ = self.outputs
        y = y()  # y is weakref
        return (1 - y ** 2) * gy


def tanh(x):
    return Tanh()(x)
