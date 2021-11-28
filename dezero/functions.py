from numbers import Number

import numpy as np

from core_simple import Function


class Square(Function):
    def forward(self, x: Number) -> Number:
        y = x ** 2
        return y

    def backward(self, gy: Number) -> Number:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Cube(Function):
    def forward(self, x: Number) -> Number:
        y = x ** 3
        return y

    def backward(self, gy: Number) -> Number:
        x = self.inputs[0].data
        gx = 3 * (x ** 2) * gy
        return gx


class Exp(Function):
    def forward(self, x: Number) -> Number:
        y = np.exp(x)
        return y

    def backward(self, gy: Number) -> Number:
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def square(x: Number) -> Number:
    return Square()(x)


def cube(x: Number) -> Number:
    return Cube()(x)


def exp(x: Number) -> Number:
    return Exp()(x)
