from numbers import Number

import numpy as np
from typing import NoReturn

from variable import Variable


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    def __eq__(self, other):
        if not isinstance(other, Function):
            raise TypeError(f"{type(other)} is not Function")
        return self.generation == other.generation

    def __lt__(self, other):
        if not isinstance(other, Function):
            raise TypeError(f"{type(other)} is not Function")
        return self.generation < other.generation

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0: Number, x1: Number) -> Number:
        y = x0 + x1
        return y
    
    def backward(self, gy: Number) -> tuple:
        return gy, gy


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


def add(x0: Number ,x1: Number) -> Number:
    return Add()(x0, x1)


def square(x: Number) -> Number:
    return Square()(x)


def cube(x: Number) -> Number:
    return Cube()(x)


def exp(x: Number) -> Number:
    return Exp()(x)


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x
