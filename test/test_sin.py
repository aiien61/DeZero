if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numbers import Number
from dezero import Variable, Function


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x: Number) -> Number:
    return Sin()(x)


def my_sin(x: Number, threshold=0.0001) -> Number:
    from math import factorial

    y = 0
    for i in range(int(1e+5)):
        c = (-1) ** i / factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y += t
        if abs(t.data) < threshold:
            break
    return y

x = Variable(np.array(np.pi/4))

y = sin(x)
y.backward()
print(y.data)
print(x.data)

y = my_sin(x)
y.backward()
print(y.data)
print(x.data)
