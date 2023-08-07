if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
from dezero import Variable, Function
from dezero.utils import plot_dot_graph

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)


def maclaurin_sin(x, threshold=1e-4):
    y = 0
    for i in range(1000):
        const = (-1) ** i / math.factorial(2 * i + 1)
        var = x ** (2 * i + 1)
        term = const * var
        y = y + term
        if abs(term.data) < threshold:
            break
    return y


if __name__ == "__main__":
    x = Variable(np.array(np.pi / 4))
    y = sin(x)
    y.backward()

    print(y.data)
    print(x.grad)

 
    x.cleargrad()
    y.cleargrad()
    x = Variable(np.array(np.pi / 4))
    y = maclaurin_sin(x)
    y.backward()
    print(y.data)
    print(x.grad)
    
    x.name = "x"
    y.name = "y"
    plot_dot_graph(y, verbose=False, to_file="sin.png")
