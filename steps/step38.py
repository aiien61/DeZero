if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero import Variable


def test_reshape():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.reshape(x, (6,))
    print(y)
    y.backward(retain_grad=True)
    print(x.grad)


def test_reshape_argument_acceptance():
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    print(x)

    y = x.reshape((2, 3))
    print(y)

    y = x.reshape([2, 3])
    print(y)

    y = x.reshape(*[2, 3])
    print(y)

    y = x.reshape(2, 3)
    print(y)


def test_transpose():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)
    print(y)
    y.backward()
    print(x.grad)


def test_transpose_property():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.transpose()
    print(y)
    y = x.T
    print(y)


if __name__ == "__main__":
    test_transpose_property()