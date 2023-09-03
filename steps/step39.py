if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero import Variable


def test_sum_function():
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)


def test_sum_arguments():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x, axis=0)
    y.backward()
    print(y)
    print(x.grad)


def test_sum_method():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.sum(keepdims=True)
    print(y.shape)


if __name__ == '__main__':
    test_sum_arguments()