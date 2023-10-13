if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import dezero.functions as F
from dezero import Variable


def matrix_multiplication():
    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))
    print('x:', x)
    print('x.shape:', x.shape)
    print('W:', W)
    print('W.shape:', W.shape)
    y = F.matmul(x, W)
    print('y = x * W =', y)
    print('y.shape:', y.shape)
    y.backward()

    print('x.grad:', x.grad)
    print('x.grad.shape', x.grad.shape)
    print('W.grad:', W.grad)
    print('W.grad.shape:', W.grad.shape)


if __name__ == '__main__':
    matrix_multiplication()