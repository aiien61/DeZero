if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import dezero.functions as F
from dezero import Variable


def example_add():
    # show broadcasting
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    print('x0:', x0)
    print('x1:', x1)
    y = x0 + x1
    print('x0 + x1 =', y)

    y.backward()
    print('x0.grad:', x0.grad)
    print('x1.grad:', x1.grad)
    

def example_sub():
    # show broadcasting
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    print('x0:', x0)
    print('x1:', x1)
    y = x0 - x1
    print('x0 - x1 =', y)

    y.backward()
    print('x0.grad:', x0.grad)
    print('x1.grad:', x1.grad)


def example_mul():
    # show broadcasting
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    print('x0:', x0)
    print('x1:', x1)
    y = x0 * x1
    print('x0 * x1 =', y)

    y.backward()
    print('x0.grad:', x0.grad)
    print('x1.grad:', x1.grad)


def example_div():
    # show broadcasting
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    print('x0:', x0)
    print('x1:', x1)
    y = x0 / x1
    print('x0 / x1 =', y)

    y.backward()
    print('x0.grad:', x0.grad)
    print('x1.grad:', x1.grad)


if __name__ == "__main__":
    print('addition broadcasting:')
    example_add()
    print('\n---\n')
    print('subtraction broadcasting:')
    example_sub()
    print('\n---\n')
    print('multiplication broadcasting:')
    example_mul()
    print('\n---\n')
    print('division broadcasting:')
    example_div()
