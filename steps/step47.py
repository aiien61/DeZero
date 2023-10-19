if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero import Variable, as_variable
from dezero.models import MLP

# example of showing how np.add.at() works
def example_before_slice():
    a = np.zeros((2, 3))
    print(a)
    
    b = np.ones((3, ))
    print(b)

    slices = 1
    np.add.at(a, slices, b)  # choose the array of dimension 1 and add b to it.
    print(a)

# example of how to slice by using get_item() towards Variable
def example_of_get_item():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.get_item(x, 1)
    print(y)

    y.backward()
    print(x.grad)

def how_get_item_works_toward_variable():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    indices = np.array([0, 0, 1])
    y = F.get_item(x, indices)
    print(y)

    Variable.__getitem__ = F.get_item

    y = x[1]
    print(y)

    y = x[:, 2]
    print(y)


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


def main():
    # size of the first output layer is 10, size of the second output is 3
    model = MLP((10, 3))

    # size of input layer is 2
    # one sample with two dimensions
    x = np.array([[0.2, -0.4]])

    # layers connectivity: 2 -> 10 -> 3
    y = model(x)
    p = softmax1d(y)

    print(y)
    print(p)
    print(np.sum(p.data))


def evaluation_using_softmax_cross_entropy():
    model = MLP((10, 3))

    x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
    t = np.array([2, 0, 1, 0])
    y = model(x)
    loss = F.softmax_cross_entropy_simple(y, t)
    print(loss)

if __name__ == '__main__':
    evaluation_using_softmax_cross_entropy()