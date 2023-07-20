import unittest

import numpy as np
from step09 import Variable, square
from step04 import numerical_diff


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    
    # by manually computing gradient
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    # by gradient checking
    def test_gradient(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        expected = numerical_diff(square, x)
        is_close = np.allclose(x.grad, expected)
        self.assertTrue(is_close)

