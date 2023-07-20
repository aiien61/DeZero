import os
import sys
import unittest
sys.path.append(os.path.join(os.getcwd(), "steps"))

import numpy as np
from step09 import Variable, exp
from step04 import numerical_diff


class ExpTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = exp(x)
        expected = np.array(np.exp(2.0))
        self.assertEqual(y.data, expected)

    # by manually computing gradient
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = exp(x)
        y.backward()
        expected = np.array(np.exp(3.0))
        self.assertEqual(x.grad, expected)

    # by gradient checking
    def test_gradient(self):
        x = Variable(np.random.rand(1))
        y = exp(x)
        y.backward()
        expected = numerical_diff(exp, x)
        is_close = np.allclose(x.grad, expected)
        self.assertTrue(is_close)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
