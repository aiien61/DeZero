if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import dezero.functions as F
from dezero import Variable


class Test(unittest.TestCase):
    testcases: dict = {
        'x_1d': Variable(np.array([1, 2, 3, 4, 5, 6])),
        'x_2d': Variable(np.array([[1, 2, 3], [4, 5, 6]])),

        'y_sum_axis_None': Variable(np.array(21)),
        'y_sum_axis_0': Variable(np.array([5, 7, 9])),
        'y_sum_axis_1': Variable(np.array([6, 15])),

        'y_sum_keepdims': Variable(np.array([[21]])),

        'x_grad_1d': Variable(np.array([1, 1, 1, 1, 1, 1])),
        'x_grad_2d': Variable(np.array([[1, 1, 1], [1, 1, 1]]))
    }
    
    def test_1D_sum(self):
        print('test_1D_sum')
        x = self.testcases['x_1d']
        print('x:', x)
        
        y = F.sum(x)
        print('y:', y)

        # ------------------------------------------------------------
        expected = self.testcases['y_sum_axis_None'].data
        actual = y.data
        print('expected y.data:', expected)
        print('actual y.data:', actual)
        self.assertTrue(actual == expected)

        # ------------------------------------------------------------
        y.backward()

        # ------------------------------------------------------------
        expected = self.testcases['x_grad_1d'].data
        actual = x.grad.data

        print('expected x.grad:', expected)
        print('actual x.grad:', actual)
        self.assertTrue((actual == expected).all())

        # ------------------------------------------------------------
        expected = self.testcases['x_grad_1d'].shape
        actual = x.grad.shape
        print('expected x.grad.shape:', expected)
        print('actual x.grad.shape:', actual)
        self.assertTrue(actual == expected)


    def test_2D_sum_with_axis(self):
        print('\ntest_2D_sum_with_axis')
        x = self.testcases['x_2d']
        print('x:', x)

        # ------------------------------------------------------------
        y = F.sum(x, axis=0)
        print('column-wise sum y:', y)

        y.backward()

        expected = self.testcases['y_sum_axis_0'].data
        actual = y.data
        print('expected y.data:', expected)
        print('actual y.data', actual)
        self.assertTrue(all(actual == expected))

        expected = self.testcases['x_grad_2d'].data
        actual = x.grad.data
        print('expected x.grad:', expected)
        print('actual x.grad:', actual)
        self.assertTrue((actual == expected).all())

        expected = self.testcases['x_grad_2d'].shape
        actual = x.shape
        print('expected x.grad.shape:', expected)
        print('actual x.grad.shape:', actual)
        self.assertTrue(actual == expected)

        # ------------------------------------------------------------
        x.cleargrad()
        y = F.sum(x, axis=1)
        print('x:', x)
        print('row-wise sum y:', y)

        y.backward()

        expected = self.testcases['y_sum_axis_1'].data
        actual = y.data
        print('expected y.data:', expected)
        print('actual y.data:', actual)
        self.assertTrue((actual == expected).all())

        expected = self.testcases['x_grad_2d'].data
        actual = x.grad.data
        print('expected x.grad:', expected)
        print('actual x.grad:', actual)
        self.assertTrue((actual == expected).all())

        expected = self.testcases['x_grad_2d'].shape
        actual = x.shape
        print('expected x.grad.shape:', expected)
        print('actual x.grad.shape:', actual)
        self.assertTrue(actual == expected)
    
    
    def test_2D_sum_with_keepdims(self):
        print('\ntest_2D_sum_with_keepdims')
        x = self.testcases['x_2d']
        print('x:', x)
        x.cleargrad()
        y = x.sum(keepdims=True)
        print('y with dims kept:', y)

        y.backward()
        
        expected = self.testcases['y_sum_keepdims'].data
        actual = y.data
        print('expected y.data:', expected)
        print('actual y.data:', actual)
        self.assertTrue((actual == expected).all())
        
        expected = self.testcases['y_sum_keepdims'].shape
        actual = y.shape
        print('expected y.shape:', expected)
        print('actual y.shape:', actual)
        self.assertTrue(actual == expected)
        
        expected = self.testcases['x_grad_2d'].data
        actual = x.grad.data
        print('expected x.grad:', expected)
        print('actual x.grad:', actual)
        self.assertTrue((actual == expected).all())

        expected = self.testcases['x_grad_2d'].shape
        actual = x.shape
        print('expected x.grad.shape:', expected)
        print('actual x.grad.shape:', actual)
        self.assertTrue(actual == expected)


    def test_multi_dims_sum(self):
        x = Variable(np.random.randn(2, 3, 4, 5))
        y = x.sum(keepdims=True)
        print('x:', x)
        print('y:', y)
        
        expected = Variable(np.random.randn(1, 1, 1, 1)).shape
        actual = y.shape
        print('expected y.shape:', expected)
        print('actual y.shape:', actual)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()