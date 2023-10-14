if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import dezero.functions as F
from dezero import Variable

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.randn(100, 1)
x, y = Variable(x), Variable(y)  # optional as x, y can be auto converted into Variable in F.matmul() function.

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

learning_rate = 0.1
iterations = 100

for i in range(iterations):
    y_pred = F.linear(x, W, b)
    # y_pred = F.linear_simple(x, W, b)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= learning_rate * W.grad.data
    b.data -= learning_rate * b.grad.data

    print(W, b, loss)
