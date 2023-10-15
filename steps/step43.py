if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F
from dezero import Variable

# example of nonlinear transformation

# data
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# initial weights
input_layer_dims, hidden_layer_dims, output_layer_dims = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(input_layer_dims, hidden_layer_dims))
b1 = Variable(np.zeros(hidden_layer_dims))
W2 = Variable(0.01 * np.random.randn(hidden_layer_dims, output_layer_dims))
b2 = Variable(np.zeros(output_layer_dims))

# TODO: change to sigmoid()
# Inference
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid_simple(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

# learning
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:
        print(loss)

# plot
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, 0.01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data)
plt.show()