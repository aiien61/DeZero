if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F
import dezero.layers as L
from dezero import Variable, Parameter


# Parameter instances
x = Variable(np.array(1.0))
p = Parameter(np.array(2.0))
y = x * p

print(isinstance(p, Parameter))
print(isinstance(x, Parameter))
print(isinstance(y, Parameter))
print('------------\n')

# Layer instance
layer = L.Layer()

layer.p1 = Parameter(np.array(1))
layer.p2 = Parameter(np.array(2))
layer.p3 = Variable(np.array(3))
layer.p4 = "test"

print(layer._params)
print('------------')

for name in layer._params:
    print(name, layer.__dict__[name])

print('------------\n')

# use layers to build network prediction structure
# data
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# define structure
l1 = L.Linear(10)  # input layer: to set the size of inputs
l2 = L.Linear(1)  # output layer

# determine prediction method
# TODO: change to sigmoid()
def predict(x):
    y = l1(x)
    y = F.sigmoid_simple(y)
    y = l2(y)
    return y

# learning process
lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

# result
print('final weights:', l1.b)


# plot
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, 0.01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data)
plt.show()