if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Function

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)

# gradient descent
learning_rate = .001
iterations = 1000

for i in range(iterations):
    print(x0, x1)
    y = rosenbrock(x0, x1)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    print(x0.grad, x1.grad)
    print(x0.data, x1.data)
    print()

    x0.data -= learning_rate * x0.grad
    x1.data -= learning_rate * x1.grad
