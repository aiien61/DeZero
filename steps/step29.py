if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
from dezero import Variable


def f(x):
    return x ** 4 - 2 * x ** 2


def gx2(x):
    return 12 * x - 4

if __name__ == "__main__":
    print("Approach by Newton's method")
    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print(i, x)
        
        y = f(x)
        x.cleargrad()
        y.backward()

        x.data -= x.grad / gx2(x.data)
    
    print("Approach by gradient descent")
    x = Variable(np.array(2.0))
    lr = 0.01

    i = 0
    while True:
        print(i, x)
        y = f(x)
        x.cleargrad()
        y.backward()
        if math.isclose(x.data, 1, abs_tol=0.001):
            break

        x.data -= lr * x.grad
        i += 1
