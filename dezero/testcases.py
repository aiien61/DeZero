from functions import add, square, exp, cube

import numpy as np

from variable import Variable


def repeated_variable():
    print("First Trial, Equation: Y = 2X")
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    if x.grad != 2:
        raise ArithmeticError(f"x.grad should be 2, not {x.grad}")
    else:
        print(f"x.grad: {x.grad} ... OK")

    print()
    print("Second Trial, Equation: Y = 3X")
    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    if x.grad != 3:
        raise ArithmeticError(f"x.grad should be 3, not {x.grad}")
    else:
        print(f"x.grad: {x.grad} ... OK")


def backward_flow_1():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()
    print(y.data)
    print(x.grad)


def backward_flow_2():
    x = Variable(np.array(2.0))
    a = exp(x)
    y = add(square(a), cube(a))
    y.backward()
    print(y.data)
    print(x.grad)


def incircular_reference():
    for i in range(10):
        print('Round', i)
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))
        y.backward()
        print(x.grad, end='\n\n')
