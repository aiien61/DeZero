import numpy as np

from dezero import Variable, using_config
from dezero.core_simple import add, mul
from dezero.functions import exp, square, cube

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


def switch_backpropogation_modes():
    print("Backpropogation mode is on:")
    Config.enable_backprop = True
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    y.backward()
    print(x.grad)

    print("Backpropogation mode is off:")
    Config.enable_backprop = False
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    try:
        y.backward()
        print(x.grad)
    except AttributeError:
        print("unable to execute y.backward()")
        print("backpropogation mode is off")


def switch_backpropogation_modes_using_with():
    print('Using "with" and using_config() to manage resource', end='...')
    with using_config("enable_backprop", False):
        x = Variable(np.array(2.0))
        y = square(square(square(x)))
    print("OK")


    print('Using "with" and no_grad() to manage resource', end='...')
    with no_grad():
        x = Variable(np.array(2.0))
        y = square(square(square(x)))
    print("OK")


def using_multiplication_function():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    y = add(mul(a, b), c)
    y.backward()
    
    print(y)
    print(a.grad)
    print(b.grad)


def using_multiplication_operator():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    y = a * b + c
    y.backward()

    print(y)
    print(a.grad)
    print(b.grad)


def using_multiplication_operator_both_sides():
    a = Variable(np.array(3.0))

    y = 3.0 * a
    print(y)

    y = 3.0 + a
    print(y)


def higher_variable_priority_in_operation():
    x = Variable(np.array(3.0))

    y = np.array([2.0]) + x
    print(y)


def basic_operation():
    print('negative Variable: x=2, y=-x')
    x = Variable(np.array(2.0))
    y = -x
    print('y=', y)

    print('Variable subtraction: x=2, y1=2-x, y2=x-1')
    x = Variable(np.array(2.0))
    y1 = 2.0 - x
    y2 = x - 1.0
    print('y1=', y1)
    print('y2=', y2)

    print('Variable division: x=2, y1=2/x, y2=x/1')
    x = Variable(np.array(2.0))
    y1 = 2.0 / x
    y2 = x / 1.0
    print('y1=', y1)
    print('y2=', y2)

    print('Power of Variable: x=-2, y=x**3')
    x = Variable(np.array(-2.0))
    y = x ** 3
    print('y=', y)


