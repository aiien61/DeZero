import numpy as np
from typing import NoReturn


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TabError(f"{type(data)} is not supported")

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, function) -> NoReturn:
        self.creator = function
        return None

    def backward(self) -> NoReturn:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        functions = [self.creator]
        while functions:
            f = functions.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # DO NOT USE += operator

                if x.creator is not None:
                    functions.append(x.creator)
        return None

    def cleargrad(self) -> NoReturn:
        self.grad = None
        return None


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def add(x0, x1):
    return Add()(x0, x1)


def square(x):
    return Square()(x)


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x

# First trial
x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
if x.grad != 2:
    raise ArithmeticError(f"x.grad should be 2, not {x.grad}")
else:
    print("x.grad:", x.grad, "...OK")

# Second trial
x.cleargrad()
y = add(add(x, x), x)
y.backward()
if x.grad != 3:
    raise ArithmeticError(f"x.grad should be 3, not {x.grad}")
else:
    print("x.grad:", x.grad, "...OK")
