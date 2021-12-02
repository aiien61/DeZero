import heapq
import weakref
import functools
import contextlib
from numbers import Number
from collections import namedtuple
from collections.abc import Iterable

import numpy as np
from typing import NoReturn


class Config:
    enable_backprop = True


@functools.total_ordering
class PriorityItem(object):
    def __init__(self, obj):
        self.obj = obj

    def __lt__(self, other):
        return isinstance(other, PriorityItem) and self.obj > other.obj


class PrioritySet(object):
    Item = namedtuple("Item", ["priority", "item", "id"])

    def __call__(self, queue=None):
        self.maxheap = []
        self.heapset = set()
        if queue is None:
            return self

        if not isinstance(queue, Iterable):
            raise TypeError(f"{type(queue)} is not iterable")

        queue = map(PriorityItem, queue)
        for x in queue:
            self.add(x)
        return self

    def add(self, x: PriorityItem) -> NoReturn:
        if id(x.obj) not in self.heapset:
            heapq.heappush(self.maxheap, x)
            self.heapset.add(id(x.obj))
        return None

    def pop(self):
        x = heapq.heappop(self.maxheap)
        self.heapset.remove(id(x.obj))
        return x.obj

    def __len__(self):
        return len(self.maxheap)

    def __repr__(self):
        priority_dict = {
            i: self.Item(priority=x.obj.generation,
                         item=type(x.obj).__name__,
                         id=id(x.obj))
            for i, x in enumerate(self.maxheap)
        }
        return str(priority_dict)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        var_str = "variable({data})"
        if (data := self.data) is None:
            text = None
        else:
            text = str(data).replace("\n", "\n" + " " * 9)
        return var_str.format(data=text)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def set_creator(self, function) -> NoReturn:
        self.creator = function
        self.generation = function.generation + 1
        return None

    def backward(self, retain_grad=False) -> NoReturn:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        def priority_set(iterable_queue: Iterable):
            return PrioritySet()(iterable_queue)

        creators_list = priority_set([self.creator])
        while creators_list:
            # print("creators_list:", creators_list)
            creator = creators_list.pop()
            # print("take creator:", creator)
            gys = [output().grad for output in creator.outputs]  # use weakref
            gxs = creator.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(creator.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # DO NOT USE += operator

                if x.creator is not None:
                    creators_list.add(PriorityItem(x.creator))
                    # print("collect creator:", x.creator)
            # print("updated creators_list:", creators_list, end="\n\n")
            if not retain_grad:
                for y in creator.outputs:
                    y().grad = None

        return None

    def cleargrad(self) -> NoReturn:
        self.grad = None
        return None


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def __eq__(self, other):
        if not isinstance(other, Function):
            raise TypeError(f"{type(other)} is not Function")
        return self.generation == other.generation

    def __lt__(self, other):
        if not isinstance(other, Function):
            raise TypeError(f"{type(other)} is not Function")
        return self.generation < other.generation

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0: Number, x1: Number) -> Number:
        y = x0 + x1
        return y

    def backward(self, gy: Number) -> tuple:
        return gy, gy


def add(x0: Number, x1: Number) -> Number:
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0: Number, x1: Number) -> Number:
        y = x0 * x1
        return y

    def backward(self, gy: Number) -> tuple:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0: Number, x1: Number) -> Number:
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x: Number) -> Number:
    return Neg()(x)


class Sub(Function):
    def forward(self, x0: Number, x1: Number) -> Number:
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0: Number, x1: Number) -> Number:
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0: Number, x1: Number) -> Number:
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0: Number, x1: Number) -> Number:
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / (x1 ** 2))
        return gx0, gx1


def div(x0: Number, x1: Number) -> Number:
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: Number, x1: Number) -> Number:
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx


def power(x: Number, c: int) -> Number:
    return Pow(c)(x)


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def setup_variable() -> NoReturn:
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = power
    return None


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)
