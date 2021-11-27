from collections.abc import Iterable

from numpy import ndarray, ones_like
from typing import NoReturn

from utils import PrioritySet, PriorityItem


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, function) -> NoReturn:
        self.creator = function
        self.generation = function.generation + 1
        return None

    def backward(self) -> NoReturn:
        if self.grad is None:
            self.grad = ones_like(self.data)

        def priority_set(iterable_queue: Iterable):
            return PrioritySet()(iterable_queue)
        
        creators_list = priority_set([self.creator])
        while creators_list:
            print("creators_list:", creators_list)
            creator = creators_list.pop()
            print("take creator:", creator)
            gys = [output.grad for output in creator.outputs]
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
                    print("collect creator:", x.creator)
            print("updated creators_list:", creators_list, end="\n\n")
        return None

    def cleargrad(self) -> NoReturn:
        self.grad = None
        return None
