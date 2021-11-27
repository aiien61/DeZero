import functools
import heapq
from collections import namedtuple
from collections.abc import Iterable
from typing import NoReturn


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
        priority_dict= {
            i: self.Item(priority=x.obj.generation,
                         item=type(x.obj).__name__,
                         id=id(x.obj))
            for i, x in enumerate(self.maxheap)
        }
        return str(priority_dict)
