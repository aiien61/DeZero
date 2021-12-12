if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(y)
print(x.grad)

x = Variable(np.random.randn(1, 2, 3))
print(x)
y = x.reshape((3, 2))
print(y)
y = x.reshape(3, 2)
print(y)