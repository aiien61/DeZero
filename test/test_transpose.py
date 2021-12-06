if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
print(y)
y.backward()
print(x.grad)

x = Variable(np.random.rand(2, 3))
print(x)
y = x.transpose()
print(y)
y = x.T
print(y)
