if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

# numpy broadcast
x0 = np.array([1, 2, 3])
x1 = np.array([10])
y = x0 + x1
print(y)


# dezero broadcast
x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)

y.backward()
print(x1.grad)  # should be variable([3])

x0.cleargrad()
x1.cleargrad()

z = x0 * x1
print(z)
z.backward()
print(x1.grad)