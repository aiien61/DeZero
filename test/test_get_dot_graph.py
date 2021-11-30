if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Config, using_config
from dezero.utils import get_dot_graph

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(2.0))
x2 = Variable(np.array(3.0))
y = (x0 + x1) * x2

x0.name = 'x0'
x1.name = 'x1'
x2.name = 'x2'
y.name = 'y'

txt = get_dot_graph(y, verbose=False)
print(txt)

with open('sample.dot', 'w') as f:
    f.write(txt)
