if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Config, using_config
from dezero.utils import _dot_func

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))

y = x0 + x1
txt = _dot_func(y.creator)
print(txt)