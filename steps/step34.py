if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F
from dezero import Variable


def test_n_order_differentiation():
    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)

    for i in range(3):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        print(x.grad)


def plot_n_order_differentiation_of_sin():
    x = Variable(np.linspace(-7, 7, 200))  # [-7, -6.92964, -6.85929, ..., 7]
    y = F.sin(x)
    y.backward(create_graph=True)

    logs = [y.data]

    for i in range(3):
        logs.append(x.grad.data)
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
    
    # plot
    labels = ["y=sin(x)", "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        plt.plot(x.data, logs[i], label=labels[i])
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    plot_n_order_differentiation_of_sin()
    
