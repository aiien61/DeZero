if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero.functions as F
import dezero.layers as L
from dezero import utils


class Model(L.Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


# generalise the model to be a multi-layer perceptron (MLP)
class MLP(Model):
    def __init__(self, full_connect_output_sizes, activation=F.sigmoid_simple):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(full_connect_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, f'l{i}', layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
