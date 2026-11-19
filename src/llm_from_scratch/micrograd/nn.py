import numpy as np
from llm_from_scratch.micrograd.engine import Value, draw_dot

class Neuron:
    def __init__(self, n_in, nonlin=True):
        np.random.seed(42)
        self.w = [Value(np.random.uniform(-1, 1)) for n in range(n_in)]
        self.b = Value(np.random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh() if self.nonlin else act
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out, nonlin=True):
        self.neurons = [Neuron(n_in, nonlin) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, n_in, n_outs):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(n_outs) - 1)) 
                       for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)   
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

