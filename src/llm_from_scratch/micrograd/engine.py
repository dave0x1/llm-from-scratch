import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

class Value:
    def __init__(self, value, _children=(), _op='', label=''):
        self.value = value
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f'Value(value={self.value}, grad={self.grad})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, (self, other), '*')

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers"
        out = Value(self.value ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.value ** (other - 1) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):          # other - self
        return (-self) + other

    def __truediv__(self, other):       # self / other
        return self * other ** -1

    def __rtruediv__(self, other):      # other / self
        return other * self ** -1

    def exp(self):
        out = Value(math.exp(self.value), (self,), 'exp')

        def _backward():
            self.grad += out.value * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.value), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.value ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.value
        if x >= 0:
            t = 1 / (1 + math.exp(-x))
        else:
            e = math.exp(x)
            t = e / (1 + e)
        out = Value(t, (self,), 'sigmoid')

        def _backward():
            self.grad += out.value * (1 - out.value) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.value), (self,), 'relu')

        def _backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # iterative post-order DFS: no recursion limit
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))      # revisit after children
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='png', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | value %.4f | grad %.4f}" % (n.label, n.value, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

 
def forward(a, b, c, d):
    A, B, C, D = Value(a), Value(b), Value(c), Value(d)
    return ((A + B) + C * D).sigmoid()

x0 = np.array([0.5, -0.2, 0.3, 2.0])

A, B, C, D = (Value(v) for v in x0)
o = ((A + B) + C * D).sigmoid()
o.backward()
analytic = np.array([A.grad, B.grad, C.grad, D.grad])

def numeric_grad_array(f, arr, h=1e-5):
    g = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        plus, minus = arr.copy(), arr.copy()
        plus[idx] += h
        minus[idx] -= h
        g[idx] = (f(plus) - f(minus)) / (2 * h)
    return g

numeric = numeric_grad_array(lambda x: forward(*x).value, x0)

# a = Value(0.5, label='a')
# b = Value(-0.2, label='b')
# c = Value(0.3, label='c')
# d = Value(2.0, label='d')
# e = a + b; e.label = 'e'
# f = c * d; f.label = 'f'
# g = e + f; g.label = 'g'
# o = g.sigmoid(); o.label = 'o'


# print(o)
# o.backward()
# draw_dot(o).render(view=True)

# print(analytic, numeric)
# print("rel err:", np.max(np.abs(analytic - numeric) / (np.abs(analytic) + np.abs(numeric) + 1e-12)))