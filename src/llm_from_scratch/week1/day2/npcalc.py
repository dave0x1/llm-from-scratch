import numpy as np

def loss(X, W, b, Y):
    E = X @ W + b - Y            # X@W: (N,d_out); + b: broadcast (d_out,)->(N,d_out); - Y: (N,d_out)
    return np.mean(E ** 2)       # scalar

def grads(X, W, b, Y):
    N, d_out = Y.shape
    E = X @ W + b - Y            # (N, d_out)
    G = 2 * E / (N * d_out)      # dL/dE, (N, d_out)
    dW = X.T @ G                 # (d_in,N) @ (N,d_out) -> (d_in, d_out)
    db = G.sum(axis=0)           # (N,d_out) summed over rows -> (d_out,)
    return dW, db

def numeric_grad_array(f, arr, h=1e-5):
    g = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        plus, minus = arr.copy(), arr.copy()
        plus[idx] += h
        minus[idx] -= h
        g[idx] = (f(plus) - f(minus)) / (2 * h)
    return g

rng = np.random.default_rng(0)
N, d_in, d_out = 5, 3, 2
X = rng.standard_normal((N, d_in))
W = rng.standard_normal((d_in, d_out))
b = rng.standard_normal(d_out)
Y = rng.standard_normal((N, d_out))

dW, db = grads(X, W, b, Y)
dW_num = numeric_grad_array(lambda W_: loss(X, W_, b, Y), W)
db_num = numeric_grad_array(lambda b_: loss(X, W, b_, Y), b)
print(np.abs(dW - dW_num).max(), np.abs(db - db_num).max())   # expect ~1e-10 or smaller