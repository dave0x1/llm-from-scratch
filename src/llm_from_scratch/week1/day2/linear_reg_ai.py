# Linear reg completed by Claude


import numpy as np
import matplotlib.pyplot as plt


def init(N, D, noise):
    rng = np.random.default_rng(42)
    w_true = rng.integers(-100, 100, D) / 10
    b_true = 4.0
    X = rng.normal(size=(N, D))                     # (N, D)  inputs
    y = X @ w_true + b_true + noise * rng.standard_normal(N)   # (N,)
    return w_true, b_true, X, y

def show(X, y):
    plt.scatter(X[:, 0], y, s=30, marker='o')
    plt.xlabel('Feature 0')
    plt.ylabel('y')
    plt.title('Feature 0 vs Target')
    plt.show()


def train(X, y, lr, n_iters):
    w, b = np.zeros(X.shape[1]), 0.0
    losses = []
    for _ in range(n_iters):
        losses.append(loss(w, b, X, y))
        dw, db = grads(w, b, X, y)
        w, b = w - lr * dw, b - lr * db
    return w, b, losses

def loss(w, b, X, y):
    E = X @ w + b - y
    return np.mean(E ** 2)

def grads(w, b, X, Y):
    n_samples, _ = X.shape
    E = X @ w + b - Y
    G = 2 * E / (n_samples)
    dW = X.T @ G
    db = G.sum(axis=0)
    return dW, db


def numeric_grad_array(f, arr, h=1e-5):
    g = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        plus, minus = arr.copy(), arr.copy()
        plus[idx] += h
        minus[idx] -= h
        g[idx] = (f(plus) - f(minus)) / (2 * h)
    return g

def rel_err(a, n):
    return np.max(np.abs(a - n) / (np.abs(a) + np.abs(n) + 1e-12))

w_true, b_true, X, y = init(100, 5, 0)

# 1. Gradient check at a random point (not at the optimum, where grads ~ 0)
rng = np.random.default_rng(0)
w0, b0 = rng.normal(size=5), rng.normal()
dw_a, db_a = grads(w0, b0, X, y)
dw_n = numeric_grad_array(lambda w: loss(w, b0, X, y), w0)
db_n = numeric_grad_array(lambda b: loss(w0, b, X, y), np.asarray(b0))
print("rel err dw:", rel_err(dw_a, dw_n), " db:", rel_err(db_a, db_n))

# 2. Loss curves: baseline, bad feature scale, standardized, lr too high
def run(X, y, lr, n=1000):
    _, _, losses = train(X, y, lr, n)
    return losses

Xs = X.copy(); Xs[:, 0] *= 1000                       # one feature 1000x larger
Xstd = (Xs - Xs.mean(0)) / Xs.std(0)                  # standardized

with np.errstate(all="ignore"):                       # divergence overflows to inf
    curves = {
        "baseline lr=0.1": run(X, y, 0.1),
        "scaled lr=0.1 (diverges)": run(Xs, y, 0.1),
        "scaled lr=1e-7 (stalls)": run(Xs, y, 1e-7),
        "standardized lr=0.1": run(Xstd, y, 0.1),
        "baseline lr=1.0 (diverges)": run(X, y, 1.0),
    }
for name, c in curves.items():
    plt.semilogy(c, label=name)
plt.xlabel("iteration"); plt.ylabel("loss"); plt.legend(); plt.show()