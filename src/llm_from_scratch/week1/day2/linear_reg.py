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
    n_samples, n_features = X.shape
    w_hat = np.zeros(n_features)
    b_hat = 0
    for _ in range(n_iters):
        dw, db = grads(w_hat, b_hat, X, y)
        w_hat = w_hat - lr * dw
        b_hat = b_hat - lr * db
    return w_hat, b_hat

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

w_true, b_true, X, y = init(100, 5, 0)
w_hat, b_hat = train(X, y, 0.1, 1000)
print(f"w_true: {w_true}, w^ : {w_hat} \n b_true: {b_true}, b^: {b_hat}")
print(np.abs(w_hat - w_true).max())