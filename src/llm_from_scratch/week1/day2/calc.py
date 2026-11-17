import math

def forward(w1, w2, b, x1, x2, t):
    # TODO: compute z, y_hat, L
    z = (w1 * x1) + (w2 * x2) + b
    y_hat = math.tanh(z)
    L = (y_hat - t) ** 2
    return L, z, y_hat

def backward(w1, w2, b, x1, x2, t):
    # TODO: call forward, then one line per step of your derivation
    _, _, y_hat = forward(w1, w2, b, x1, x2, t)
    delta = 2 * (y_hat - t) * (1 - y_hat ** 2)
    dw1 = delta * x1
    dw2 = delta * x2
    db = delta
    return dw1, dw2, db

def numeric_grad(f, params, h=1e-5):
    # f: list of params -> scalar loss
    # TODO: for each i, nudge params[i] by +h and -h, central difference
    grads = []
    for i in range(len(params)):
        plus = list(params)     # copy of the list
        minus = list(params)    # another copy
        plus[i] += h
        minus[i] -= h
        grads.append((f(plus) - f(minus)) / (2 * h))
    return grads  # list, same length as params

w1, w2, b, x1, x2, t = 0.5, -0.3, 0.1, 2.0, -1.0, 0.7

analytic = backward(w1, w2, b, x1, x2, t)
numeric = numeric_grad(lambda p: forward(p[0], p[1], p[2], x1, x2, t)[0], [w1, w2, b])

for name, a, n in zip(["w1", "w2", "b"], analytic, numeric):
    print(f"{name}: analytic={a:.8f} numeric={n:.8f} diff={abs(a - n):.2e}")