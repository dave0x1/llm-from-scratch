# Week 1, Day 2 notes: linear regression from scratch

Date: 2026-11-17
Plan coverage: Day 1 (NumPy shapes, vectorization) and Day 2 (gradients verified against finite differences).

## What I did

- Worked through the Google ML Crash Course up to the Data section, then paused before its neural networks section to build instead.
- Implemented linear regression with gradient descent in plain NumPy (no sklearn model, no autograd).
- Generated synthetic data from a known `w_true`, `b_true` so the correct answer is known.
- Refactored into `init`, `loss`, `grads`, `train` and `numeric_grad_array`.
- Ran with `noise=0`: after 1000 iterations at lr=0.1, `max|w_hat - w_true|` reached about 3.6e-15 (float64 floor).

## Core math and shapes

```
X: (N, D)   w: (D,)   b: scalar   y: (N,)
E      = X @ w + b - y          (N,)   residuals
L      = mean(E**2)             scalar
dL/dw  = (2/N) * X.T @ E        (D,)
dL/db  = (2/N) * E.sum()        scalar
```

The gradient must use the same normalization as the loss (`1/N`). Dividing by `N * D` made it D times too small.

## Key ideas worth remembering

**Synthetic data as an oracle.** With `w_true`, `b_true` known, the test is `w_hat ~ w_true`. With `noise=0` recovery should be essentially exact, and with noise the best possible test MSE is about `noise**2`. Real data has no answer key.

**Finite differences.** `dL/dw_i ~ (L(w + h*e_i) - L(w - h*e_i)) / (2h)` with h around 1e-5. It only calls `loss`, so it is a slow but trustworthy reference for the fast, error-prone analytic gradient. Cost is `2(D+1)` loss evaluations per gradient, which is why it is a test tool and not a training method. Pass criterion: relative error `|a - n| / (|a| + |n|)` around 1e-8 or better; above about 1e-6 means a bug. Check at a random point, not at the optimum where gradients are near zero.

**Convergence of gradient descent on a quadratic.** The Hessian is `H = (2/N) X.T @ X` (with a ones column appended for `b`). Along each eigendirection the error shrinks by `(1 - lr * lambda)` per step, so:
- stable only if `lr < 2 / lambda_max`
- the loss is a straight line on a log axis (geometric decay)
- speed is set by the smallest eigenvalues, stability by the largest

**Feature scaling.** One feature scaled by 1000 makes `lambda_max` about 1e6 times larger, so stability forces a tiny lr and every other direction barely moves (lr=0.1 diverges, lr=1e-7 stalls). Standardizing brings the eigenvalues back together and restores fast convergence.

## Bugs hit and what they taught

1. **`w_true` was all zeros.** y was pure noise, so even a broken model would pass. Test data must contain signal.
2. **`return` indented inside the `for` loop.** Training ran one step. Symptom: `w_hat / w_true` was about 0.2 (one step at lr=0.1 with Hessian about 2).
3. **Gradient divided by `N * D` instead of `N`.** Silent: it still converged, just with effective lr/D (8.2e-12 vs 3.6e-15 after 1000 iterations). Constant-factor gradient errors do not crash anything. This is exactly what the finite-difference check catches.
4. **Loss plot clipped.** Diverging runs overflow to `inf`/`nan` and the log axis fell back to a default window. Fix: mask non-finite values and set explicit `ylim`.

## Final reference code

```python
def loss(w, b, X, y):
    E = X @ w + b - y
    return np.mean(E ** 2)

def grads(w, b, X, y):
    n = X.shape[0]
    G = 2 * (X @ w + b - y) / n
    return X.T @ G, G.sum()

def numeric_grad_array(f, arr, h=1e-5):
    g = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        plus, minus = arr.copy(), arr.copy()
        plus[idx] += h
        minus[idx] -= h
        g[idx] = (f(plus) - f(minus)) / (2 * h)
    return g
```

`numeric_grad_array` works on any array shape, so it carries over to micrograd and later layers. For scalar `b`, pass `np.asarray(b)`.

## Open items

- Report the gradient-check relative errors (expected about 1e-8 to 1e-10).
- Write one sentence: why the scaled-feature run diverges at lr=0.1 but stalls at lr=1e-7.
- Fix the loss plot (mask `inf`/`nan`, set `ylim`).
- Optional: compute `lambda_max` of the Hessian and check that the divergence threshold matches `2 / lambda_max` (baseline and scaled-feature data).
- Logistic regression (sigmoid, binary cross-entropy, its gradient) deferred to Day 7 with softmax and cross-entropy.
- MLCC neural networks section: read after micrograd.

## Next steps

1. **Micrograd part 1 (plan Day 3).** Watch Karpathy's "The spelled-out intro to neural networks and backpropagation: building micrograd" (https://www.youtube.com/watch?v=VMj-3S1tku0), pausing to implement each piece yourself in a blank file: `Value` class, `+` and `*`, manual `_backward`.
2. **Use `numeric_grad_array` as the test harness** for every op you add. Each new `_backward` gets a finite-difference check.
3. **Before starting:** derive by hand the gradients of `L = (a*b + c)**2` with respect to `a`, `b`, `c`, then verify numerically. This is the chain rule that micrograd automates.
4. **Plan status:** started 2026-11-11, so the schedule is slightly behind. Plan Days 3 and 4 (micrograd, topological sort, Neuron/Layer/MLP) are the priority, and Day 7 is buffer if needed.