# Week 1, Day 4: micrograd part 2 (2026-11-19)

**Goal:** harden the `Value` engine, then build Neuron / Layer / MLP on top of it.
**Status:** done. Finite-difference check on a first-layer weight matches `.grad`.

---

## 1. Hardening the `Value` engine

Bugs found in review and fixed:

| Bug | Cause | Fix |
|---|---|---|
| `3 - Value(2)` returned the wrong sign | `__rsub__` computed `self - other` instead of `other - self` | `return (-self) + other` |
| `1 / x` raised `TypeError` | `__rtruediv__` was missing | `return other * self ** -1` |
| `tanh` overflowed for large `x` | `(exp(2x) - 1) / (exp(2x) + 1)` overflows near `x > 354` | use `math.tanh` |
| `sigmoid` overflowed for very negative `x` | `exp(-x)` overflows near `x < -709` | branch on the sign of `x` so `exp` only sees a non-positive argument |
| `backward()` would hit `RecursionError` on deep graphs | recursive topological sort; `sum(losses)` over N samples builds a chain of depth N | iterative post-order DFS with an explicit stack of `(node, children_done)` |

Other cleanups:
- All backward closures read `self.value`, `other.value`, `out.value` **at backward time**, with no forward-time locals. This also keeps `x * x` correct, since both operands are the same object.
- Consistent naming (`_backward`, lowercase op labels), and `__repr__` shows the grad.

**Sanity tests that caught the bugs:** `a + a` gives grad 2, `a * a` gives grad 2a, `3 - x` gives grad -1, `6 / x` gives grad `-6/x^2`, stability at `tanh(1000)` and `sigmoid(+/-1000)`, and a 100k-long addition chain.

### Concepts worth keeping
- **Gradient accumulation (`+=`)** is what makes a variable used twice work (`a + a`). It's also why grads must be zeroed between steps (Day 5).
- **Desugaring to a smaller core.** `/` is `* x**-1`, `-` is `+ (-x)`, and `neg` is `* -1`. The primitives (`+`, `*`, `**`, `exp`) carry the derivative rules, and the rest comes for free. This is the same idea as lowering surface syntax to a small core language.
- **Limits that are conventions, not bugs:** `relu'(0) = 0` (matches PyTorch), and `x ** -1` at `x = 0` raises `ZeroDivisionError` in the forward pass.

---

## 2. Neuron / Layer / MLP

### Parameter count for `MLP(3, [4, 4, 1])`
- Per layer: `(n_in + 1) * n_out` (the +1 is the bias)
- L1: (3+1)\*4 = 16, L2: (4+1)\*4 = 20, Out: (4+1)\*1 = 5, **total = 41**
- Checked by hand (12+16+4 weights, 4+4+1 biases) and with `len(mlp.parameters())`.

### Shapes (scalar engine, lists of `Value`)
`3 -> 4 -> 4 -> 1`. A neuron maps a length-`n_in` list to one scalar. A layer maps `n_in -> n_out` (a list). In matrix form `W1` is `(4, 3)` with `out = W @ x + b`; PyTorch's `nn.Linear` stores `(out, in)`.

### Output activation matches the target range
| Target | Output activation |
|---|---|
| Unbounded real (regression) | none (identity) |
| Probability in (0, 1) | sigmoid |
| Labels in {-1, +1} | tanh |
| Many classes | none, then softmax (Day 7) |

`tanh` or `sigmoid` on an unbounded target puts a hard ceiling on the output. `relu` has a floor at 0, and a neuron whose pre-activation is negative gets gradient exactly 0. Frameworks usually output raw logits and fold the sigmoid or softmax into the loss for numerical stability.

### Review findings on my MLP code
1. **Hiding graph construction inside NumPy.** `self.w @ x` worked only through NumPy's object-dtype matmul fallback. It's fragile and obscures the graph. Fix: `sum((wi * xi for wi, xi in zip(self.w, x)), self.b)`.
2. **`Layer` unwrapped single outputs.** It crashed for any single-neuron layer that isn't last (e.g. `MLP(3, [4, 1, 4])`). Fix: `Layer` always returns a list; `MLP.__call__` unwraps once at the end.
3. **`nonlin` flag was stored but never read.** `Neuron.__call__` always applied `tanh`, so the output layer was still squashed. Test: set the last bias to 10 and assert the output exceeds 1. Fix: branch on the flag, and pass it through the constructors (`Layer(n_in, n_out, nonlin)`) instead of mutating neurons after the fact, so the last-layer rule lives in one place.

### Gradient-flow expectations
- After **one** `backward()` on a fresh net with `tanh` hidden layers, all 41 parameters have nonzero grad (`tanh' > 0` everywhere).
- With `relu`, some grads can be exactly 0: a neuron with negative pre-activation has local derivative 0, so everything feeding it gets 0 (dead ReLU).
- The second `backward()` *accumulates* onto stale grads, which is why `zero_grad()` is needed.

---

## 3. Open items / to do before or during Day 5
- [ ] Verify empirically: count of nonzero grads for `tanh` (expect 41/41) vs `relu` across a few seeds, and explain the missing ones.
- [ ] Seed the RNG (`random.seed(...)` or `np.random.default_rng(seed)`); current runs aren't reproducible.
- [ ] Add `MLP.zero_grad()`.
- [ ] Optional: set bias init to 0.
- [ ] Remove the unused `draw_dot` import where it isn't needed.

## 4. Day 5 preview: training loop
- Toy dataset: four 3-feature inputs with +/-1 targets.
- Loop: forward, loss (sum of squared errors), `zero_grad`, `backward`, SGD update.
- **Write predictions first**, then run the experiments:
  1. Skip `zero_grad`: does the loss diverge, stall, or behave erratically?
  2. Learning rate of 10: what happens to the loss curve?