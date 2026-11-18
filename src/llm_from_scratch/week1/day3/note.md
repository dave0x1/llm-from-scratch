# Week 1, Day 3: micrograd (autograd engine)

Date: 2026-11-18
Plan coverage: Day 3 (Value class, `+` and `*`, manual backward) plus the start of Day 4 (topological-sort backward).

## Goal

Build a scalar reverse-mode autodiff engine from scratch. Each `Value` remembers how it was computed, and `backward()` applies the chain rule through the computation graph in reverse.

## What I built

- `Value` class holding `value`, `grad`, `_prev` (children), `_op`, `label` and a `_backward` closure.
- Ops: `+`, `*`, `**` (int/float exponent), `sigmoid`, each with its own local-derivative `_backward`.
- `backward()` using a DFS topological sort, seeding `self.grad = 1.0` and running `_backward()` in reverse order.
- `trace` / `draw_dot` (graphviz) to render the graph with values and grads.
- Finite-difference check of the engine against a numerical reference.

## Core ideas

**Backprop = chain rule in reverse topological order.** Every node knows only its local derivative:

| op | local derivative | `_backward` |
|---|---|---|
| `out = a + b` | 1, 1 | `a.grad += out.grad`; `b.grad += out.grad` |
| `out = a * b` | b, a | `a.grad += b.value * out.grad`; `b.grad += a.value * out.grad` |
| `out = a ** n` | n * a^(n-1) | `a.grad += n * a.value**(n-1) * out.grad` |
| `out = sigmoid(a)` | s * (1 - s) | `a.grad += s * (1 - s) * out.grad` |

**Why `+=` and not `=`.** When a value feeds several downstream nodes, the gradients from the different paths must be summed (multivariable chain rule). For `L = x * x`, the correct `dL/dx` is `2x`. With `=`, the second write overwrites the first and gives `x`, half the right answer.

**Why topological order.** A node's gradient must be complete (every consumer already propagated into it) before the node propagates to its own children. The DFS visits a node's children first and appends the node after them, so reversing the list processes each node after all its consumers.

```python
def backward(self):
    topo, visited = [], set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)          # once per node, AFTER the children loop
    build_topo(self)
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

**Vanishing gradient from saturation.** In the first test graph, `g = 14`, and `sigmoid'(14) ~ 8e-7`, so every grad printed `0.0000`. A saturated sigmoid passes almost no gradient back, and the picture could not distinguish a correct engine from a broken one. Test with inputs near zero.

**Finite differences as the oracle.** Compare analytic grads against `(f(x+h) - f(x-h)) / 2h` on a function that only builds the graph and returns the value. The check does not rely on the backprop code, so it catches bugs in the engine itself.

## Bugs hit and what they taught

1. **`__add__` backward used `other.value` and omitted `out.grad`.** It borrowed the multiplication rule. The graph's numbers were exactly reproducible by hand from the bug (e.g. `e.grad = f.value = 1.0`, `a.grad = b.value = 7`), so a wrong picture can still be explained by the code if you trace it.
2. **`__pow__` had no `_backward`.** Silent failure: grads stay 0.0 and nothing complains.
3. **`o.backward` without parentheses.** A method reference, not a call. Symptom: every grad is exactly 0.0.
4. **`topo.append(v)` inside the `for child` loop.** Nodes were appended once per child (duplicates), leaves were never appended, and `_backward()` ran multiple times per node. Result: analytic/correct ratios of `a: 2x, b: 2x, c: 4x, d: 4x`, relative error 0.6. The exact multipliers depend on the iteration order of the `_prev` set, so it can differ between runs.

## Verification

Graph: `o = sigmoid((a + b) + c * d)` at `a=0.5, b=-0.2, c=0.3, d=2.0`.

| | grads (a, b, c, d) | rel err vs numeric |
|---|---|---|
| before topo fix | `[0.411, 0.411, 1.644, 0.2466]` | 0.6 |
| after topo fix | `[0.2055, 0.2055, 0.4110, 0.06165]` | 5.2e-11 |

Pass criterion: relative error `|a - n| / (|a| + |n|)` around 1e-8 or better; above about 1e-6 means a bug.

## Open items

- **Shared-node tests** (this graph has no reused nodes, so it cannot test accumulation):
  - `a*a + a` at `a=3`: expected `2a + 1 = 7`
  - `s = a + b; L = s * s`: expected `dL/da = dL/db = 2(a + b)`
- Reusable `gradcheck(f, x0)` helper.
- Missing ops: `__neg__`, `__sub__`, `__radd__`, `__rmul__`, `__truediv__` (as `self * other**-1`), optionally `tanh`.
- Optional: guard `math.exp(-x)` against `OverflowError` for very negative `x` (below about -709).
- Unrelated leftover from Day 2: one-sentence answer on why the scaled-feature run diverges at lr=0.1 but stalls at lr=1e-7.

## Next steps (plan Day 4, then Day 5)

1. Write `gradcheck(f, x0)` and use it on every op.
2. Add the missing ops, each checked with `gradcheck`.
3. Build `Neuron(n_in)`, `Layer(n_in, n_out)` and `MLP(n_in, [h1, ..., n_out])`, each with `parameters()`. Before running, predict the parameter count of `MLP(3, [4, 4, 1])` and the shape at each stage for one input of length 3. Then `gradcheck` the whole MLP on one example.
4. Day 5: training loop on a toy dataset. Break the learning rate and `zero_grad` on purpose. Because grads accumulate with `+=`, forgetting `zero_grad` makes one step's grads pile into the next, so predict the symptom first.

## References

- Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd": https://www.youtube.com/watch?v=VMj-3S1tku0
- micrograd repo : https://github.com/karpathy/micrograd