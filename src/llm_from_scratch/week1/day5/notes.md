# Week 1, Day 5: Training loop, and breaking it on purpose

Date: 2026-11-19
Code: `src/llm_from_scratch/week1/day5/nn_test.py`

## What I did
- Trained `MLP(3, [4, 4, 1])` (41 parameters) on a 4-sample toy dataset using only my micrograd engine.
- Loop per step: forward pass, loss as the sum of squared errors (a single `Value`), zero grads, `backward()`, SGD update `p.value -= lr * p.grad`.
- Dataset: `xs = [[2,3,-1],[3,-1,0.5],[0.5,1,1],[1,1,-1]]`, `ys = [1,-1,-1,1]`.
- Ran experiments A to D. Found and fixed a bug in my `Neuron` along the way.

## The bug I found
`Neuron.__call__` applied `tanh` unconditionally and ignored the `nonlin` flag, so the last layer was never linear even though `MLP` passed `nonlin=False` correctly.

```python
out = act.tanh() if self.nonlin else act   # the fix
```

- The Day 4 parameter count test (41) checks structure, not behavior, so it missed this.
- To add: a regression test that scales the last layer's params by 100 and asserts `|output| > 1` (tanh can never exceed 1).
- Everything labeled "tanh output" below was run on the buggy network. The lr = 0.01 and lr = 2.0 runs labeled "linear output" were run after the fix.

## Results

| Exp | Change | What I saw | Why |
|---|---|---|---|
| A (tanh output) | lr = 0.5 | Overshoots in the first steps, then settles near 2.37e-8 and barely moves | See "tanh saturation" below |
| A (tanh output) | lr = 1.0, 2.0 | Loss pinned at ~8.0, all outputs exactly -1.0, `max abs grad` ~1e-13 | Saturation: stuck at a dead point, not converged |
| A (linear output) | lr = 2.0 | Loss explodes, `OverflowError` at step ~82 | Unbounded output, step larger than `2/lambda` |
| B | lr = 1e-5 | ~100,000 steps to reach 0.05 (my earlier observation, buggy network) | Small-lr regime: lr x steps roughly constant. Not rerun at 1e-3 |
| C | No zeroing | Grads accumulate. Seeded test: `off[1] = on[0] + on[1]` (0.4572 + 0.5301 = 0.9873) | See "Accumulation" below |
| D | Zero after `backward()` | Flat loss line | Every update is `p.value -= lr * 0` |
| Baseline (linear output) | lr = 0.01, zeroing on | 6.5 -> plateau around 2.5 to 2.8 until step ~40 -> 0.002 at step 99, still falling | Plain SGD, slower than the buggy no-zeroing runs |

## Key concepts to remember

**Accumulation and zeroing.**
- Within one `backward()`, `+=` is required: a node reached by several paths, including one parameter used in all 4 samples, gets one contribution per path. `=` would keep only the last.
- Across steps, the graph is rebuilt every forward pass, but parameters are persistent leaf nodes, so their `.grad` carries stale state. Without a reset, `p.grad` is a sum of gradients taken at different parameter values.
- So the reset has to happen outside the engine (PyTorch: `zero_grad()`).

**Tanh saturation.**
- `d tanh/dz = 1 - tanh^2`, which is ~0 for large `|z|`. One huge step pushes pre-activations onto the flat tails, gradients vanish, and nothing can pull the model back.
- A flat loss is not the same as convergence. Check the gradient too.
- Derived, not yet verified numerically: near the target, MSE + tanh gives `dL/dz ~ -4e^2` (where `e = 1 - y`), so `dL/dstep ~ -L^2` and `L ~ 1/t`. That's the slow crawl at lr = 0.5.
- Cross-entropy cancels the `(1 - y^2)` factor (revisit on Day 7).

**Step size and stability.**
- Gradient descent's linear approximation only holds when `lr x curvature` is small. On a quadratic, `lr > 2/lambda_max` diverges.
- With tanh output, divergence turned into saturation (bounded output). With a linear output it was a true blow-up.
- Measured at lr = 2.0: loss ratio between steps is 6241 = 79^2, so the error is multiplied by -79 per step. That implies `lambda ~ 40` and `lr < 0.05` for stability.
- Hypothesis (not yet verified): the hidden layer collapsed to one +/-1 pattern `h`, so `|phi|^2 = 4 + 1 = 5` and `lambda = 2 x 4 x 5 = 40`.

**Why the loss is a single `Value`.**
- Reverse-mode AD starts at one scalar root (`dL/dL = 1`) and gives the gradient of that one output w.r.t. all 41 inputs in one backward pass. A list of losses has no single root.

## Process lessons
- Compare runs from the same init: seed the RNG before building the model (the first accumulation test was invalid because the inits differed).
- Don't trust a prediction over a measurement (my "frictionless momentum should oscillate" prediction conflicted with the observed convergence).
- Name variables in the order they're bound (`zip(ypred, ys)`, not `zip(ys, ypred)`).

## Open items
- [ ] Add the regression test for the linear last layer
- [ ] Experiment B at lr = 1e-3 (expect on the order of 100 to 200 steps to reach 0.05)
- [ ] Verify the lambda ~ 40 hypothesis: print the hidden activations for all 4 inputs after ~10 steps at lr = 2.0 (expect four identical +/-1 rows)
- [ ] Revisit the accumulation-as-momentum puzzle (zeroing off converged to ~1e-16 on the buggy network) in Week 2 with momentum and Adam
- [ ] Eigenvalue lr-threshold check (`lr < 2/lambda_max`), deferred since Day 2