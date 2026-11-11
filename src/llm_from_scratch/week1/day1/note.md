# Week 1, Day 1: Environment, Intuition, Shapes, Matmul

Date: 2026-11-11

## What was done

- Set up the `llm-from-scratch` project with `uv` (Python 3.12; numpy, matplotlib; dev: ruff, pytest) and committed `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`.
- Watched 3Blue1Brown's Neural Networks series (ch. 1-4).
- NumPy shapes and broadcasting: predict-then-run exercises, strides.
- Matmul by triple loop vs vectorized, with a pytest equivalence test and timings.

## Key takeaways

### Neural net basics

- One neuron: `a = sigma(w . x + b)`. A layer is `a = sigma(W x + b)` with `W: (out, in)`.
- Params for 784-16-16-10: `784*16+16 = 12,560`, `16*16+16 = 272`, `16*10+10 = 170`, total **13,002**.
- Nonlinearity is required because composing linear maps is linear: `W2(W1 x) = (W2 W1) x`. Without it, depth collapses to a single layer.
- The cost is a function of the **parameters** (the data is fixed). `-grad C` is the direction of steepest descent.
- Backprop efficiently computes `grad C` with respect to every parameter by applying the chain rule backward through the computation. Gradient descent then uses that gradient to update the parameters.

### Broadcasting

Rule: align shapes **from the right**. Two dims are compatible if they are equal or one of them is 1. Missing leading dims count as 1. A size-1 dim is stretched virtually (stride 0, no copy).

| Expression | Result |
|---|---|
| `(3,1) + (1,4)` | `(3,4)` |
| `(5,3) + (3,)` | `(5,3)` |
| `(5,3) + (5,)` | **error**: right-aligned, so 3 vs 5 mismatch. Fix: `f[:, None]` to make it `(5,1)` |
| `(2,3,4) @ (4,5)` | `(2,3,5)` (the 2D operand is broadcast over the batch dim) |
| `(2,3,4) @ (2,4,5)` | `(2,3,5)` (batched matmul) |
| `k.sum(axis=1)` on `(2,3,4)` | `(2,4)` |
| `k.sum(axis=(0,2), keepdims=True)` | `(1,3,1)` |

`@` multiplies the last two dims as matrices and broadcasts everything before them as batch dims.

Strides: `x = np.arange(12).reshape(3,4)` (int64) has strides `(32, 8)` bytes. `x.T` has `(8, 32)` and is a **view**: transposing just swaps strides, no data moves. Consequence: many ops are free, but a transposed array is no longer contiguous, which can make later ops slower.

PL angle: shapes behave like a tiny type system that NumPy checks at runtime. Broadcasting is implicit coercion, convenient but a classic source of silent bugs (e.g. `(5,) - (5,1)` gives `(5,5)`, not an error).

### Matmul

- `(m,k) @ (k,n)` costs `m*k*n` multiply-adds (`2mkn` FLOPs). Square case: O(n^3).
- Why the loop version is orders of magnitude slower:
  1. Interpreter overhead on every scalar op (boxed floats, bytecode dispatch).
  2. No SIMD or vectorization.
  3. Poor cache behavior: the inner loop walks down a column of `B`.
  4. NumPy calls an optimized BLAS: blocked for cache, vectorized, multithreaded.
- C stretch: with row-major storage, `ikj` order has an inner loop over `j` that streams through contiguous memory in both `B` and `C`, while `ijk` strides by `n` through `B` and causes cache misses.

### Habit to keep

Write the expected shape in a comment **before** running any tensor code. Most transformer bugs are shape bugs.

## Timings (fill in)

| n | loops (s) | vectorized (s) | ratio |
|---|---|---|---|
| 64 | | | |
| 128 | | | |
| 256 | | | |

## Gaps and things I got wrong (fill in)

- Shape predictions I got wrong:
- Concepts I could not explain simply:

## Cheat sheet

```bash
uv add <pkg>            # add dependency (updates pyproject.toml + uv.lock)
uv add --dev <pkg>      # dev-only dependency
uv run python file.py   # run inside the project env
uv run pytest           # run tests
uv sync                 # rebuild env from the lockfile
```

## Next: Day 2

Linear algebra and calculus essentials; hand-computed gradient verified with finite differences.