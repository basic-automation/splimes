# Benchmarks and conformance

What splimes 1.0 costs, how accurate each backend is, and how to reproduce both.

## Machine

| | |
|-|-|
| CPU | AMD Ryzen 9 7950X3D, 16 cores (SMT off), 124 GiB RAM |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, driver 610.57.04, Vulkan |
| OS | Linux 7.2 (Arch-based) |
| Toolchain | rustc 1.100.0-nightly (2026-09-22), release profile |
| Crates | splimes 1.0.0, wgpu 30.0.1, rayon 1.12.0, criterion 0.8 |
| Date | 2026-10-05 |

The machine had other work running during these runs (load average 15–20 on 16 cores),
so absolute figures, the `Parallel` and GPU ones most, are pessimistic and noisy; the
best-of-N tables are the steadier ones. Treat the numbers as relative, and rerun them on
your own hardware.

## Reproducing

```bash
cargo bench --bench interpolation               # the criterion tables below
cargo run --release --example calibrate         # the crossover table
cargo test --release --test contract -- --nocapture --test-threads 1   # the accuracy tables
```

Every benchmark interpolates a deterministic irregular series (a sine, knot spacing
jittered by up to 40%) onto a millisecond grid spanning it, so the whole grid is
interpolation, and returns `f64` unless the row says otherwise. Times are criterion
medians.

## Backends

Cubic, 4,096 input points, `run_f64`. Criterion medians.

| Grid points | `Cpu` | `Parallel` | `Gpu` f64 | `Gpu` f32 |
|------------:|------:|-----------:|----------:|----------:|
| 4,096 | 391 µs | 387 µs | 606 µs | 559 µs |
| 65,536 | 2.77 ms | 1.61 ms | 1.89 ms | 1.59 ms |
| 1,048,576 | 44.1 ms | 11.7 ms | 14.5 ms | 12.0 ms |
| 16,777,216 | 670 ms | 102 ms | 116 ms | 83 ms |

The GPU's kernel time is a small part of each call. Building the output (a timestamp,
a value and a provenance label per point, about 30 bytes of fresh memory) costs the same
whichever backend produced the values, and the GPU adds an upload of the grid times
(96-bit integers, prepared on the host) and a read-back. On this machine that leaves
the GPU level with rayon, or modestly ahead in `f32` at the largest sizes, and behind it
below a million points.

That is why `Backend::Auto` leaves the GPU off by default. `calibrate()` measures the
real crossover on the machine it runs on, in both precisions, and turns the GPU on where
it wins. Best of three per cell:

```text
cubic, 4,096 inputs, f64 values, best of 3

    points          cpu     parallel      gpu f64      gpu f32
      1024     134.64µs     133.95µs     383.99µs     270.95µs
      4096     400.27µs     396.69µs     420.60µs     409.46µs
     16384     906.93µs     901.29µs       1.06ms     586.04µs
     65536       2.87ms       1.49ms       2.72ms       2.34ms
    262144      12.66ms       5.21ms       4.59ms       8.00ms
   1048576      44.08ms       6.86ms       7.72ms       6.58ms
   4194304     166.74ms      22.01ms      25.51ms      17.77ms
  16777216     683.71ms      95.35ms     107.83ms      94.64ms

parallel from 1024 points, GPU f64 never, GPU f32 from 1048576 points
```

A backend wins from the smallest size at which it is faster at that size and every
larger one. On a machine this loaded the crossovers move between runs (another run put
the `f32` GPU from 256 Ki points); the built-in rayon threshold, 64 Ki points, is
deliberately conservative, because below it rayon only matches one thread on a quiet
machine and loses on a busy one.

**GPU start-up**, measured by `prewarm_gpu()` in a fresh process: 160–250 ms (opening
the device and compiling both kernels). `Auto` never pays it: it uses the GPU only once
the program has started it.

## Methods

4,096 inputs, 1,048,576 grid points, best of 15.

| Method | `Parallel` | `Cpu` |
|--------|-----------:|------:|
| `Linear` | 5.7 ms (184 M points/s) | 26.6 ms (39 M points/s) |
| `Quadratic` | 6.3 ms (166 M points/s) | 38.1 ms (28 M points/s) |
| `Cubic` | 8.2 ms (128 M points/s) | 42.7 ms (25 M points/s) |
| `Polynomial(5, None)` | 9.5 ms (110 M points/s) | 61.1 ms (17 M points/s) |
| `Polynomial(8, None)` | 11.7 ms (90 M points/s) | 87.8 ms (12 M points/s) |

Consecutive grid points that share a window reuse its Lagrange weights; each point still
takes its `degree + 1` time differences exactly, from integer nanoseconds, which is what
keeps the error bound independent of how the knots are spaced.

Since 1.0.0, when the knots and the grid all lie within about 146 years (2⁶² ns) of the
first knot, which is nearly every series, those differences are taken in `i64` instead of
`i128`, with bit-identical results. Same benchmark, before (`main`) and after, in one
session, criterion medians:

| Method | `Parallel` before | `Parallel` after | `Cpu` before | `Cpu` after |
|--------|-------:|------:|-------:|------:|
| `Linear` | 5.79 ms | 4.85 ms | 25.5 ms | 17.7 ms |
| `Quadratic` | 6.44 ms | 5.58 ms | 34.0 ms | 24.9 ms |
| `Cubic` | 7.72 ms | 6.14 ms | 39.3 ms | 27.0 ms |
| `Polynomial(5, None)` | 8.97 ms | 6.77 ms | 56.7 ms | 36.4 ms |
| `Polynomial(8, None)` | 10.7 ms | 8.37 ms | 81.3 ms | 49.6 ms |

Measured 2026-10-08 on the machine above with rustc 1.99.0 (stable), otherwise idle.
`cargo bench --bench interpolation -- methods` reproduces it (the `cpu/` rows are `Cpu`).

## `BigDecimal` at the edges

The same cubic interpolation through `run` (`BigDecimal` in and out) and `run_f64`,
on `Parallel`, 4,096 inputs. Criterion medians.

| Grid points | `f64` | `BigDecimal` | Ratio |
|------------:|------:|-------------:|------:|
| 65,536 | 2.26 ms | 5.90 ms | 2.6× |
| 1,048,576 | 8.71 ms | 31.4 ms | 3.6× |

Converting each interpolated `f64` to its shortest `BigDecimal` is the cost: about
20 ns per point of wall time on this (loaded) 16-core machine. If your values start or
end as `f64`, use `run_f64`.

## Input size

Cubic, `Parallel`, 1,048,576 grid points. Criterion medians.

| Input points | Time |
|-------------:|-----:|
| 16 | 9.0 ms |
| 4,096 | 8.2 ms |
| 1,048,576 | 46.4 ms |

The knot search is logarithmic, so input size barely matters until there are about as
many inputs as grid points: then preparing the input (sorting, de-duplicating and
normalising a million points) and a fresh window per grid point dominate. (Measured on
1.0.0, when preparation was single-threaded; see below.)

## Preparing input

Since 1.0.0, every backend but `Cpu` prepares inputs of 16 Ki points or more on rayon's
pool, and converts each value to `f64` once instead of twice. A million inputs on a
one-point grid, so preparation is the whole cost; "shuffled" is the same series in a
scrambled order. Criterion medians, before (`main` at 1.0.0) and after, in one session.

| 1,048,576 inputs | `Cpu` before | `Cpu` after | `Parallel` before | `Parallel` after |
|------------------|-------:|------:|-------:|------:|
| `f64`, in order | 22.4 ms | 16.2 ms | 21.3 ms | 13.9 ms |
| `f64`, shuffled | 70.2 ms | 47.3 ms | 69.7 ms | 24.2 ms |
| `BigDecimal`, shuffled | 371 ms | 187 ms | 368 ms | 43.4 ms |

And the same change in whole calls, `Parallel` cubic as in [Input size](#input-size):

| | Before | After |
|-|-------:|------:|
| 1,048,576 inputs, 1,048,576 grid points | 41.5 ms | 26.3 ms |
| `Auto`, 32 `BigDecimal` inputs, 60 grid points | 16.8 µs | 12.4 µs |

Measured 2026-10-08 on the machine above with rustc 1.99.0 (stable), otherwise idle (95%
idle outside the benchmark's own threads). Results are unchanged, bit for bit.

```bash
cargo bench --bench interpolation -- "prepare|inputs|small"
```

## Small calls

`Auto`, cubic, 32 `BigDecimal` inputs: a typical query. Criterion medians.

| Grid points | Time |
|------------:|-----:|
| 60 | 19 µs |
| 1,440 | 184 µs |

## Conformance

The worst error measured by `tests/contract.rs` against an exact 60-digit reference, in
the units of the published contract: `(|got − exact| − ε·|exact|) / (range · Λ(t))`, with
`Λ(t)` the window's Lebesgue function at the grid point (see the crate docs' "Numerical
contract"). 258 series — 256 randomised (knot spacings from microseconds to 30 days,
regular and irregular, values tiny, huge and large-offset), a dense burst at the end of a
sparse series, and geometric gaps — each on five grids: across both edges (two edge
spacings out), the middle, the densest stretch, and 100 edge spacings out. CI prints the
same report on every run: for lavapipe (f64 and f32), which the Linux job requires, and
for the Windows and macOS runners' adapters (WARP and Metal, f32) when they expose one.
The 1.0.0 release run's results are [below](#ci-runners).

**CPU** (`Cpu` and `Parallel`) and **NVIDIA RTX 4070 Ti SUPER, f64**:

| Method | CPU inside | CPU outside | GPU f64 inside | GPU f64 outside | Published bound |
|--------|-----------:|------------:|---------------:|----------------:|------:|
| `Linear` | 2.1e-16 | 2.8e-16 | 2.1e-16 | 2.2e-16 | 1e-13 |
| `Quadratic` | 2.0e-16 | 2.8e-16 | 2.0e-16 | 2.2e-16 | 1e-13 |
| `Cubic` | 2.7e-16 | 2.8e-16 | 2.7e-16 | 2.0e-16 | 1e-13 |
| `Polynomial(4, 0.5)` | 3.0e-16 | 2.8e-16 | 3.0e-16 | 2.6e-16 | 1e-13 |
| `Polynomial(5, None)` | 4.1e-16 | 2.8e-16 | 4.1e-16 | 2.6e-16 | 1e-13 |
| `Polynomial(8, None)` | 5.2e-16 | 3.1e-16 | 5.2e-16 | 3.1e-16 | 1e-13 |

Measured this way the error is flat across methods, degrees and distance: a couple of
ulps, as the Lagrange form allows. (Measured against the value range alone, the
geometric-gap series' degree-8 interpolant errs by 10¹⁰ × the range — because the
polynomial itself is that large between its knots, and any evaluation from rounded
inputs inherits it. Λ is the honest yardstick.) The two-million-knot series is within
1.1e-16 (linear) to 8.4e-16 (degree 5) on both CPU and GPU.

**NVIDIA RTX 4070 Ti SUPER, f32:**

| Method | Inside | Outside | Published bound |
|--------|-------:|--------:|------:|
| `Linear` | 1.4e-7 | 2.0e-7 | 1e-5 |
| `Quadratic` | 1.4e-7 | 1.9e-7 | 1e-5 |
| `Cubic` | 1.9e-7 | 1.9e-7 | 1e-5 |
| `Polynomial(4, 0.5)` | 2.0e-7 | 1.9e-7 | 1e-5 |
| `Polynomial(5, None)` | 2.3e-7 | 1.9e-7 | 1e-5 |
| `Polynomial(8, None)` | 3.0e-7 | 1.9e-7 | 1e-5 |

The two-million-knot series is within 5.7e-8 (linear) to 5.1e-7 (degree 5) in f32.

### CI runners

From the `v1.0.0` release run, on GitHub's hosted runners. These are software and
virtualised adapters, not GPUs you would deploy on, but they run each platform's own
shader compiler: Metal, for one, compiles shaders with fast math, which is what 1.0's
integer time representation defends against. Of the three, only lavapipe has f64;
its f64 table matches the CPU table above digit for digit.

| Method | Metal inside | Metal outside | lavapipe inside | lavapipe outside | WARP inside | WARP outside | Published bound |
|--------|------:|------:|------:|------:|------:|------:|------:|
| `Linear` | 1.5e-7 | 2.0e-7 | 1.3e-7 | 2.6e-7 | 1.3e-7 | 2.6e-7 | 1e-5 |
| `Quadratic` | 1.4e-7 | 1.9e-7 | 1.4e-7 | 1.9e-7 | 1.5e-7 | 1.9e-7 | 1e-5 |
| `Cubic` | 2.1e-7 | 1.9e-7 | 1.9e-7 | 1.9e-7 | 1.9e-7 | 1.9e-7 | 1e-5 |
| `Polynomial(4, 0.5)` | 2.0e-7 | 1.9e-7 | 2.0e-7 | 1.9e-7 | 2.0e-7 | 1.9e-7 | 1e-5 |
| `Polynomial(5, None)` | 2.0e-7 | 1.9e-7 | 2.3e-7 | 1.9e-7 | 2.3e-7 | 1.9e-7 | 1e-5 |
| `Polynomial(8, None)` | 3.1e-7 | 1.9e-7 | 3.0e-7 | 1.9e-7 | 3.0e-7 | 1.9e-7 | 1e-5 |

f32 throughout. The two-million-knot series is within 5.7e-8 (linear) to 5.1e-7
(degree 5) on all three (4.9e-7 on Metal). The adapters, as wgpu reports them: **Metal**
is the Apple Paravirtual device, the virtualised GPU of the `macos-latest` runner;
**lavapipe** is llvmpipe (LLVM 20.1.2, 256 bits), Vulkan, on `ubuntu-latest`; **WARP** is
the Microsoft Basic Render Driver, DX12, on `windows-latest`.

The matrix still lacks AMD and Intel GPUs, and Apple silicon outside a virtual machine;
see the roadmap.
