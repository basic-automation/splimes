# Benchmarks and conformance

What splimes 1.0 costs, how accurate each backend is, and how to reproduce both.

## Machine

| | |
|-|-|
| CPU | AMD Ryzen 9 7950X3D, 16 cores (SMT off), 124 GiB RAM |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, driver 610.57.04, Vulkan |
| OS | Linux 7.2 (Arch-based) |
| Toolchain | rustc 1.100.0-nightly (2026-09-22), release profile |
| Crates | splimes 1.0.0 (1.1.0 in the "after" columns), wgpu 30.0.1, rayon 1.12.0, criterion 0.8 |
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

On a machine with more than one GPU, or to measure another graphics API, choose the
adapter with wgpu's variables, for example
`WGPU_BACKEND=vulkan WGPU_ADAPTER_NAME="rtx 4070"` in front of any of these.

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

In 1.1.0, when the knots and the grid all lie within about 146 years (2⁶² ns) of the
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

Building the output itself got cheaper too: each output timestamp used to cost a
`checked_add_signed` in chrono, about 10 ns; it is now built from integer POSIX seconds
on a calendar date computed once per day, giving the same instants. Starting from the
"after" column above, in a later session. `Parallel` medians moved by up to 21% between
the two sessions (`Polynomial(8, None)`: 8.37 ms above, 10.1 ms here), so read the
`Parallel` columns of both tables as indicative; the `Cpu` ones agree within 2%:

| Method | `Parallel` before | `Parallel` after | `Cpu` before | `Cpu` after |
|--------|-------:|------:|-------:|------:|
| `Linear` | 5.03 ms | 4.76 ms | 18.0 ms | 13.2 ms |
| `Cubic` | 6.70 ms | 5.93 ms | 27.5 ms | 24.1 ms |
| `Polynomial(8, None)` | 10.1 ms | 8.24 ms | 49.7 ms | 46.4 ms |

The 1 Mi-point `f64` call in [`BigDecimal` at the edges](#bigdecimal-at-the-edges) went
from 6.98 to 5.52 ms. Two smaller cases moved within their noise: the 64 Ki-point
`BigDecimal` call (3.77 → 4.00 ms, its `f64` twin ±11%) and the 60-point `Auto` call
(10.4 → 10.6 µs).

### Output assembly in one pass (1.2.0)

Output assembly used to fill all three output columns with placeholders and then
overwrite them. It now writes each column once, straight into its allocation, with
rayon's indexed `collect_into_vec` / `unzip_into_vecs`, and the provenance walk compares
`i64` offsets whenever the kernel does. Before (`main` at 1829232, 1.1.0) and after, the
same benchmarks, interleaved: three rounds of before, after, after, before, criterion
medians, then the median and the best of the runs. Two runs taken while another
process's `rustc` held 9–13 cores are left out, so each cell is 5 or 6 runs.

| Benchmark | Before, median | After, median | Before, best | After, best |
|-----------|---------:|---------:|---------:|---------:|
| `backends/parallel`, 65,536 points | 0.94 ms | 0.77 ms | 0.83 ms | 0.55 ms |
| `backends/parallel`, 1,048,576 points | 5.65 ms | 5.11 ms | 5.46 ms | 4.68 ms |
| `backends/parallel`, 16,777,216 points | 70.8 ms | 53.5 ms | 68.5 ms | 52.2 ms |
| `backends/parallel`, 4,096 points | 0.139 ms | 0.144 ms | 0.138 ms | 0.143 ms |
| `backends/cpu`, 1,048,576 points | 23.0 ms | 22.7 ms | 22.7 ms | 21.4 ms |
| `backends/cpu`, 16,777,216 points | 343 ms | 343 ms | 337 ms | 341 ms |
| `methods`, `Linear` / `Quadratic` / `Cubic` | 4.50 / 5.78 / 5.53 ms | 3.90 / 4.88 / 5.47 ms | 4.11 / 5.20 / 5.10 ms | 3.67 / 4.45 / 4.78 ms |
| `methods/cpu`, `Linear` / `Quadratic` / `Cubic` | 13.4 / 21.6 / 23.2 ms | 13.4 / 20.5 / 22.7 ms | 12.2 / 20.0 / 21.7 ms | 13.0 / 20.3 / 22.5 ms |

So `Parallel` calls of 64 Ki points and more take 10–25% less time (the medians), and
`Cpu` is unchanged within this session's run-to-run spread (±6%). A 4,096-point `Parallel` call costs about 5 µs more
(two parallel passes instead of one); `Auto` runs grids that small on `Cpu`.

Measured 2026-10-10, 00:29–01:06, on the machine above with rustc 1.99.0 (stable),
load average 1.4 before the runs, with no other process above 40% of a core apart from
the excluded `rustc` window and brief blips (Steam, Prowlarr). `cargo bench --bench
interpolation -- backends/parallel`, `-- backends/cpu` and `-- methods` reproduce it.

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

In 1.1.0, every backend but `Cpu` prepares inputs of 16 Ki points or more on rayon's
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

`Auto`, cubic, 32 `BigDecimal` inputs: a typical query. Criterion medians, measured on
1.0.0 on the loaded machine; [Preparing input](#preparing-input) has the 60-point call on
an idle machine, before and after 1.1.0.

| Grid points | Time |
|------------:|-----:|
| 60 | 19 µs |
| 1,440 | 184 µs |

## Streaming

Time to first output, cubic, 4,096 inputs, chunks of 262,144 grid points. `run_f64`
returns nothing until the whole grid is done; `stream_f64` and `stream_epoch` (epoch
microseconds in, ticks out) hand over each chunk as it finishes. Criterion medians, 10
samples.

| Grid points | Backend | `run_f64` | `stream_f64` first | `stream_f64` total | `stream_epoch` first | `stream_epoch` total |
|------------:|---------|------:|------:|------:|------:|------:|
| 1,048,576 | `Parallel` | 4.28 ms | 1.69 ms | 6.46 ms | 2.22 ms | 6.35 ms |
| 1,048,576 | `Gpu` f64 | 11.4 ms | 2.37 ms | 7.05 ms | 3.23 ms | 6.02 ms |
| 1,048,576 | `Gpu` f32 | 12.0 ms | 4.10 ms | 9.92 ms | 3.82 ms | 5.62 ms |
| 16,777,216 | `Parallel` | 97.4 ms | 1.89 ms | 105 ms | 1.86 ms | 102 ms |
| 16,777,216 | `Gpu` f64 | 145 ms | 2.95 ms | 107 ms | 1.95 ms | 91.3 ms |
| 16,777,216 | `Gpu` f32 | 193 ms | 2.85 ms | 130 ms | 4.29 ms | 149 ms |

The first chunk arrives in 2–4 ms at either size. On the GPU the streamed total also beats
`run_f64`, because readback overlaps compute. On `Parallel` at 1 Mi points, streaming
costs about 50% more in total than `run_f64` (per-chunk overhead). Use it for latency, not
throughput, at that size.

Measured 2026-11-04 on an RTX 4070 Ti SUPER (NVIDIA 615.78.08, Vulkan) and a Ryzen 9
7950X3D, Linux 7.2.8, rustc 1.100.0-nightly (2026-09-22). The machine was **not** quiet:
1-minute load was 29 at the start and 67 at the end, and a 20-minute wait never brought
it below 4. Another process held 8.4 GB of GPU memory. Expect noise of ±10–20%.

```bash
cargo bench --bench interpolation -- stream
```

## Many series

Command batching was dropped before 1.0 and re-added for 1.3 to check that decision on a
many-series workload: `run_f64_batch` packed every series into one command encoder, one
queue submission and one read-back on the GPU (and spread the series across rayon's pool
on `Parallel`). It was measured and dropped again. The harness stays as
`benches/batch.rs`, now comparing one call per series against the same calls spread
across series with rayon (`across`), which is the CPU result below without any new API.

Measured 2026-11-04 at commit `489fe60` (branch `p5/command-batching`): splimes 1.2.0 plus
the batch API, rustc 1.100.0-nightly (2026-09-22), wgpu 30.0.1, rayon 1.12.0, criterion
0.8.2, RTX 4070 Ti SUPER on driver 615.78.08 (Vulkan), Ryzen 9 7950X3D (16 cores).
**The machine was not quiet**: the 1-minute load average was 28–53 throughout (it never
fell below 13 in a 20-minute wait beforehand), so CPU and GPU figures are both
pessimistic. Each case is `series` series of `points` millisecond grid points over
`points / 4` irregular knots, Cubic, `f64` out; three full runs, criterion medians, the
cell is the median of the three and `±` half the range across runs as a percentage of it.
Cases over 2²⁴ points in total were skipped (`SPLIMES_BATCH_MAX_POINTS`).

| Series × points | `cpu` batched | `cpu` per-call | `parallel` batched | `parallel` per-call | `gpu-f64` batched | `gpu-f64` per-call | `gpu-f32` batched | `gpu-f32` per-call |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 × 64 | 2 µs (±5%) | 2 µs (±3%) | 2 µs (±4%) | 2 µs (±1%) | 300 µs (±21%) | 243 µs (±21%) | 281 µs (±9%) | 266 µs (±6%) |
| 1 × 1024 | 25 µs (±3%) | 25 µs (±5%) | 27 µs (±3%) | 26 µs (±3%) | 301 µs (±6%) | 335 µs (±4%) | 316 µs (±7%) | 299 µs (±0%) |
| 1 × 8192 | 192 µs (±0%) | 192 µs (±1%) | 206 µs (±2%) | 203 µs (±3%) | 563 µs (±9%) | 571 µs (±4%) | 524 µs (±90%) | 470 µs (±17%) |
| 1 × 65536 | 1.54 ms (±0%) | 1.53 ms (±1%) | 1.27 ms (±4%) | 1.25 ms (±12%) | 2.73 ms (±23%) | 2.53 ms (±17%) | 2.10 ms (±9%) | 2.07 ms (±3%) |
| 16 × 64 | 26 µs (±1%) | 26 µs (±12%) | 24 µs (±18%) | 29 µs (±3%) | 482 µs (±4%) | 3.55 ms (±4%) | 466 µs (±5%) | 3.47 ms (±3%) |
| 16 × 1024 | 396 µs (±2%) | 398 µs (±2%) | 126 µs (±7%) | 421 µs (±13%) | 939 µs (±0%) | 4.64 ms (±1%) | 886 µs (±6%) | 4.81 ms (±13%) |
| 16 × 8192 | 3.16 ms (±5%) | 3.14 ms (±2%) | 660 µs (±9%) | 3.26 ms (±1%) | 4.45 ms (±8%) | 9.12 ms (±15%) | 4.44 ms (±20%) | 8.68 ms (±60%) |
| 16 × 65536 | 33.48 ms (±21%) | 33.20 ms (±12%) | 7.14 ms (±3%) | 21.37 ms (±14%) | 34.08 ms (±5%) | 36.42 ms (±9%) | 24.72 ms (±3%) | 31.95 ms (±11%) |
| 256 × 64 | 428 µs (±1%) | 427 µs (±1%) | 196 µs (±11%) | 463 µs (±1%) | 3.38 ms (±1%) | 54.40 ms (±10%) | 2.60 ms (±23%) | 56.26 ms (±29%) |
| 256 × 1024 | 6.39 ms (±2%) | 6.31 ms (±2%) | 927 µs (±4%) | 6.66 ms (±1%) | 6.85 ms (±16%) | 79.15 ms (±69%) | 7.52 ms (±15%) | 64.25 ms (±23%) |
| 256 × 8192 | 71.62 ms (±7%) | 68.31 ms (±3%) | 6.05 ms (±47%) | 71.66 ms (±4%) | 34.20 ms (±2%) | 144.31 ms (±11%) | 21.95 ms (±29%) | 152.13 ms (±6%) |
| 256 × 65536 | 555.11 ms (±4%) | 561.28 ms (±3%) | 60.25 ms (±6%) | 369.23 ms (±8%) | 246.08 ms (±18%) | 636.07 ms (±81%) | 242.97 ms (±57%) | 614.40 ms (±59%) |
| 4096 × 64 | 7.27 ms (±12%) | 7.00 ms (±7%) | 2.25 ms (±14%) | 8.16 ms (±4%) | 50.69 ms (±1%) | 1127.96 ms (±11%) | 39.34 ms (±3%) | 935.03 ms (±20%) |
| 4096 × 1024 | 106.60 ms (±2%) | 108.08 ms (±2%) | 12.05 ms (±12%) | 114.10 ms (±2%) | 100.42 ms (±23%) | 1200.25 ms (±37%) | 73.65 ms (±14%) | 1129.81 ms (±19%) |

What it shows:

- Batching the GPU is a real, repeatable win over one GPU call per series once there are
  16 or more small series: 7× at 16 × 64, 16–22× at 256 × 64, 22–24× at 4,096 × 64,
  shrinking as each series grows (1.1–1.3× at 16 × 65,536, none for a single series).
  The per-call GPU path pays a submission and a blocking read-back per series, about
  0.2–0.3 ms each.
- It never makes the GPU the right choice. The CPU spread across series (`parallel`
  batched, the same as rayon over `run_f64` calls) is faster than the batched GPU in every
  multi-series case, f32 included: 6–19× at 64–1,024 points per series, 3.5–7× at 8,192
  and 65,536. Batched GPU beats the single-threaded `cpu` loop only from 256 × 8,192
  up (2.3–3.1×) and at 4,096 × 1,024 (1.5×).
  The cost the GPU doesn't remove is the one this file already names under "Backends":
  building each result's output on the host, and the upload of each series' grid times.
- So the API added nothing a caller can't get with `jobs.par_iter().map(|j| interpolator.run_f64(..))`,
  and `Backend::Auto` would never have sent batches to the GPU on this machine. Dropped again;
  revisit only if output assembly moves off the host (GPU-resident results).

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
The tables both release runs printed, 1.0.0's and 1.1.0's, identical, are
[below](#ci-runners).

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

**f32 at the gap limit.** The `f32` kernel computes every window whose knot gaps are
within 1,024× of the series' mean spacing and hands the rest to `f64`. With one-second
gaps 1,000× below the mean among sparse ones, it computed every window itself (no point
recomputed), worst 1.2e-7 (cubic), 1.6e-7 (degree 5) and 1.1e-7 (degree 8): the limit
leaves the bound about 60× headroom. At 1,100× those windows went to `f64`, as designed
(`tests/contract.rs`, `f32_windows_at_the_gap_limit_meet_the_bound`; RTX 4070 Ti SUPER,
2026-10-08).

### CI runners

From the `v1.0.0` release run, on GitHub's hosted runners; the `v1.1.0` release run
measured the same figures, digit for digit. These are software and virtualised
adapters, not GPUs you would deploy on, but they run each platform's own shader
compiler: Metal, for one, compiles shaders with fast math, which is what 1.0's integer
time representation defends against. Of the three, only lavapipe has f64; its f64 table
matches the CPU table above digit for digit.

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
